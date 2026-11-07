#!/usr/bin/env python3
"""와트투비트 프로토타입 오케스트레이터.

한 프로세스 안에 들어 있는 것:
  1) 태양광·가정 시뮬레이터 (발전, 부하, 배터리, 출력제한 상태, 온수 탱크)
  2) 노드 제어 (잉여전력 → 가동/감속/정지, 히스테리시스)
  3) 작업 큐/스케줄러 (노드가 poll 로 가져감, 이탈 시 재배정)
  4) 웹 대시보드 + 제어 API   → http://127.0.0.1:8700

표준 라이브러리만 사용. 실행: python3 orchestrator.py [--backend mock|llama]
"""
import argparse, json, math, os, subprocess, sys, threading, time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
TICK = 0.25          # 실제 초
HEARTBEAT_DEAD = 2.5 # 이 시간 poll 이 없으면 노드 이탈로 간주

# ---- 가상 노드/가정 프로파일 (실측값이 아니라 모사용 가정값) ----
PROFILES = [
    dict(id="A", name="A동 · 5090급", gpu="5090급", speed=1.00, active_kw=0.70, idle_kw=0.10, solar_kw=5.0, batt_kwh=8.0),
    dict(id="B", name="B동 · 3060급", gpu="3060급", speed=0.45, active_kw=0.22, idle_kw=0.05, solar_kw=3.0, batt_kwh=8.0),
    dict(id="C", name="C동 · 3060급", gpu="3060급", speed=0.45, active_kw=0.22, idle_kw=0.05, solar_kw=4.0, batt_kwh=8.0),
    dict(id="D", name="D동 · 1080급", gpu="1080급", speed=0.35, active_kw=0.23, idle_kw=0.05, solar_kw=3.0, batt_kwh=8.0),
]
TANK_L, TANK_MAX_C, TANK_RESUME_C = 120.0, 65.0, 55.0
SAMPLE_PROMPTS = [
    "다음 문장을 한 줄로 요약해줘: 낮에 태양광이 만든 전기가 전력망에 들어가지 못해 버려지는 일이 늘고 있다.",
    "다음 문장을 한 줄로 요약해줘: 서버에서 나온 열을 온수와 난방으로 돌려주면 에너지를 두 번 쓸 수 있다.",
    "다음 문장의 감정을 한 단어로 말해줘: 오늘 날씨가 맑아서 기분이 좋다.",
    "다음 문장을 영어로 번역해줘: 남는 전기를 그 자리에서 연산으로 바꾼다.",
]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def solar_curve(h):
    """맑은 날 발전 비율(0~1). 6시 일출, 18시 일몰."""
    if h < 6 or h > 18:
        return 0.0
    return math.sin(math.pi * (h - 6) / 12.0) ** 1.5


def load_kw(h):
    bump = lambda c, w, a: a * math.exp(-((h - c) ** 2) / (2 * w * w))
    return 0.35 + bump(7.5, 1.0, 0.9) + bump(20.0, 1.6, 1.3)


class Node:
    def __init__(self, p):
        self.p = p
        self.id = p["id"]
        self.proc = None
        self.alive = False
        self.last_seen = 0.0
        self.state = "OFF"       # OFF / RUN / HEAT_FULL / LOST
        self.reason = "시작 전"
        self.throttle = 1.0
        self.since = time.time()
        self.hold_start = None
        self.low_since = None
        self.inflight = {}
        self.done = 0
        self.tokens = 0
        self.busy = False
        # 가정 상태
        self.soc = 0.5
        self.avail_kw = 0.0
        self.solar_now = 0.0
        self.load_now = 0.0
        self.curtailed = False
        self.used_kw = 0.0
        self.tank_c = 45.0
        self.heat_kwh = 0.0
        self.surplus_kwh = 0.0   # 발생한 잉여(출력제한) 전력량
        self.used_kwh = 0.0      # 그중 연산에 쓴 양
        self.grid_kwh = 0.0      # 잉여를 넘어 쓴 양(0 이어야 정상)


class World:
    def __init__(self, backend, llama_url, unit_ms):
        self.lock = threading.RLock()
        self.backend, self.llama_url, self.unit_ms = backend, llama_url, unit_ms
        self.nodes = {p["id"]: Node(p) for p in PROFILES}
        self.hour = 9.0
        self.day = 1
        self.speed = 0.15          # 시뮬레이션 시간/실제 초
        self.paused = False
        self.solar_scale = 1.0
        self.cloud_until = 0.0
        self.auto_feed = False
        self.queue = deque()
        self.unit_seq = 0
        self.events = deque(maxlen=14)
        self.done_times = deque(maxlen=400)
        self.tok_times = deque(maxlen=400)
        self.total_done = 0
        self.requeued = 0
        self.t_last = time.time()
        self.log("시뮬레이션 준비 완료. 노드를 시작합니다.")

    # ---------------- 이벤트 ----------------
    def log(self, msg):
        hh = int(self.hour) % 24
        mm = int((self.hour % 1) * 60)
        self.events.appendleft(f"[{hh:02d}:{mm:02d}] {msg}")

    # ---------------- 노드 프로세스 ----------------
    def spawn(self, nid):
        n = self.nodes[nid]
        if n.proc and n.proc.poll() is None:
            return
        cmd = [sys.executable, os.path.join(HERE, "node.py"), "--id", nid, "--orch", f"http://127.0.0.1:{PORT}",
               "--speed", str(n.p["speed"]), "--backend", self.backend, "--llama-url", self.llama_url,
               "--unit-ms", str(self.unit_ms)]
        n.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        n.state, n.reason, n.since = "OFF", "프로세스 시작", time.time()
        n.last_seen = time.time()
        n.alive = True

    def kill(self, nid):
        n = self.nodes[nid]
        if n.proc and n.proc.poll() is None:
            n.proc.kill()
            self.log(f"노드 {nid} 강제 종료(장애 주입)")

    # ---------------- 작업 ----------------
    def submit(self, count):
        with self.lock:
            for _ in range(count):
                self.unit_seq += 1
                self.queue.append({"id": self.unit_seq, "prompt": SAMPLE_PROMPTS[self.unit_seq % len(SAMPLE_PROMPTS)]})
        self.log(f"작업 {count}건 접수 (대기 {len(self.queue)})")

    def ping(self, body):
        with self.lock:
            n = self.nodes.get(body.get("id"))
            if n:
                n.last_seen = time.time()

    def poll(self, body):
        nid = body.get("id")
        with self.lock:
            n = self.nodes.get(nid)
            if not n:
                return {"state": "OFF"}
            n.last_seen = time.time()
            if not n.alive:
                n.alive = True
                n.state, n.reason, n.since = "OFF", "재접속", time.time()
                self.log(f"노드 {nid} 재접속")
            for r in body.get("results", []):
                u = n.inflight.pop(r["unit_id"], None)
                if u is not None and r.get("ok"):
                    n.done += 1
                    n.tokens += int(r.get("tokens", 0))
                    self.total_done += 1
                    now = time.time()
                    self.done_times.append(now)
                    self.tok_times.append((now, int(r.get("tokens", 0))))
                elif u is not None:
                    self.queue.appendleft(u)
            unit = None
            if n.state == "RUN" and not n.inflight and self.queue:
                unit = self.queue.popleft()
                n.inflight[unit["id"]] = unit
            n.busy = bool(n.inflight)
            return {"state": n.state, "throttle": round(n.throttle, 3), "unit": unit}

    # ---------------- 틱 ----------------
    def tick(self):
        now = time.time()
        dt_real = now - self.t_last
        self.t_last = now
        with self.lock:
            # 프로세스 이탈 감지 + 작업 재배정
            for n in self.nodes.values():
                dead = (n.proc is not None and n.proc.poll() is not None) or (now - n.last_seen > HEARTBEAT_DEAD)
                if n.alive and dead and n.proc is not None:
                    n.alive = False
                    n.state, n.reason, n.since = "LOST", "노드 이탈", now
                    if n.inflight:
                        k = len(n.inflight)
                        for u in n.inflight.values():
                            self.queue.appendleft(u)
                        n.inflight.clear()
                        self.requeued += k
                        self.log(f"노드 {n.id} 이탈 → 작업 {k}건 재배정")
                    else:
                        self.log(f"노드 {n.id} 이탈")
            if self.auto_feed and len(self.queue) < 6:
                self.submit(8)

            dt_h = 0.0 if self.paused else dt_real * self.speed
            if not self.paused:
                self.hour += dt_h
                if self.hour >= 24:
                    self.hour -= 24
                    self.day += 1
            cloud = 0.2 if now < self.cloud_until else 1.0
            h = self.hour
            for n in self.nodes.values():
                self._household(n, h, dt_h, cloud, now)

    def _household(self, n, h, dt_h, cloud, now):
        p = n.p
        n.solar_now = p["solar_kw"] * solar_curve(h) * cloud * self.solar_scale
        n.load_now = load_kw(h)
        net = n.solar_now - n.load_now
        avail = 0.0
        eff = 0.95
        if net > 0:
            room = (1 - n.soc) * p["batt_kwh"]
            if dt_h > 0:
                chg = min(net * dt_h * eff, room)
                absorbed = chg / (dt_h * eff)
                n.soc = clamp(n.soc + chg / p["batt_kwh"], 0, 1)
            else:  # 일시정지: 상태를 바꾸지 않고 현재 판단만 유지
                absorbed = net if n.soc < 0.985 else 0.0
            avail = max(0.0, net - absorbed)
        else:
            n.soc = clamp(n.soc - (-net) * dt_h / eff / p["batt_kwh"], 0, 1)
        n.avail_kw = avail
        n.curtailed = avail > 0.05 and n.soc >= 0.985

        # 노드 제어 (실시간 기준 히스테리시스)
        act, idl = p["active_kw"], p["idle_kw"]
        # 전력 한도(power limit): 잉여 안에서만 쓰도록 연산 강도를 정함
        thr = clamp((avail - idl) / (act - idl), 0.0, 1.0) if act > idl else 0.0
        if n.state in ("RUN",):
            n.throttle = max(thr, 0.0)
            n.low_since = (n.low_since or now) if thr < 0.25 else None
            if n.tank_c >= TANK_MAX_C:
                n.state, n.reason, n.since = "HEAT_FULL", "온수 가득(열 환원 불가)", now
                self.log(f"노드 {n.id} 정지: 온수 탱크 {n.tank_c:.0f}°C")
            elif n.low_since and now - n.low_since > 1.0 and now - n.since > 4.0:
                n.state, n.reason, n.since = "OFF", "잉여전력 부족", now
                n.hold_start = None
                self.log(f"노드 {n.id} 정지: 잉여 {avail:.2f}kW")
        elif n.state in ("OFF", "HEAT_FULL") and n.alive:
            if n.state == "HEAT_FULL" and n.tank_c > TANK_RESUME_C:
                pass
            elif thr >= 0.5 and n.curtailed:
                n.hold_start = n.hold_start or now
                if now - n.hold_start > 1.5:
                    n.state, n.reason, n.since = "RUN", "출력제한 감지 → 가동", now
                    n.throttle, n.low_since = thr, None
                    self.log(f"노드 {n.id} 가동 (잉여 {avail:.2f}kW)")
            else:
                n.hold_start = None
                if n.state == "OFF":
                    n.reason = "잉여 없음" if not n.curtailed else "대기"

        # 전력/열 계산
        if n.state == "RUN":
            n.used_kw = p["idle_kw"] + (act - p["idle_kw"]) * n.throttle if n.busy else p["idle_kw"]
        else:
            n.used_kw = 0.0
        if dt_h > 0:
            n.surplus_kwh += avail * dt_h
            n.used_kwh += min(n.used_kw, avail) * dt_h
            n.grid_kwh += max(0.0, n.used_kw - avail) * dt_h
            n.heat_kwh += n.used_kw * dt_h
            # 온수 탱크: 열 유입 - 손실, 7시·21시 사용(온수 60L)
            mass = TANK_L
            n.tank_c += (n.used_kw * 3600 * dt_h) / (mass * 4.186)
            n.tank_c -= (n.tank_c - 20.0) * 0.012 * dt_h
            for use_h in (7.0, 21.0):
                prev = (h - dt_h) % 24
                if (prev < use_h <= h) or (prev > h and (use_h > prev or use_h <= h)):
                    f = 40.0 / mass
                    n.tank_c = n.tank_c * (1 - f) + 15.0 * f
        if n.state == "HEAT_FULL" and n.tank_c <= TANK_RESUME_C:
            n.state, n.reason, n.since = "OFF", "온수 사용으로 여유 생김", now

    # ---------------- 상태 ----------------
    def state(self):
        with self.lock:
            now = time.time()
            thr10 = sum(1 for t in self.done_times if now - t <= 10) / 10.0
            tok10 = sum(k for t, k in self.tok_times if now - t <= 10) / 10.0
            curve = [round(solar_curve(x / 2.0), 3) for x in range(49)]
            return {
                "hour": round(self.hour, 3), "day": self.day, "speed": self.speed, "paused": self.paused,
                "solar_scale": self.solar_scale, "cloud": now < self.cloud_until, "auto_feed": self.auto_feed,
                "backend": self.backend, "curve": curve,
                "queue": len(self.queue), "inflight": sum(len(n.inflight) for n in self.nodes.values()),
                "total_done": self.total_done, "requeued": self.requeued,
                "units_per_s": round(thr10, 2), "tokens_per_s": round(tok10, 1),
                "events": list(self.events),
                "nodes": [{
                    "id": n.id, "name": n.p["name"], "gpu": n.p["gpu"], "speed": n.p["speed"], "alive": n.alive,
                    "state": n.state, "reason": n.reason, "throttle": round(n.throttle, 2),
                    "solar": round(n.solar_now, 2), "load": round(n.load_now, 2), "soc": round(n.soc, 3),
                    "avail": round(n.avail_kw, 2), "curtailed": n.curtailed, "used": round(n.used_kw, 2),
                    "tank": round(n.tank_c, 1), "done": n.done, "inflight": len(n.inflight),
                    "heat_kwh": round(n.heat_kwh, 3), "surplus_kwh": round(n.surplus_kwh, 3),
                    "used_kwh": round(n.used_kwh, 3), "grid_kwh": round(n.grid_kwh, 4),
                    "active_kw": n.p["active_kw"],
                } for n in self.nodes.values()],
            }

    def control(self, c):
        a = c.get("action")
        with self.lock:
            if a == "speed":
                self.speed = clamp(float(c["value"]), 0.01, 3.0)
            elif a == "pause":
                self.paused = bool(c["value"])
            elif a == "solar_scale":
                self.solar_scale = clamp(float(c["value"]), 0.0, 1.5)
                self.log(f"태양광 세기 {self.solar_scale*100:.0f}%로 조정")
            elif a == "cloud":
                self.cloud_until = time.time() + 12
                self.log("구름 이벤트: 발전량 20%로 급감 (12초)")
            elif a == "hour":
                self.hour = clamp(float(c["value"]), 0, 23.99)
            elif a == "submit":
                self.submit(int(c.get("value", 20)))
            elif a == "auto_feed":
                self.auto_feed = bool(c["value"])
            elif a == "kill":
                self.kill(c["id"])
            elif a == "restart":
                self.spawn(c["id"])
                self.log(f"노드 {c['id']} 재시작")
            elif a == "reset":
                for n in self.nodes.values():
                    n.done = n.tokens = 0
                    n.surplus_kwh = n.used_kwh = n.grid_kwh = n.heat_kwh = 0.0
                    n.tank_c, n.soc = 45.0, 0.5
                self.total_done = self.requeued = 0
                self.queue.clear()
                self.log("통계 초기화")


WORLD = None
PORT = 8700


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/state"):
            self._send(200, json.dumps(WORLD.state(), ensure_ascii=False))
        elif self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                self._send(200, f.read(), "text/html")
        else:
            self._send(404, "{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/node/poll":
            self._send(200, json.dumps(WORLD.poll(body)))
        elif self.path == "/node/ping":
            WORLD.ping(body)
            self._send(200, "{}")
        elif self.path == "/control":
            WORLD.control(body)
            self._send(200, "{}")
        else:
            self._send(404, "{}")


def main():
    global WORLD, PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "llama"], default="mock")
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--unit-ms", type=int, default=400, help="mock 백엔드 작업 1건의 CPU 시간(ms)")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--nodes", type=int, default=4, help="가동할 노드 수(1~4)")
    a = ap.parse_args()
    PORT = a.port
    WORLD = World(a.backend, a.llama_url, a.unit_ms)
    for p in PROFILES[a.nodes:]:
        WORLD.nodes.pop(p["id"])
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for nid in WORLD.nodes:
        WORLD.spawn(nid)
    print(f"대시보드: http://127.0.0.1:{PORT}   (백엔드: {a.backend}, 종료: Ctrl+C)", flush=True)
    try:
        while True:
            WORLD.tick()
            time.sleep(TICK)
    except KeyboardInterrupt:
        pass
    finally:
        for n in WORLD.nodes.values():
            if n.proc and n.proc.poll() is None:
                n.proc.kill()


if __name__ == "__main__":
    main()
