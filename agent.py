#!/usr/bin/env python3
"""와트투비트 실제 노드 에이전트 (NVIDIA GPU 가 달린 PC 에서 실행).

하는 일
  1) nvidia-smi 로 GPU 를 실측 (전력·온도·클럭·활용률·VRAM, 0.5초 간격)
  2) 오케스트레이터가 내려준 전력 한도를 실제로 적용
       pl   : nvidia-smi -pl  (데스크톱/서버 GPU, 관리자 권한 필요)
       duty : 요청 사이에 쉬는 시간을 넣어 평균 전력을 맞춤 (노트북처럼 -pl 이 안 될 때)
       auto : pl 이 되면 pl, 안 되면 duty   (기본)
     한도가 GPU 최소 한도보다 낮으면 pl + duty 를 같이 씀
  3) 같은 PC 의 llama-server(CUDA) 에 실제 추론 요청
  4) 실측 텔레메트리와 결과를 오케스트레이터에 보고

표준 라이브러리만 사용 (Linux / Windows).

예)
  python3 agent.py --orch http://192.168.0.10:8700 --id SRV --system-w 90 \\
      --llama-bin llama-server --gguf ~/models/qwen2.5-7b-instruct-q4_k_m.gguf
  python3 agent.py --orch http://192.168.0.10:8700 --id SRV --sim-gpu 3080   # GPU 없이 테스트
"""
import argparse, atexit, json, os, platform, shutil, signal, subprocess, sys, threading, time
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vgpu  # noqa: E402

QUERY = ("name,memory.total,memory.used,power.draw,power.limit,power.default_limit,"
         "power.min_limit,power.max_limit,temperature.gpu,clocks.sm,clocks.max.sm,utilization.gpu")


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def http_json(url, obj=None, timeout=5, token=None):
    data = json.dumps(obj).encode() if obj is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    if token:
        req.add_header("X-W2B-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None   # "[N/A]", "[Not Supported]"


# ====================================================================== 실제 GPU
class NvidiaGPU:
    def __init__(self, index, power_mode, smi="nvidia-smi"):
        self.idx = str(index)
        self.smi = shutil.which(smi) or smi
        self.lock = threading.Lock()
        self.cur = self._query_once()
        c = self.cur
        self.name = c["name"]
        self.vram_total_gb = round((c["mem_total"] or 0) / 1024, 2)
        self.default_pl = c["pl_default"] or c["pl"] or c["pl_max"]
        self.orig_pl = c["pl"]
        self.min_pl, self.max_pl = c["pl_min"], c["pl_max"]
        self.tdp_w = self.default_pl or self.max_pl or 100.0   # 노트북(N/A)은 실측 최대값으로 갱신
        self.peak_w = 0.0
        self.idle_w = None                      # 첫 몇 초 측정으로 채움
        self.target_w = self.tdp_w
        self.applied_pl = self.orig_pl
        self._last_pl_set = 0.0
        self.reason = "IDLE"
        # 텔레메트리 적분
        self.energy_j = 0.0
        self._win = [0.0, 0.0, 0.0, 0]   # 시간, 에너지, util 적분, 토큰
        self._samples = []
        self.mode = self._pick_mode(power_mode)
        self._stop = False
        self._t_last = time.time()
        threading.Thread(target=self._sampler, daemon=True).start()

    # ---------------- nvidia-smi
    def _parse(self, line):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 12:
            return None
        return {"name": f[0], "mem_total": num(f[1]), "mem_used": num(f[2]), "power": num(f[3]),
                "pl": num(f[4]), "pl_default": num(f[5]), "pl_min": num(f[6]), "pl_max": num(f[7]),
                "temp": num(f[8]), "clock": num(f[9]), "clock_max": num(f[10]), "util": num(f[11])}

    def _query_once(self):
        out = subprocess.run([self.smi, "-i", self.idx, "--query-gpu=" + QUERY, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            raise RuntimeError(f"nvidia-smi 실패: {out.stderr.strip() or out.stdout.strip()}")
        r = self._parse(out.stdout.strip().splitlines()[0])
        if not r:
            raise RuntimeError(f"nvidia-smi 출력 해석 실패: {out.stdout!r}")
        return r

    def _sampler(self):
        """nvidia-smi -lms 500 스트림을 읽어 전력을 적분."""
        while not self._stop:
            try:
                p = subprocess.Popen([self.smi, "-i", self.idx, "--query-gpu=" + QUERY,
                                      "--format=csv,noheader,nounits", "-lms", "500"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                self._smi_proc = p
                for line in p.stdout:
                    if self._stop:
                        break
                    r = self._parse(line)
                    if r:
                        self._ingest(r)
                p.kill()
            except Exception as e:
                log(f"nvidia-smi 샘플링 오류: {e}")
            time.sleep(1.0)

    def _ingest(self, r):
        now = time.time()
        with self.lock:
            dt = min(2.0, now - self._t_last)
            self._t_last = now
            pw = r["power"]
            if pw is None:      # 일부 노트북은 power.draw 미지원 → 한도 비율로 추정
                pw = (r["util"] or 0) / 100 * (self.target_w or self.tdp_w)
            r["power"] = pw
            self.cur = r
            if pw > self.peak_w:
                self.peak_w = pw
                if not self.default_pl and pw > self.tdp_w:
                    self.tdp_w = round(pw, 1)
            self.energy_j += pw * dt
            w = self._win
            w[0] += dt; w[1] += pw * dt; w[2] += (r["util"] or 0) / 100 * dt
            if self.idle_w is None:
                self._samples.append(pw)
                if len(self._samples) >= 6:
                    self.idle_w = round(min(self._samples), 1)

    def energy_now(self):
        with self.lock:
            return self.energy_j, time.time()

    # ---------------- 전력 제어
    def _set_pl(self, w):
        cmd = [self.smi, "-i", self.idx, "-pl", f"{w:.0f}"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode != 0 and os.name == "posix" and os.geteuid() != 0 and shutil.which("sudo"):
            # 리눅스 비 root: sudoers 에 nvidia-smi NOPASSWD 가 있으면 성공 (README 참고)
            out2 = subprocess.run(["sudo", "-n"] + cmd, capture_output=True, text=True, timeout=10)
            if out2.returncode == 0:
                return True, out2.stdout.strip()
        return out.returncode == 0, (out.stdout + out.stderr).strip()

    def _pick_mode(self, want):
        if want == "duty":
            return "duty"
        ok = False
        if self.min_pl and self.max_pl and self.orig_pl:
            ok, msg = self._set_pl(self.orig_pl)   # 같은 값으로 써 보며 권한 확인
            if not ok:
                log(f"전력 한도 설정 불가 → duty 방식 사용 ({msg[:100]})")
        else:
            log("이 GPU 는 전력 한도 조회/설정을 지원하지 않음 → duty 방식 사용")
        if want == "pl" and not ok:
            log("경고: --power-mode pl 을 요청했지만 불가능하여 duty 로 대체")
        return "pl" if ok else "duty"

    def set_power_limit(self, w):
        self.target_w = max(0.0, float(w))
        if self.mode != "pl":
            return
        # 공장 기본 한도 이상으로는 올리지 않음 (최대 한도는 오버클럭 영역)
        want = max(self.min_pl, min(self.default_pl or self.max_pl, self.max_pl, self.target_w))
        now = time.time()
        if self.applied_pl is not None and abs(want - self.applied_pl) < 5:
            return
        if now - self._last_pl_set < 2.0:
            return
        ok, msg = self._set_pl(want)
        self._last_pl_set = now
        if ok:
            self.applied_pl = want
        else:
            log(f"nvidia-smi -pl {want:.0f} 실패 → duty 로 전환: {msg[:100]}")
            self.mode = "duty"

    def restore(self):
        self._stop = True
        if self.mode == "pl" and self.orig_pl and self.applied_pl != self.orig_pl:
            ok, _ = self._set_pl(self.orig_pl)
            log(f"전력 한도 원복 {self.orig_pl:.0f}W " + ("완료" if ok else "실패"))
        p = getattr(self, "_smi_proc", None)
        if p:
            p.kill()

    # ---------------- 보고
    def info(self):
        return {"kind": "real", "gpu": self.name, "tdp_w": self.tdp_w, "idle_w": self.idle_w or 10.0,
                "min_pl_w": self.min_pl, "vram_total_gb": self.vram_total_gb, "power_mode": self.mode}

    def telemetry(self, model_name):
        with self.lock:
            c, w = self.cur, self._win
            avg_p = w[1] / w[0] if w[0] > 0 else (c["power"] or 0)
            util = w[2] / w[0] if w[0] > 0 else (c["util"] or 0) / 100
            tps = w[3] / w[0] if w[0] > 0 else 0.0
            self._win = [0.0, 0.0, 0.0, 0]
        return {"gpu": self.name, "clock_mhz": int(c["clock"] or 0), "temp_c": c["temp"] or 0.0,
                "power_w": round(avg_p, 1), "power_limit_w": round(self.target_w, 1),
                "hw_limit_w": c["pl"], "tdp_w": self.tdp_w, "util": round(util, 3), "tok_s": round(tps, 1),
                "reason": self.reason, "vram_used_gb": round((c["mem_used"] or 0) / 1024, 2),
                "vram_total_gb": self.vram_total_gb, "energy_wh": round(self.energy_j / 3600, 4),
                "model": model_name, "power_mode": self.mode, "measured": True}

    def add_tokens(self, n):
        with self.lock:
            self._win[3] += n


# ====================================================================== 시뮬 GPU (하드웨어 없이 테스트)
class SimGPU:
    def __init__(self, key, model):
        self.g = vgpu.VirtualGPU(key, realtime=True)
        self.g.load_model(model)
        self.mode = "sim"
        self.name = self.g.s["name"]
        self._last = time.perf_counter()

    def tick_idle(self):
        now = time.perf_counter()
        self.g.idle(now - self._last)
        self._last = now

    def info(self):
        s = self.g.s
        return {"kind": "sim", "gpu": s["name"], "tdp_w": s["tdp_w"], "idle_w": s["idle_w"],
                "min_pl_w": s["min_pl_w"], "vram_total_gb": s["vram_gb"], "power_mode": "sim"}

    def set_power_limit(self, w):
        self.g.set_power_limit(w)

    def telemetry(self, model_name):
        t = self.g.telemetry()
        t["measured"] = False
        return t

    def infer(self, prompt, n_predict):
        self.tick_idle()
        r = self.g.infer(vgpu.estimate_tokens(prompt), n_predict)
        self._last = time.perf_counter()
        return r

    def restore(self):
        pass


# ====================================================================== llama-server
class Llama:
    def __init__(self, url, bin_=None, gguf=None, ngl=99, ctx=4096, port=8080, extra="", ready_timeout=180):
        self.url = url.rstrip("/")
        self.proc = None
        self.model_name = os.path.basename(gguf) if gguf else None
        if bin_ and gguf:
            cmd = [bin_, "-m", os.path.expanduser(gguf), "-ngl", str(ngl), "-c", str(ctx),
                   "--host", "127.0.0.1", "--port", str(port)] + (extra.split() if extra else [])
            log("llama-server 시작: " + " ".join(cmd))
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            self.url = f"http://127.0.0.1:{port}"
        self.wait_ready(ready_timeout if (bin_ and gguf) else min(ready_timeout, 30))

    def wait_ready(self, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError("llama-server 가 바로 종료됨 (모델 경로·CUDA 빌드 확인)")
            try:
                http_json(self.url + "/health", timeout=2)
                try:
                    props = http_json(self.url + "/props", timeout=2)
                    mp = props.get("model_path") or ""
                    if mp:
                        self.model_name = os.path.basename(mp)
                except Exception:
                    pass
                log(f"llama-server 준비 완료: {self.url} ({self.model_name or '모델명 미확인'})")
                return
            except urllib.error.HTTPError as e:
                if e.code != 503:   # 503 = 모델 로딩 중
                    raise
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError(f"llama-server({self.url}) 응답 없음")

    def complete(self, prompt, n_predict):
        body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.2, "cache_prompt": True}
        r = http_json(self.url + "/completion", body, timeout=300)
        tm = r.get("timings") or {}
        return {"tokens": int(r.get("tokens_predicted", tm.get("predicted_n", 0)) or 0),
                "prompt_tokens": int(tm.get("prompt_n", 0) or 0),
                "decode_tps": round(float(tm.get("predicted_per_second", 0) or 0), 1),
                "prefill_ms": round(float(tm.get("prompt_ms", 0) or 0), 1),
                "text": (r.get("content") or "").strip()[:120]}

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except Exception:
                self.proc.kill()


# ====================================================================== 메인
def run_real_unit(gpu, llama, unit, n_predict):
    """실제 추론 1건 + 전력 한도에 맞춘 duty 휴지."""
    e0, t0 = gpu.energy_now()
    out = llama.complete(unit["prompt"], n_predict)
    time.sleep(0.6)                       # 샘플러가 마지막 구간을 적분하도록
    e1, t1 = gpu.energy_now()
    dt = max(0.05, t1 - t0)
    p_run = (e1 - e0) / dt
    gpu.add_tokens(out["tokens"])
    out["energy_j"] = round(e1 - e0, 1)
    out["avg_w"] = round(p_run, 1)
    lim, idle = gpu.target_w, (gpu.idle_w or 10.0)
    need_duty = gpu.mode == "duty" or (gpu.min_pl and lim < gpu.min_pl)
    if need_duty and p_run > lim:
        if lim <= idle + 1:
            gpu.reason = "DUTY"
            return out, max(1.0, dt)      # 한도가 유휴 이하: 최대한 쉼
        t_off = dt * (p_run - lim) / (lim - idle)
        gpu.reason = "DUTY"
        return out, min(t_off, 30.0)
    gpu.reason = "PWR_CAP" if (gpu.mode == "pl" and gpu.applied_pl and gpu.applied_pl < (gpu.default_pl or 0) - 5) else "NONE"
    return out, 0.0


def preflight(a):
    """--check: 실제 기기에서 첫 실행 전 점검. 오케스트레이터에 작업을 요청하지 않음."""
    ok_all = True

    def item(ok, title, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok) or ok is None
        mark = "OK  " if ok else ("주의" if ok is None else "실패")
        print(f"[{mark}] {title}" + (f" — {detail}" if detail else ""), flush=True)

    print(f"== 와트투비트 노드 점검 ({platform.system()} {platform.release()}, Python {platform.python_version()}) ==")
    gpu = None
    try:
        gpu = NvidiaGPU(a.gpu_index, a.power_mode)
        c = gpu.cur
        item(True, "nvidia-smi", f"{gpu.name}, VRAM {gpu.vram_total_gb}GB, {c['temp']}°C, {c['clock']}MHz")
        item(c["power"] is not None or None, "전력 측정(power.draw)",
             f"{c['power']} W" if c["power"] is not None else "미지원 → 활용률로 추정 (정확도 낮음)")
        item(True if gpu.mode == "pl" else None, "전력 한도 제어",
             f"nvidia-smi -pl 가능 ({gpu.min_pl:.0f}~{gpu.default_pl:.0f}W)" if gpu.mode == "pl"
             else "duty 방식 사용 (노트북은 정상)")
        time.sleep(2.5)
        with gpu.lock:
            n_samp = gpu._win[0]
        item(n_samp > 1.0, "nvidia-smi -lms 스트리밍", f"{n_samp:.1f}초 분량 수신")
        if "laptop" in gpu.name.lower():
            item(None, "노트북 전원", "충전기 연결 + 전원 모드 '최고 성능'(Windows) / performance 프로파일(Linux) 권장")
    except Exception as e:
        item(False, "nvidia-smi", str(e)[:150])

    llama = None
    try:
        llama = Llama(a.llama_url, a.llama_bin, a.gguf, a.ngl, a.ctx, a.llama_port, a.llama_extra, ready_timeout=180 if a.llama_bin else 10)
        item(True, "llama-server", f"{llama.url} · {llama.model_name}")
        e0 = gpu.energy_now()[0] if gpu else 0
        t0 = time.time()
        r = llama.complete("다음 문장을 한 줄로 요약해줘: 남는 태양광 전기를 연산으로 바꾼다.", 32)
        time.sleep(0.6)
        dt = time.time() - t0
        pw = (gpu.energy_now()[0] - e0) / dt if gpu else 0
        item(r["tokens"] > 0, "실제 추론", f"{r['tokens']}토큰, 디코드 {r['decode_tps']} tok/s, 평균 {pw:.0f}W, 출력: {r['text'][:40]!r}")
        if r["decode_tps"] == 0:
            item(None, "llama timings", "응답에 timings 가 없음 (llama.cpp 버전 확인)")
        if gpu:
            with gpu.lock:
                used = (gpu.cur["mem_used"] or 0) / 1024
            item(used > 1.0 or None, "GPU 오프로드", f"VRAM 사용 {used:.1f}GB" + ("" if used > 1.0 else " — 모델이 GPU 에 안 올라갔을 수 있음(-ngl, CUDA 빌드 확인)"))
    except Exception as e:
        item(False, "llama-server", str(e)[:150])
    finally:
        if llama:
            llama.close()

    orch = a.orch.rstrip("/")
    try:
        http_json(orch + "/state", timeout=4)
        item(True, "오케스트레이터 접속", orch)
        try:
            http_json(orch + "/node/ping", {"id": "__check__"}, timeout=4, token=a.token)
            item(True, "토큰", "통과" if a.token else "토큰 없이 허용됨")
        except urllib.error.HTTPError as e:
            item(False, "토큰", f"HTTP {e.code} — --token 값 확인")
    except Exception as e:
        item(False, "오케스트레이터 접속", f"{orch}: {e} (IP·방화벽·같은 네트워크인지 확인)")

    if gpu:
        gpu.restore()
    print("== 결과: " + ("모두 통과. --check 를 빼고 실행하세요." if ok_all else "실패 항목을 먼저 해결하세요.") + " ==")
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser(description="와트투비트 실제 노드 에이전트")
    ap.add_argument("--orch", required=True, help="오케스트레이터 주소, 예: http://192.168.0.10:8700")
    ap.add_argument("--id", required=True, help="노드 이름(짧게), 예: LAP1, LAP2, SRV")
    ap.add_argument("--token", default=os.environ.get("W2B_TOKEN"), help="오케스트레이터 --token 과 같은 값")
    ap.add_argument("--system-w", type=float, default=None, help="GPU 를 뺀 나머지 PC 소비전력(W). 노트북≈25, 서버≈80")
    ap.add_argument("--solar-kw", type=float, default=None, help="이 노드에 연결된 가상 태양광 용량(kW)")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--power-mode", choices=["auto", "pl", "duty"], default="auto")
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080", help="이미 떠 있는 llama-server 주소")
    ap.add_argument("--llama-bin", default=None, help="지정하면 에이전트가 llama-server 를 직접 띄움")
    ap.add_argument("--gguf", default=None, help="--llama-bin 과 함께: 모델 파일 경로")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--llama-port", type=int, default=8080)
    ap.add_argument("--llama-extra", default="", help="llama-server 에 넘길 추가 인자")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--sim-gpu", default=None, choices=list(vgpu.GPUS), help="실제 GPU 대신 가상 GPU 로 테스트")
    ap.add_argument("--sim-model", default="7B", choices=list(vgpu.MODELS))
    ap.add_argument("--check", action="store_true", help="실행 전 점검만 하고 종료 (GPU·llama-server·네트워크)")
    a = ap.parse_args()
    if a.check:
        sys.exit(preflight(a))
    orch = a.orch.rstrip("/")

    llama = None
    if a.sim_gpu:
        gpu = SimGPU(a.sim_gpu, a.sim_model)
        model_name = vgpu.MODELS[a.sim_model]["name"] + " (가상)"
    else:
        gpu = NvidiaGPU(a.gpu_index, a.power_mode)
        log(f"GPU: {gpu.name} · VRAM {gpu.vram_total_gb}GB · 전력한도 {gpu.min_pl}~{gpu.max_pl}W "
            f"(현재 {gpu.orig_pl}) · 제어방식 {gpu.mode}")
        llama = Llama(a.llama_url, a.llama_bin, a.gguf, a.ngl, a.ctx, a.llama_port, a.llama_extra)
        model_name = llama.model_name or "llama-server"

    is_laptop = any(k in gpu.name.lower() for k in ("laptop", "mobile", "max-q"))
    system_w = a.system_w if a.system_w is not None else (25.0 if is_laptop else 80.0)

    def cleanup(*_):
        gpu.restore()
        if llama:
            llama.close()
    atexit.register(cleanup)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    stop_beat = threading.Event()

    def beat():
        while True:
            if not stop_beat.is_set():
                try:
                    http_json(orch + "/node/ping", {"id": a.id}, timeout=2, token=a.token)
                except Exception:
                    pass
            time.sleep(0.5)
    threading.Thread(target=beat, daemon=True).start()

    results, rest_until, connected = [], 0.0, False
    log(f"노드 {a.id} 시작 → {orch} (system_w={system_w:.0f}W)")
    while True:
        if isinstance(gpu, SimGPU):
            gpu.tick_idle()
        info = gpu.info()
        info.update({"system_w": system_w, "solar_kw": a.solar_kw, "model": model_name, "host": platform.node()})
        body = {"id": a.id, "results": results, "pid": os.getpid(), "info": info,
                "accept": time.time() >= rest_until,
                "gpu": gpu.telemetry(model_name), "error": None}
        try:
            resp = http_json(orch + "/node/poll", body, timeout=5, token=a.token)
            results = []
            if not connected:
                log("오케스트레이터 연결됨")
                connected = True
        except urllib.error.HTTPError as e:
            log(f"오케스트레이터 거부: HTTP {e.code} (토큰 확인)")
            time.sleep(3)
            continue
        except Exception as e:
            if connected:
                log(f"오케스트레이터 연결 끊김: {e}")
            connected = False
            time.sleep(1.0)
            continue

        if resp.get("cmd") == "disconnect":     # 대시보드의 '강제 이탈' (장애 주입)
            secs = float(resp.get("seconds", 15))
            log(f"장애 주입: {secs:.0f}초 동안 응답 중단 (처리 중 작업은 버림)")
            stop_beat.set()
            time.sleep(secs)
            stop_beat.clear()
            results = []
            continue

        if resp.get("power_limit_w") is not None:
            gpu.set_power_limit(resp["power_limit_w"])
        unit = resp.get("unit")
        if resp.get("state") != "RUN" or not unit:
            if resp.get("state") != "RUN" and not isinstance(gpu, SimGPU):
                gpu.reason = "IDLE"
            time.sleep(0.25)
            continue

        t0 = time.perf_counter()
        try:
            if isinstance(gpu, SimGPU):
                out, rest = gpu.infer(unit["prompt"], a.n_predict), 0.0
            else:
                out, rest = run_real_unit(gpu, llama, unit, a.n_predict)
            ok = True
        except Exception as e:
            out, rest, ok = {"tokens": 0, "error": str(e)[:100]}, 1.0, False
            log(f"작업 {unit['id']} 실패: {e}")
        ms = round((time.perf_counter() - t0) * 1000)
        results.append({"unit_id": unit["id"], "ok": ok, "ms": ms, **out})
        if rest > 0:
            rest_until = time.time() + rest


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
