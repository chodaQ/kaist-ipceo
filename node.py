#!/usr/bin/env python3
"""와트투비트 가상 GPU 노드 (프로세스 1개 = 가정 1곳).

오케스트레이터에 주기적으로 poll 하면서
  - 가동 명령(state, throttle)을 받고
  - 가동 중일 때만 작업 단위(unit)를 받아 실제로 연산한 뒤 결과를 보고한다.
연산 백엔드:
  mock  : CPU 에서 실제 계산 루프를 돌림 (모델 없이 테스트용)
  llama : 같은 PC 의 llama-server(/completion)에 실제 추론 요청
GPU 속도 차이는 --speed 로 흉내 낸다(1.0 = 이 PC 의 실제 속도, 낮을수록 느림).
"""
import argparse, hashlib, json, os, sys, threading, time, urllib.request, urllib.error


def post(url, obj, timeout=5):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_mock(unit_ms):
    """실제 CPU 연산(해시 반복)을 unit_ms 만큼 수행."""
    end = time.perf_counter() + unit_ms / 1000.0
    x = b"watt2bit"
    n = 0
    while time.perf_counter() < end:
        for _ in range(2000):
            x = hashlib.sha256(x).digest()
        n += 2000
    return {"tokens": n // 1000}


def run_llama(prompt, url, n_predict):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.2, "cache_prompt": True}
    r = post(url.rstrip("/") + "/completion", body, timeout=120)
    return {"tokens": int(r.get("tokens_predicted", 0)), "text": (r.get("content") or "")[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--orch", default="http://127.0.0.1:8700")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--backend", choices=["mock", "llama"], default="mock")
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--unit-ms", type=int, default=400)
    ap.add_argument("--n-predict", type=int, default=32)
    a = ap.parse_args()

    def beat():
        while True:  # 연산 중에도 살아 있음을 알림(heartbeat)
            try:
                post(a.orch + "/node/ping", {"id": a.id}, timeout=2)
            except Exception:
                pass
            time.sleep(0.5)
    threading.Thread(target=beat, daemon=True).start()

    results = []
    state, throttle = "OFF", 1.0
    print(f"[node {a.id}] pid={os.getpid()} backend={a.backend} speed={a.speed}", flush=True)
    while True:
        try:
            resp = post(a.orch + "/node/poll", {"id": a.id, "results": results, "pid": os.getpid()})
            results = []
        except Exception as e:
            print(f"[node {a.id}] orchestrator 연결 실패: {e}", flush=True)
            time.sleep(1.0)
            continue
        state, throttle = resp.get("state", "OFF"), float(resp.get("throttle", 1.0))
        unit = resp.get("unit")
        if state != "RUN" or not unit:
            time.sleep(0.25)
            continue
        t0 = time.perf_counter()
        try:
            if a.backend == "llama":
                out = run_llama(unit["prompt"], a.llama_url, a.n_predict)
            else:
                out = run_mock(a.unit_ms)
            ok = True
        except Exception as e:  # llama-server 미기동 등
            out, ok = {"tokens": 0, "error": str(e)[:80]}, False
        busy = time.perf_counter() - t0
        # GPU 등급(속도)과 전력 제한(throttle)을 반영: 같은 일을 더 느리게
        eff = max(0.05, min(1.0, a.speed) * max(0.05, throttle))
        time.sleep(busy * (1.0 / eff - 1.0))
        total_ms = (time.perf_counter() - t0) * 1000
        results.append({"unit_id": unit["id"], "ok": ok, "ms": round(total_ms), **out})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
