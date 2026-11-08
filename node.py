#!/usr/bin/env python3
"""와트투비트 가상 GPU 노드 (프로세스 1개 = 가정 1곳).

오케스트레이터에 주기적으로 poll 하면서
  - 가동 명령(state, throttle)을 받고
  - 가동 중일 때만 작업 단위(unit)를 받아 실제로 연산한 뒤 결과를 보고한다.
연산 백엔드:
  mock  : CPU 에서 실제 계산 루프를 돌림 (모델 없이 테스트용)
  llama : 같은 PC 의 llama-server(/completion)에 실제 추론 요청
  vgpu  : 가상 GPU(vgpu.py)가 스펙·전력 한도·온도에 따라 시간·전력·열을 계산하고 그만큼 실제로 걸림
mock/llama 의 GPU 속도 차이는 --speed 로 흉내 낸다(1.0 = 이 PC 의 실제 속도, 낮을수록 느림).
"""
import argparse, hashlib, json, os, sys, threading, time, urllib.request, urllib.error
import vgpu


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
    ap.add_argument("--backend", choices=["vgpu", "mock", "llama"], default="vgpu")
    ap.add_argument("--gpu", default="3060", choices=list(vgpu.GPUS))
    ap.add_argument("--model", default="7B", choices=list(vgpu.MODELS))
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

    gpu, gpu_error = None, None
    if a.backend == "vgpu":
        gpu = vgpu.VirtualGPU(a.gpu, realtime=True)
        try:
            gpu.load_model(a.model)
        except vgpu.VGPUError as e:
            gpu_error = str(e)
    last_t = time.perf_counter()

    results = []
    state, throttle = "OFF", 1.0
    print(f"[node {a.id}] pid={os.getpid()} backend={a.backend} gpu={a.gpu if gpu else '-'} speed={a.speed}", flush=True)
    ngpu = 1   # 오케스트레이터가 정해 주는 가동 GPU 장수 (같은 GPU 여러 장을 병렬로 모사)
    while True:
        if gpu:  # 작업 없이 흐른 시간만큼 유휴 전력·냉각 반영
            now = time.perf_counter()
            gpu.idle(now - last_t)
            last_t = now
        body = {"id": a.id, "results": results, "pid": os.getpid()}
        if gpu:
            t = gpu.telemetry()
            # 같은 GPU ngpu 장이 같은 일을 나눠 한다고 보고 전력·속도를 장수만큼 합산
            t["power_w"] = round(t["power_w"] * ngpu, 1)
            t["power_limit_w"] = round(t["power_limit_w"] * ngpu, 1)
            t["tdp_w"] = t["tdp_w"] * ngpu
            t["tok_s"] = round(t["tok_s"] * ngpu, 1)
            t["gpus"] = ngpu
            body["gpu"] = t
            body["error"] = gpu_error
        try:
            resp = post(a.orch + "/node/poll", body)
            results = []
        except Exception as e:
            print(f"[node {a.id}] orchestrator 연결 실패: {e}", flush=True)
            time.sleep(1.0)
            continue
        state, throttle = resp.get("state", "OFF"), float(resp.get("throttle", 1.0))
        if gpu:
            ngpu = max(1, int(resp.get("gpus") or 1))
            if resp.get("power_limit_w") is not None:
                gpu.set_power_limit(resp["power_limit_w"] / ngpu)   # 장당 한도
        unit = resp.get("unit")
        if state != "RUN" or not unit:
            time.sleep(0.1)
            continue
        t0 = time.perf_counter()
        if gpu:
            gpu.idle(t0 - last_t)
            try:
                out, ok = gpu.infer(vgpu.estimate_tokens(unit["prompt"]), a.n_predict), True
                out["tokens"] *= ngpu            # ngpu 장이 동시에 처리
                out["energy_j"] = round(out["energy_j"] * ngpu, 2)
                out["gpus"] = ngpu
            except vgpu.VGPUError as e:
                out, ok = {"tokens": 0, "error": str(e)[:80]}, False
            last_t = time.perf_counter()
            results.append({"unit_id": unit["id"], "ok": ok, "ms": round((last_t - t0) * 1000), **out})
            continue
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
