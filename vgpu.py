#!/usr/bin/env python3
"""와트투비트 가상 GPU (소프트웨어 GPU 시뮬레이터).

실제 GPU 없이 "GPU 한 장"을 소프트웨어로 모사한다. 표준 라이브러리만 사용.

모사하는 것
  - 장치 스펙: SM 수, 클럭 범위, 연산 성능(TFLOPS), 메모리 대역폭, VRAM, TDP
  - VRAM 할당기: 모델 가중치·KV 캐시 할당, 부족하면 OOM
  - 성능: 루프라인(roofline) 모델 → 스텝 시간 = max(연산시간, 메모리시간) + 런치 오버헤드
  - 전력: P = 유휴 + 코어(활용률 × f·V²) + 메모리(대역폭 활용률)
  - DVFS: 전력 한도(power limit)와 온도 한도 안에서 가능한 최고 클럭(15MHz 단위) 선택
          최저 클럭으로도 한도를 넘으면 듀티 사이클(쉬는 시간 삽입)로 평균 전력을 맞춤
  - 열: RC 열모델 (방열판 열용량 C, 열저항 R), 목표 온도 초과 시 열 스로틀
  - 텔레메트리: nvidia-smi 비슷한 클럭/온도/전력/활용률/VRAM/누적 에너지

워크로드: LLM 추론 1건 = 프리필(프롬프트 한 번에 처리, 연산 위주)
                        + 디코드(토큰 1개씩, 가중치 전체를 읽는 메모리 위주)

주의: 스펙 수치는 공개 스펙 기반 근사값이고, 효율 계수는 가정값이다.
      실제 텍스트는 생성하지 않는다(시간·전력·열만 모사).

단독 실행:  python3 vgpu.py bench            # GPU별 / 전력 한도별 성능·효율 표
            python3 vgpu.py smi --gpu 3060   # 부하를 주며 nvidia-smi 같은 출력
"""
import argparse, math, time

# ---------------------------------------------------------------- 스펙 (근사)
# matmul_tflops: 행렬연산 실효 피크(텐서코어 FP16 dense, 1080 은 FP32 CUDA 코어)
GPUS = {
    "5090": dict(name="RTX 5090 (가상)", arch="Blackwell", sm=170, min_mhz=600, base_mhz=2017, boost_mhz=2407,
                 matmul_tflops=209.0, mem_gbps=1792.0, vram_gb=32.0, tdp_w=575.0, min_pl_w=400.0, idle_w=30.0,
                 compute_eff=0.40, mem_eff=0.80, launch_us=6.0, heat_cap_j_per_c=1600.0, steady_c_at_tdp=72.0),
    "3080": dict(name="RTX 3080 10GB (가상)", arch="Ampere", sm=68, min_mhz=600, base_mhz=1440, boost_mhz=1710,
                 matmul_tflops=119.0, mem_gbps=760.0, vram_gb=10.0, tdp_w=320.0, min_pl_w=100.0, idle_w=18.0,
                 compute_eff=0.40, mem_eff=0.80, launch_us=7.0, heat_cap_j_per_c=900.0, steady_c_at_tdp=75.0),
    "4060L": dict(name="RTX 4060 Laptop (가상)", arch="Ada", sm=24, min_mhz=600, base_mhz=1470, boost_mhz=2370,
                  matmul_tflops=58.0, mem_gbps=256.0, vram_gb=8.0, tdp_w=115.0, min_pl_w=35.0, idle_w=5.0,
                  compute_eff=0.40, mem_eff=0.80, launch_us=8.0, heat_cap_j_per_c=250.0, steady_c_at_tdp=82.0),
    "3060": dict(name="RTX 3060 12GB (가상)", arch="Ampere", sm=28, min_mhz=600, base_mhz=1320, boost_mhz=1777,
                 matmul_tflops=51.0, mem_gbps=360.0, vram_gb=12.0, tdp_w=170.0, min_pl_w=100.0, idle_w=13.0,
                 compute_eff=0.40, mem_eff=0.80, launch_us=8.0, heat_cap_j_per_c=520.0, steady_c_at_tdp=74.0),
    "1080": dict(name="GTX 1080 (가상)", arch="Pascal", sm=20, min_mhz=600, base_mhz=1607, boost_mhz=1733,
                 matmul_tflops=8.9, mem_gbps=320.0, vram_gb=8.0, tdp_w=180.0, min_pl_w=90.0, idle_w=8.0,
                 compute_eff=0.55, mem_eff=0.72, launch_us=10.0, heat_cap_j_per_c=450.0, steady_c_at_tdp=80.0),
}

# LLM 모델 (llama.cpp GGUF Q4_K_M 기준 근사)
MODELS = {
    "0.5B": dict(name="Qwen2.5-0.5B Q4", params=0.49e9, weight_bytes=0.40e9, layers=24, hidden=896, kv_heads=2, head_dim=64),
    "7B":   dict(name="Mistral-7B Q4", params=7.24e9, weight_bytes=4.37e9, layers=32, hidden=4096, kv_heads=8, head_dim=128),
    "14B":  dict(name="Qwen2.5-14B Q4", params=14.8e9, weight_bytes=8.99e9, layers=48, hidden=5120, kv_heads=8, head_dim=128),
}

AMBIENT_C = 25.0
THROTTLE_C = 83.0      # 열 스로틀 목표 온도
SHUTDOWN_C = 95.0
V_MIN, V_MAX = 0.75, 1.05
CLOCK_STEP = 15        # MHz
CORE_SHARE, MEM_SHARE = 0.85, 0.25   # 동적 전력 중 코어/메모리 최대 비중 (가정)
CORE_FLOOR = 0.55                    # 메모리 위주 커널의 코어 활동률 하한 (가정)


class VGPUError(Exception):
    pass


class VGPUOOM(VGPUError):
    pass


class VirtualGPU:
    def __init__(self, key, realtime=True):
        if key not in GPUS:
            raise VGPUError(f"알 수 없는 GPU: {key} (가능: {', '.join(GPUS)})")
        self.key = key
        self.s = dict(GPUS[key])
        s = self.s
        self.realtime = realtime
        self.vram_total = int(s["vram_gb"] * 1024 ** 3)
        self.allocs = {}
        self.power_limit_w = s["tdp_w"]
        self.temp_c = AMBIENT_C + 8.0
        self.r_th = (s["steady_c_at_tdp"] - AMBIENT_C) / s["tdp_w"]   # °C/W
        self.clock_mhz = s["base_mhz"]
        self.thermal_cap_mhz = s["boost_mhz"]
        self.power_w = s["idle_w"]
        self.reason = "IDLE"
        self.energy_j = 0.0
        self.busy_s = 0.0
        self.time_s = 0.0
        self.tokens = 0
        self._win = [0.0, 0.0, 0.0, 0]   # (시간, 에너지, 바쁜 시간, 토큰) 텔레메트리 창
        self._sleep_debt = 0.0
        self.model = None

    # ------------------------------------------------------------ VRAM
    def vram_used(self):
        return sum(self.allocs.values())

    def malloc(self, tag, nbytes):
        nbytes = int(nbytes)
        if self.vram_used() + nbytes > self.vram_total:
            raise VGPUOOM(f"VRAM 부족: {tag} {nbytes/1e9:.2f}GB 요청, 남은 {(self.vram_total-self.vram_used())/1e9:.2f}GB")
        self.allocs[tag] = self.allocs.get(tag, 0) + nbytes

    def free(self, tag):
        self.allocs.pop(tag, None)

    def load_model(self, key, ctx=4096):
        m = MODELS[key]
        self.free("weights"); self.free("kv_cache"); self.free("runtime")
        self.malloc("runtime", 0.35e9)                         # CUDA 컨텍스트·버퍼 (가정)
        self.malloc("weights", m["weight_bytes"])
        self.malloc("kv_cache", self.kv_bytes_per_token(m) * ctx)
        self.model, self.model_key, self.ctx = m, key, ctx

    @staticmethod
    def kv_bytes_per_token(m):
        return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * 2   # K,V × FP16

    # ------------------------------------------------------------ 전력/클럭
    def set_power_limit(self, w):
        self.power_limit_w = max(0.0, min(self.s["tdp_w"], float(w)))

    def _voltage(self, f):
        s = self.s
        x = (f - s["min_mhz"]) / (s["boost_mhz"] - s["min_mhz"])
        return V_MIN + (V_MAX - V_MIN) * max(0.0, min(1.0, x))

    def _power(self, f, u_core, u_mem):
        s = self.s
        dyn = s["tdp_w"] - s["idle_w"]
        g = (f / s["boost_mhz"]) * (self._voltage(f) / V_MAX) ** 2
        # 메모리 대기 중에도 코어는 클럭을 유지하며 전력을 씀 → 활동률 하한
        act = CORE_FLOOR + (1 - CORE_FLOOR) * u_core
        return s["idle_w"] + dyn * (CORE_SHARE * 1.08 * act * g + MEM_SHARE * u_mem)

    def _timing(self, f, flops, nbytes):
        s = self.s
        t_c = flops / (s["matmul_tflops"] * 1e12 * s["compute_eff"] * f / s["boost_mhz"])
        t_m = nbytes / (s["mem_gbps"] * 1e9 * s["mem_eff"])
        t = max(t_c, t_m) + s["launch_us"] * 1e-6
        return t, t_c / t, t_m / t

    def _pick_clock(self, flops, nbytes):
        """DVFS: 한도 안의 최고 클럭. 반환 (클럭, 시간, 전력, 사유)."""
        s = self.s
        top = min(s["boost_mhz"], self.thermal_cap_mhz)
        f = top
        while True:
            t, uc, um = self._timing(f, flops, nbytes)
            p = self._power(f, uc, um)
            if p <= self.power_limit_w or f - CLOCK_STEP < s["min_mhz"]:
                break
            f -= CLOCK_STEP
        if p <= self.power_limit_w:
            reason = "THERMAL" if top < s["boost_mhz"] and f == top else ("PWR_CAP" if f < top else "NONE")
        else:
            reason = "DUTY"
        return f, t, p, reason

    # ------------------------------------------------------------ 시간 진행
    def _advance(self, dt, p, busy):
        """dt 초 동안 전력 p 로 동작: 열·에너지 갱신 (+실시간 모드면 실제 대기)."""
        if dt <= 0:
            return
        c = self.s["heat_cap_j_per_c"]
        t_ss = AMBIENT_C + p * self.r_th
        self.temp_c = t_ss + (self.temp_c - t_ss) * math.exp(-dt / (self.r_th * c))
        self.energy_j += p * dt
        self.time_s += dt
        self.power_w = p
        w = self._win
        w[0] += dt; w[1] += p * dt
        if busy:
            self.busy_s += dt
            w[2] += dt
        # 열 스로틀 제어
        s = self.s
        if self.temp_c > THROTTLE_C:
            self.thermal_cap_mhz = max(s["min_mhz"], self.thermal_cap_mhz - 2 * CLOCK_STEP)
        elif self.temp_c < THROTTLE_C - 3:
            self.thermal_cap_mhz = min(s["boost_mhz"], self.thermal_cap_mhz + CLOCK_STEP)
        if self.realtime:
            self._sleep_debt += dt
            if self._sleep_debt >= 0.02:
                time.sleep(self._sleep_debt)
                self._sleep_debt = 0.0

    def idle(self, dt):
        """작업 없이 흘러간 실제 시간을 반영(냉각). 실시간 대기는 하지 않음."""
        rt, self.realtime = self.realtime, False
        if dt > 0.15:   # 통신 사이의 짧은 틈은 유휴로 표시하지 않음
            self.clock_mhz, self.reason = self.s["min_mhz"], "IDLE"
        self._advance(dt, self.s["idle_w"], busy=False)
        self.realtime = rt

    def step(self, flops, nbytes):
        """커널(스텝) 하나 실행. 반환: 걸린 시간(초)."""
        if self.temp_c >= SHUTDOWN_C:
            raise VGPUError("과열 보호 정지")
        f, t, p, reason = self._pick_clock(flops, nbytes)
        self.clock_mhz, self.reason = f, reason
        self._advance(t, p, busy=True)
        total = t
        if reason == "DUTY":
            idle = self.s["idle_w"]
            lim = self.power_limit_w
            if lim <= idle + 1:
                raise VGPUError(f"전력 한도 {lim:.0f}W 가 유휴 전력 이하라 연산 불가")
            t_off = t * (p - lim) / (lim - idle)
            self._advance(t_off, idle, busy=False)
            total += t_off
        return total

    # ------------------------------------------------------------ LLM 추론
    def infer(self, prompt_tokens, n_predict, prefill_chunk=512):
        if not self.model:
            raise VGPUError("모델이 로드되지 않음")
        m = self.model
        if prompt_tokens + n_predict > self.ctx:
            raise VGPUError("컨텍스트 길이 초과")
        kvb = self.kv_bytes_per_token(m)
        t0, e0 = self.time_s, self.energy_j
        pos = 0
        while pos < prompt_tokens:                 # 프리필: 연산 위주
            n = min(prefill_chunk, prompt_tokens - pos)
            flops = 2 * m["params"] * n + 2 * m["layers"] * m["hidden"] * n * (pos + n)
            nbytes = m["weight_bytes"] + kvb * n
            self.step(flops, nbytes)
            pos += n
        t_prefill = self.time_s - t0
        for _ in range(n_predict):                # 디코드: 메모리 위주
            flops = 2 * m["params"] + 4 * m["layers"] * m["hidden"] * pos
            nbytes = m["weight_bytes"] + kvb * pos
            self.step(flops, nbytes)
            pos += 1
        self.tokens += n_predict
        self._win[3] += n_predict
        if self.realtime and self._sleep_debt > 0:
            time.sleep(self._sleep_debt)
            self._sleep_debt = 0.0
        dt, de = self.time_s - t0, self.energy_j - e0
        t_dec = dt - t_prefill
        return {"tokens": n_predict, "prompt_tokens": prompt_tokens, "gpu_s": round(dt, 4),
                "prefill_ms": round(t_prefill * 1000, 1),
                "decode_tps": round(n_predict / t_dec, 1) if t_dec > 0 else 0.0,
                "energy_j": round(de, 2), "tok_per_j": round(n_predict / de, 3) if de > 0 else 0.0}

    # ------------------------------------------------------------ 텔레메트리
    def telemetry(self, reset_window=True):
        w = self._win
        avg_p = w[1] / w[0] if w[0] > 0 else self.power_w
        util = w[2] / w[0] if w[0] > 0 else 0.0
        tps = w[3] / w[0] if w[0] > 0 else 0.0
        out = {"gpu": self.s["name"], "clock_mhz": int(self.clock_mhz), "temp_c": round(self.temp_c, 1),
               "power_w": round(avg_p, 1), "power_limit_w": round(self.power_limit_w, 1), "tdp_w": self.s["tdp_w"],
               "util": round(util, 3), "tok_s": round(tps, 1), "reason": self.reason,
               "vram_used_gb": round(self.vram_used() / 1024 ** 3, 2), "vram_total_gb": self.s["vram_gb"],
               "energy_wh": round(self.energy_j / 3600, 4), "model": self.model["name"] if self.model else None}
        if reset_window:
            self._win = [0.0, 0.0, 0.0, 0]
        return out


def estimate_tokens(text):
    """대략적 토큰 수 (한국어 ≈ 글자당 1토큰, 영어 ≈ 4글자당 1토큰) + 템플릿."""
    ko = sum(1 for ch in text if "가" <= ch <= "힣")
    return 24 + ko + max(0, len(text) - ko) // 4


# ---------------------------------------------------------------- CLI
def bench(args):
    print(f"모델 {MODELS[args.model]['name']} / 프롬프트 {args.prompt}토큰 + 생성 {args.n}토큰  (모두 모사값)\n")
    print(f"{'GPU':<22}{'전력한도':>8}{'클럭':>8}{'디코드':>10}{'1건':>9}{'평균전력':>9}{'토큰/J':>8}{'온도':>7}  사유")
    for key in (GPUS if args.gpu == "all" else [args.gpu]):
        s = GPUS[key]
        for frac in sorted({1.0, 0.8, 0.6, s["min_pl_w"] / s["tdp_w"], 0.35}, reverse=True):
            g = VirtualGPU(key, realtime=False)
            try:
                g.load_model(args.model)
            except VGPUOOM as e:
                print(f"{s['name']:<22}  {e}")
                break
            g.set_power_limit(s["tdp_w"] * frac)
            g.temp_c = 60.0
            r = None
            for _ in range(3):
                r = g.infer(args.prompt, args.n)
            t = g.telemetry()
            print(f"{s['name']:<22}{g.power_limit_w:>7.0f}W{t['clock_mhz']:>6}MHz{r['decode_tps']:>7.1f}t/s"
                  f"{r['gpu_s']:>8.2f}s{r['energy_j']/r['gpu_s']:>8.0f}W{r['tok_per_j']:>8.2f}{t['temp_c']:>6.0f}°C  {t['reason']}")
        print()


def smi(args):
    g = VirtualGPU(args.gpu, realtime=True)
    g.load_model(args.model)
    g.set_power_limit(args.pl or g.s["tdp_w"])
    print(f"{g.s['name']} 에 부하를 겁니다 (Ctrl+C 종료)")
    try:
        while True:
            g.infer(args.prompt, args.n)
            t = g.telemetry()
            print(f"{t['clock_mhz']:>5}MHz {t['temp_c']:>5.1f}°C {t['power_w']:>6.1f}/{t['power_limit_w']:.0f}W "
                  f"util {t['util']*100:>3.0f}% {t['tok_s']:>6.1f}tok/s VRAM {t['vram_used_gb']}/{t['vram_total_gb']}GB {t['reason']}",
                  flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="와트투비트 가상 GPU")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bench", "smi"):
        p = sub.add_parser(name)
        p.add_argument("--gpu", default="all" if name == "bench" else "3060")
        p.add_argument("--model", default="7B", choices=list(MODELS))
        p.add_argument("--prompt", type=int, default=200)
        p.add_argument("--n", type=int, default=128)
        if name == "smi":
            p.add_argument("--pl", type=float, default=None, help="전력 한도(W)")
    a = ap.parse_args()
    bench(a) if a.cmd == "bench" else smi(a)
