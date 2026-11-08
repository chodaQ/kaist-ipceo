# 와트투비트 프로토타입 (MVP)

가상 GPU 노드 4개 + 태양광 시뮬레이터 + 풀(pull) 방식 작업 큐 + 대시보드.
가상 GPU(`vgpu.py`)는 스펙 기반 소프트웨어 GPU 시뮬레이터(루프라인 성능·DVFS·전력·열·VRAM 모델)입니다.
파이썬 표준 라이브러리만 사용합니다 (설치할 패키지 없음).

## 실행 (Arch Linux, 5625U / 12GB)

```bash
python3 orchestrator.py                     # 기본: vgpu 백엔드, 7B 모델, 포트 8700, 노드 4개
python3 orchestrator.py --model 14B         # 1080(8GB)은 VRAM 부족으로 작업을 받지 않음
python3 orchestrator.py --backend mock      # 예전 방식(sha256 + sleep)
# 브라우저에서 http://127.0.0.1:8700
```

옵션: `--nodes 4 --unit-ms 400 --port 8700 --model 0.5B|7B|14B`

## 3D 시뮬레이터 (발표·설명용)

`simulator3d.html` 을 브라우저로 열기만 하면 됩니다 (서버·GPU 불필요, 인터넷은 three.js·글꼴 로딩에만 필요).
오케스트레이터를 켰다면 `http://127.0.0.1:8700/3d` 로도 열립니다.

- 4가구 마을: 지붕 태양광 → 인버터 → 배터리 / 전력망 판매(수용 한도까지) / 남는 전기 → GPU 노드 → 폐열로 온수
- 노드 제어·전력 한도·DVFS·작업 재배정·온수 한계는 `orchestrator.py`·`vgpu.py` 와 같은 규칙을 JS 로 옮김
- B동은 팀 실제 장비 사양(RTX 3080), A동(RTX 5090)·C·D동(RTX 4080 16GB)은 비교용 가상 장비
- 조작: 시각·속도, 태양광 세기, 전력망 수용 한도, 구름, 노드 GPU 잉여만큼 자동(최대 8장)/1장/4장 고정, 노드 끄기(비교), 장애 주입, 시점 전환, 집 클릭 선택
- 아래 패널: 노드 없음/있음 하루 합계, 하루 전력 흐름 그래프, 전력 한도↔속도 그래프(선택한 집 GPU)

## 가상 GPU 단독 실행

```bash
python3 vgpu.py bench                 # GPU별 × 전력 한도별 디코드 속도·평균전력·토큰/J 표
python3 vgpu.py smi --gpu 3060 --pl 100   # 부하를 걸고 nvidia-smi 처럼 클럭/온도/전력 출력
```

가상 GPU가 하는 일:
- 노드가 받은 추론 1건을 프리필(연산 위주) + 디코드(토큰마다 가중치 전체 읽기, 메모리 위주) 스텝으로 쪼갬
- 스텝마다 전력 한도·온도 한도 안의 최고 클럭(15MHz 단위)을 고르고, 루프라인으로 시간을 계산, 그 시간만큼 실제로 대기
- 최저 클럭으로도 한도를 넘으면 쉬는 시간을 끼워 넣어(듀티 사이클) 평균 전력을 맞춤
- RC 열모델로 온도 계산 → 83°C 초과 시 열 스로틀
- 오케스트레이터는 `전력 한도 = 잉여전력 − 나머지 PC 소비(system_w)`를 내려보내고,
  노드가 보고한 GPU 전력으로 사용량을 집계

관찰되는 현상(모델상): LLM 디코드는 메모리 대역폭에 묶여 있어 전력 한도를 낮춰도 tok/s 는 거의 그대로이고 토큰/J 는 좋아진다.

대시보드에서 할 수 있는 것: 시각 이동/속도 조절, 구름(발전 급감) 주입, 작업 자동 투입,
노드 강제 종료·재시작(장애 주입).

## 진짜 연산 붙이기 (선택)

```bash
sudo pacman -S llama.cpp            # 패키지명은 환경에 따라 다름 (AUR: llama.cpp-git 등)
llama-server -m 작은모델.gguf -c 2048 -t 6 --port 8080   # 0.5B급 Q4 권장
python3 orchestrator.py --backend llama
```

## 무엇이 진짜이고 무엇이 모델인가

- 진짜: 노드는 별도 프로세스, 작업 큐·하트비트·이탈 감지·재큐잉, CPU 연산(mock은 sha256 반복, llama는 실제 추론).
- 모델(가정): 태양광 발전량, 잉여 전력, GPU 속도/전력/온도(vgpu: 공개 스펙 근사 + 효율 계수 가정), 온수 탱크 열 환원.
  vgpu 는 텍스트를 실제로 생성하지 않습니다(시간·전력·열만 모사).
  발표·시연 시 "모델링한 값"이라고 명시할 것.

## 알려진 한계

- llama 백엔드는 이 환경에서 테스트하지 못했습니다 (직접 확인 필요).
- 온수 환원은 부분 예열 수준으로 모델링했습니다 (과장 금지).
- 층 단위 분산(llama.cpp RPC)은 v0.2 계획이며 미구현입니다.
# kaist-ipceo

---

## 실제 하드웨어 구성 (노트북 2대 + 3080 서버)

```
 [이 PC · 5625U]  orchestrator.py --real   ← 대시보드, 태양광·가정 시뮬레이터, 작업 큐
        ▲  LAN (HTTP, 노드가 먼저 접속)
        ├── [노트북 1] agent.py ── llama-server(CUDA) ── RTX 노트북 GPU   (duty 제어)
        ├── [노트북 2] agent.py ── llama-server(CUDA) ── RTX 노트북 GPU   (duty 제어)
        └── [3080 서버] agent.py ── llama-server(CUDA) ── RTX 3080       (nvidia-smi -pl)
```

진짜인 것: GPU 전력·온도·클럭·VRAM(nvidia-smi 실측), 추론(llama.cpp 실제 생성), 전력 한도 적용,
LAN 위의 작업 분배·이탈 감지·재배정. 모사인 것: 각 노드에 붙인 태양광·가정 부하·배터리·온수.

### 빠른 시작: clone → 스크립트 하나

| 장비 | 실행 | 하는 일 |
|---|---|---|
| 오케스트레이터 PC (이 PC) | `./start_orchestrator.sh` | 토큰 생성·저장 후 `--real` 로 실행, 노드 접속 주소·토큰 출력 |
| RTX 3080 서버 (Linux) | `./start_server_3080.sh` | 아래 공통 과정 + `nvidia-smi -pl` 용 sudoers 규칙 추가 제안 |
| 4060 Laptop (Linux) | `./start_laptop_linux.sh` | 아래 공통 과정 + 충전기·전원 프로파일 확인 |
| 4060 Laptop (Windows) | `start_laptop_windows.bat` 더블클릭 | 아래 공통 과정 (Python 없으면 winget 설치 제안) |

노드 스크립트 공통 과정:
1. Python 3.8+, `nvidia-smi`, 드라이버 버전 확인
2. **처음 한 번만** 오케스트레이터 주소·토큰·노드 ID·소비전력을 물어 `.w2b/node.env`(Windows: `.w2b\node.json`)에 저장
3. llama.cpp CUDA 배포판(태그 `b12189` 고정, Linux CUDA 12.8 / Windows CUDA 12.4)을 `.w2b/` 에 받고 CUDA 런타임을 같은 폴더에 배치
4. 모델 Qwen2.5-7B-Instruct Q4_K_M(4.7GB)을 `.w2b/models/` 에 받고 크기 검증 — 끊기면 다시 실행해 이어받기
5. `agent.py --check` 로 점검 → 통과하면 에이전트 실행, 비정상 종료 시 5초 뒤 자동 재시작 (Ctrl+C 로 종료)

```bash
# 오케스트레이터 PC
git clone https://github.com/chodaQ/kaist-ipceo.git && cd kaist-ipceo
./start_orchestrator.sh            # 출력된 "실제 노드 접속 주소"와 토큰을 메모

# 3080 서버 / Linux 노트북
git clone https://github.com/chodaQ/kaist-ipceo.git && cd kaist-ipceo
./start_server_3080.sh             # 또는 ./start_laptop_linux.sh
```
Windows 노트북: clone(또는 ZIP 다운로드) 후 `start_laptop_windows.bat` 더블클릭.

옵션: `--reconfigure`(설정 다시) `--check-only`(점검만) `--skip-check` — Windows 는 `-Reconfigure` `-CheckOnly` `-SkipCheck`.
고급: 직접 받은 모델·빌드를 쓰려면 Linux 는 `MODEL_PATH=... LLAMA_BIN=...` 환경변수, Windows 는 `node.json` 의 `model_path`/`llama_bin`.
디스크: llama.cpp ≈1GB(압축 포함) + 모델 4.7GB. 내려받은 파일·설정(토큰 포함)은 `.w2b/` 에만 저장되고 git 에는 올라가지 않음.

방화벽을 쓰면 오케스트레이터 PC 의 TCP 8700 을 연다 (`sudo ufw allow 8700/tcp` 등). 노드 쪽은 나가는 연결만 필요.
대시보드: 오케스트레이터 PC 에서 `http://127.0.0.1:8700/?token=<토큰>`.

### 수동 실행 (스크립트 없이)

```bash
python3 orchestrator.py --real --token 비밀값
python3 agent.py --orch http://<오케스트레이터IP>:8700 --id SRV --token 비밀값 --system-w 90 \
    --llama-bin <llama-server 경로> --gguf <모델.gguf>
```
- 이미 llama-server 를 따로 띄워 뒀다면 `--llama-bin/--gguf` 대신 `--llama-url http://127.0.0.1:8080`
- `--solar-kw`: 이 노드에 붙일 가상 태양광 용량. 생략하면 접속 순서대로 5/4/3kW

### 4) 전력 한도 제어 방식 (`--power-mode auto` 가 자동 선택)

| 방식 | 대상 | 원리 |
|---|---|---|
| `pl` | 3080 서버 등 데스크톱 GPU | `nvidia-smi -pl W` 로 하드웨어 한도 설정. 공장 기본값 이상으로는 안 올림, 종료 시 원복 |
| `duty` | 노트북(-pl 미지원) | 요청 1건의 실측 평균전력이 한도보다 크면 그 비율만큼 다음 요청 전에 쉼 |

- 한도가 GPU 최소 한도(3080 ≈100W)보다 낮으면 `pl` + `duty` 를 같이 씀
- Linux 에서 `-pl` 은 root 권한 필요. 에이전트를 sudo 로 돌리지 말고 아래처럼 nvidia-smi 만 허용:
  ```bash
  echo "$USER ALL=(root) NOPASSWD: $(command -v nvidia-smi)" | sudo tee /etc/sudoers.d/w2b-nvidia-smi
  ```
  (`start_server_3080.sh` 가 visudo 로 검증한 뒤 같은 규칙을 넣어 줌. 이 규칙은 nvidia-smi 의 모든 root 기능을 허용함)
  (에이전트가 실패 시 `sudo -n nvidia-smi -pl` 로 자동 재시도). Windows 는 관리자 권한 터미널에서 실행.
- 권한이 없으면 자동으로 `duty` 로 내려감 (로그에 표시)

### 4-1) 이 팀 장비 기준 (RTX 4060 Laptop ×2 + RTX 3080 서버)

| 장비 | VRAM | 모델 | 제어 | 예상 디코드(가상 GPU 추정) |
|---|---|---|---|---|
| RTX 4060 Laptop | 8GB | Qwen2.5-7B Q4_K_M (≈4.7GB, ctx 4096 여유 있음) | duty | ≈45 tok/s, ≈90–115W |
| RTX 3080 10GB | 10GB | 같은 모델 | `-pl` (100~320W) | ≈135 tok/s, ≈250W |

예상치는 `python3 vgpu.py bench --gpu 4060L` / `--gpu 3080` 결과이며 실측과 비교해 볼 것.

**노트북 OS**: 두 스크립트 모두 llama.cpp 배포판을 받으므로 빌드는 필요 없다. 어느 쪽이든 노트북은 `-pl` 이
막혀 있어 duty 제어로 동작. 충전기 연결 + 고성능 전원 모드 필수 (배터리 구동 시 GPU 전력이 크게 제한됨).
Linux 배포판은 CUDA 12.8 이라 **NVIDIA 드라이버 570 이상**, Windows 배포판은 CUDA 12.4 라 551 이상 필요.

3080 서버: `./start_server_3080.sh` 가 sudoers 규칙 추가를 제안 → 수락하면 점검에서 "전력 한도 제어 OK".

### 5) 시연 팁

- 실제 GPU 는 수백 W 수준이라 가상 가정(3~5kW)에서는 잉여가 늘 충분함.
  전력 한도가 움직이는 걸 보이려면 대시보드 **태양광 세기를 10~15%** 로 내리거나 **구름 이벤트**를 사용.
- **강제 이탈 15초**: 해당 에이전트가 15초간 응답을 끊음 → 처리 중 작업이 다른 노드로 재배정되는 것을 확인.
  진짜 장애는 랜선을 뽑거나 에이전트를 Ctrl+C 로 끄면 됨 (다시 실행하면 자동 재접속).
- GPU 없이 흐름만 점검: `python3 agent.py --orch ... --id TEST --sim-gpu 3080`

### 첫 실행 전 점검

각 노드에서 `--check` 로 nvidia-smi·전력 측정·-pl 가능 여부·스트리밍·llama-server 실제 추론·GPU 오프로드·
오케스트레이터 접속·토큰을 한 번에 확인한다 (작업은 받지 않음, 전력 한도는 건드린 뒤 원복).

### 실측으로 검증 못 한 부분 (이 PC 에는 NVIDIA GPU 가 없음)

가짜 nvidia-smi·llama-server 로 등록·전력한도 적용/원복·duty·이탈 재배정·토큰 인증까지 확인했지만,
실제 기기에서 아래는 처음 실행 때 확인 필요:
- 노트북 드라이버가 `power.draw` 를 주는지 (`[N/A]` 면 활용률×한도로 추정 — 정확도 떨어짐)
- llama-server 버전별 `/completion` 응답의 `timings` 필드
- Windows 에서 `nvidia-smi -lms` 스트리밍 동작
