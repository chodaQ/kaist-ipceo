# 와트투비트 프로토타입 (MVP)

가상 GPU 노드 4개 + 태양광 시뮬레이터 + 풀(pull) 방식 작업 큐 + 대시보드.
파이썬 표준 라이브러리만 사용합니다 (설치할 패키지 없음).

## 실행 (Arch Linux, 5625U / 12GB)

```bash
python3 orchestrator.py --backend mock      # 기본: 포트 8700, 노드 4개
# 브라우저에서 http://127.0.0.1:8700
```

옵션: `--nodes 4 --unit-ms 400 --port 8700`

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
- 모델(가정): 태양광 발전량, 잉여 전력, GPU 속도/전력 프로필, 온수 탱크 열 환원.
  발표·시연 시 "모델링한 값"이라고 명시할 것.

## 알려진 한계

- llama 백엔드는 이 환경에서 테스트하지 못했습니다 (직접 확인 필요).
- 온수 환원은 부분 예열 수준으로 모델링했습니다 (과장 금지).
- 층 단위 분산(llama.cpp RPC)은 v0.2 계획이며 미구현입니다.
# kaist-ipceo
