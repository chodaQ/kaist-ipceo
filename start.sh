#!/usr/bin/env bash
# 와트투비트 서버 실행 (오케스트레이터 PC)
#
#   ./start.sh            시뮬레이션: 이 PC 안에 가상 GPU 노드 4개 (GPU·네트워크 불필요)
#   ./start.sh real       실제 장비: LAN 에 열고 3080 서버·노트북 에이전트 접속 대기 (토큰 자동 생성)
#   ./start.sh 3d         3D 시뮬레이터만 브라우저로 열기 (서버 없이)
#
# 옵션(뒤에 붙이면 orchestrator.py 로 전달): --port 8700  --model 0.5B|7B|14B  --nodes 1~4 ...
# 브라우저 자동 열기 끄기: NO_BROWSER=1 ./start.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${1:-sim}"
case "$MODE" in sim|real|3d) shift || true ;; -*) MODE=sim ;; *) echo "사용법: ./start.sh [sim|real|3d] [옵션]"; exit 2 ;; esac

command -v python3 >/dev/null || { echo "python3 가 필요합니다"; exit 1; }

PORT=8700
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do [ "${args[$i]}" = "--port" ] && PORT="${args[$((i+1))]:-8700}"; done

open_url() {  # 서버가 뜬 뒤 브라우저 열기
  [ "${NO_BROWSER:-0}" = 1 ] && return
  local opener=""
  for c in xdg-open open; do command -v "$c" >/dev/null && { opener="$c"; break; }; done
  [ -n "$opener" ] || return
  (
    for _ in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$PORT/state" && break; sleep 0.25; done
    for u in "$@"; do "$opener" "$u" >/dev/null 2>&1 || true; done
  ) &
}

if [ "$MODE" = 3d ]; then
  f="$ROOT/simulator3d.html"
  if command -v xdg-open >/dev/null; then xdg-open "$f"; elif command -v open >/dev/null; then open "$f"; else echo "브라우저로 열기: $f"; fi
  exit 0
fi

if curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$PORT/state"; then
  echo "[w2b] 포트 $PORT 에 이미 서버가 떠 있습니다. 끄고 다시 실행하거나 --port 로 다른 포트를 쓰세요"
  exit 1
fi

if [ "$MODE" = real ]; then
  # 토큰은 .w2b/orch.env 에 한 번 만들어 재사용
  mkdir -p .w2b
  [ -f .w2b/orch.env ] || (umask 077; echo "W2B_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')" > .w2b/orch.env)
  # shellcheck disable=SC1091
  source .w2b/orch.env
  open_url "http://127.0.0.1:$PORT/?token=$W2B_TOKEN" "http://127.0.0.1:$PORT/3d"
  exec ./start_orchestrator.sh "$@"
fi

echo "[w2b] 시뮬레이션 모드 · 대시보드 http://127.0.0.1:$PORT  · 3D http://127.0.0.1:$PORT/3d  (종료: Ctrl+C)"
open_url "http://127.0.0.1:$PORT/" "http://127.0.0.1:$PORT/3d"
exec python3 orchestrator.py "$@"
