#!/usr/bin/env bash
# 오케스트레이터 PC(이 PC)용: ./start_orchestrator.sh   — 토큰을 한 번 만들어 .w2b/orch.env 에 저장
# 옵션은 orchestrator.py 로 그대로 전달 (예: --port 8700)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; W="$ROOT/.w2b"; mkdir -p "$W"
if [ ! -f "$W/orch.env" ]; then
  umask 077
  echo "W2B_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')" > "$W/orch.env"
fi
# shellcheck disable=SC1091
source "$W/orch.env"
export W2B_TOKEN
echo "[w2b] 토큰: $W2B_TOKEN   (노드 스크립트가 물어보면 이 값을 입력)"
cd "$ROOT"
exec python3 orchestrator.py --real "$@"
