#!/usr/bin/env bash
# 와트투비트 노드 원클릭 실행 (Linux) — start_server_3080.sh / start_laptop_linux.sh 가 호출
#
#   처음 실행: 오케스트레이터 주소·토큰·노드 ID 를 묻고 .w2b/node.env 에 저장
#   필요하면 llama.cpp(CUDA 빌드)·모델을 .w2b/ 에 내려받음 (끊겨도 이어받기)
#   점검(agent.py --check) 후 에이전트 실행, 죽으면 5초 뒤 자동 재시작 (Ctrl+C 로 종료)
#
# 옵션: --reconfigure (설정 다시 묻기)  --check-only (점검만)  --skip-check (점검 생략)
# 환경변수로 덮어쓰기: ORCH TOKEN NODE_ID SYSTEM_W SOLAR_KW GPU_INDEX MODEL_PATH LLAMA_BIN LLAMA_TAG
set -euo pipefail

ROLE="${W2B_ROLE:-laptop}"          # server | laptop
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$ROOT/.w2b"
CONF="$W/node.env"
LLAMA_TAG="${LLAMA_TAG:-b12189}"     # 검증한 릴리스로 고정 (GitHub 'latest' 는 다른 태그를 가리킴)
CUDA_VER="12.8"                      # 리눅스 배포 바이너리의 CUDA 버전 → 드라이버 570 이상 필요
MIN_DRIVER=570
MODEL_URL="${MODEL_URL:-https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf}"
MODEL_SIZE=4683074240

RECONF=0; CHECK_ONLY=0; SKIP_CHECK="${SKIP_CHECK:-0}"
for a in "$@"; do
  case "$a" in
    --reconfigure) RECONF=1 ;;
    --check-only)  CHECK_ONLY=1 ;;
    --skip-check)  SKIP_CHECK=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "알 수 없는 옵션: $a" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;34m[w2b]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[w2b] 주의:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[w2b] 오류:\033[0m %s\n' "$*" >&2; exit 1; }
interactive() { [ -t 0 ]; }
ask() {  # ask 변수명 "질문" 기본값
  local __v="$1" __q="$2" __d="${3:-}" __a
  if interactive; then read -r -p "  $__q [${__d}]: " __a || true; else __a=""; fi
  printf -v "$__v" '%s' "${__a:-$__d}"
}
yesno() {  # yesno "질문" Y|N
  local a d="${2:-N}"
  interactive || return 1     # 비대화형 실행에서는 변경 작업을 하지 않음
  read -r -p "  $1 [$( [ "$d" = Y ] && echo Y/n || echo y/N )]: " a || true
  a="${a:-$d}"; [[ "$a" =~ ^[Yy] ]]
}

mkdir -p "$W/dl" "$W/models"
cd "$ROOT"

# ------------------------------------------------------------------ 1. 필수 도구
say "필수 도구 확인"
command -v python3 >/dev/null || die "python3 가 없습니다 (Arch: sudo pacman -S python / Ubuntu: sudo apt install python3)"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' || die "Python 3.8 이상이 필요합니다 ($(python3 --version))"
for t in curl tar; do command -v "$t" >/dev/null || die "$t 가 없습니다"; done
command -v nvidia-smi >/dev/null || die "nvidia-smi 가 없습니다. NVIDIA 독점 드라이버를 설치하고 재부팅하세요
       (Arch: sudo pacman -S nvidia-open (또는 nvidia) / Ubuntu: sudo ubuntu-drivers install)"
GPU_LINE="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)" \
  || die "nvidia-smi 실행 실패 — 드라이버가 로드되지 않았습니다 (재부팅/Secure Boot 확인)"
[ -n "$GPU_LINE" ] || die "NVIDIA GPU 를 찾지 못했습니다"
DRIVER="$(echo "$GPU_LINE" | cut -d, -f2 | tr -d ' ')"
say "GPU: $GPU_LINE"

# ------------------------------------------------------------------ 2. 설정
DEF_ID="SRV"; DEF_SYS=90
if [ "$ROLE" = laptop ]; then
  DEF_ID="$(hostname | tr -cd 'A-Za-z0-9' | tr 'a-z' 'A-Z' | cut -c1-10)"; DEF_ID="${DEF_ID:-LAP1}"; DEF_SYS=25
fi
if [ ! -f "$CONF" ] || [ "$RECONF" = 1 ]; then
  [ -n "${ORCH:-}" ] && [ -n "${TOKEN:-}" ] || interactive || die "설정이 없습니다. 터미널에서 직접 실행하거나 ORCH·TOKEN 환경변수를 주세요"
  say "처음 설정 (Enter = 기본값). 오케스트레이터 PC 에서 ./start_orchestrator.sh 가 출력한 주소·토큰을 입력하세요"
  while [ -z "${ORCH:-}" ]; do ask ORCH "오케스트레이터 주소 (예: http://192.168.0.10:8700)" ""; done
  case "$ORCH" in http://*|https://*) ;; *) ORCH="http://$ORCH" ;; esac
  case "$ORCH" in *:[0-9]*) ;; *) ORCH="$ORCH:8700" ;; esac
  [ -n "${TOKEN:-}" ]    || ask TOKEN "토큰" ""
  [ -n "${NODE_ID:-}" ]  || ask NODE_ID "노드 ID (노드마다 다르게, 16자 이내)" "$DEF_ID"
  [ -n "${SYSTEM_W:-}" ] || ask SYSTEM_W "GPU 를 뺀 나머지 PC 소비전력(W)" "$DEF_SYS"
  [ -n "${SOLAR_KW:-}" ] || ask SOLAR_KW "가상 태양광 용량 kW (비우면 자동)" ""
  umask 077
  {
    printf 'ORCH=%q\nTOKEN=%q\nNODE_ID=%q\nSYSTEM_W=%q\nSOLAR_KW=%q\n' "$ORCH" "$TOKEN" "$NODE_ID" "$SYSTEM_W" "$SOLAR_KW"
    printf 'GPU_INDEX=%q\nMODEL_PATH=%q\nLLAMA_BIN=%q\n' "${GPU_INDEX:-0}" "${MODEL_PATH:-}" "${LLAMA_BIN:-}"
  } > "$CONF"
  say "설정 저장: $CONF  (다시 설정: --reconfigure)"
fi
# 저장값 읽되, 이미 준 환경변수가 우선
_env_ORCH="${ORCH:-}"; _env_TOKEN="${TOKEN:-}"; _env_ID="${NODE_ID:-}"; _env_SYS="${SYSTEM_W:-}"
_env_SOLAR="${SOLAR_KW:-}"; _env_GI="${GPU_INDEX:-}"; _env_MP="${MODEL_PATH:-}"; _env_LB="${LLAMA_BIN:-}"
# shellcheck disable=SC1090
source "$CONF"
ORCH="${_env_ORCH:-$ORCH}"; TOKEN="${_env_TOKEN:-$TOKEN}"; NODE_ID="${_env_ID:-$NODE_ID}"; SYSTEM_W="${_env_SYS:-$SYSTEM_W}"
SOLAR_KW="${_env_SOLAR:-${SOLAR_KW:-}}"; GPU_INDEX="${_env_GI:-${GPU_INDEX:-0}}"
MODEL_PATH="${_env_MP:-${MODEL_PATH:-}}"; LLAMA_BIN="${_env_LB:-${LLAMA_BIN:-}}"
[ -n "$TOKEN" ] || warn "토큰이 비어 있습니다 (오케스트레이터를 --token 없이 띄운 경우에만 동작)"

if curl -s --max-time 4 "$ORCH/state" >/dev/null; then say "오케스트레이터 응답 확인: $ORCH"
else warn "오케스트레이터($ORCH)에 아직 연결되지 않습니다 — 켜져 있는지, 같은 네트워크인지 확인 (계속 진행)"; fi

# ------------------------------------------------------------------ 3. llama.cpp (CUDA)
fetch() {  # fetch URL 파일 — 이어받기 + 재시도
  local url="$1" out="$2"
  [ -s "$out" ] && [ ! -f "$out.part" ] && return 0
  say "내려받는 중: $(basename "$out")"
  [ -f "$out" ] && [ ! -f "$out.part" ] && mv "$out" "$out.part"
  curl -fL --retry 5 --retry-delay 3 -C - --progress-bar -o "$out.part" "$url" || die "다운로드 실패: $url (다시 실행하면 이어받습니다)"
  mv "$out.part" "$out"
}
if [ -z "$LLAMA_BIN" ]; then
  LLAMA_DIR="$W/llama-$LLAMA_TAG-cuda$CUDA_VER"
  LLAMA_BIN="$LLAMA_DIR/llama-server"
  if [ ! -x "$LLAMA_BIN" ]; then
    if [ "${DRIVER%%.*}" -lt "$MIN_DRIVER" ] 2>/dev/null; then
      warn "드라이버 $DRIVER < $MIN_DRIVER: 배포 바이너리(CUDA $CUDA_VER)가 동작하지 않을 수 있습니다. 드라이버를 올리거나 LLAMA_BIN=직접빌드경로 로 지정하세요"
    fi
    BASE="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG"
    MAIN="llama-$LLAMA_TAG-bin-ubuntu-cuda-$CUDA_VER-x64.tar.gz"
    RT="cudart-llama-$LLAMA_TAG-bin-ubuntu-cuda-$CUDA_VER-x64.tar.gz"
    say "llama.cpp $LLAMA_TAG (CUDA $CUDA_VER) 준비 — 약 750MB"
    fetch "$BASE/$MAIN" "$W/dl/$MAIN"
    fetch "$BASE/$RT" "$W/dl/$RT"
    rm -rf "$LLAMA_DIR.tmp"; mkdir -p "$LLAMA_DIR.tmp"
    tar xzf "$W/dl/$MAIN" -C "$LLAMA_DIR.tmp" --strip-components=1 || die "압축 해제 실패: $MAIN (파일을 지우고 다시 실행: rm $W/dl/$MAIN)"
    tar xzf "$W/dl/$RT"   -C "$LLAMA_DIR.tmp" --strip-components=1 || die "압축 해제 실패: $RT (rm $W/dl/$RT 후 재실행)"
    [ -x "$LLAMA_DIR.tmp/llama-server" ] || die "압축 안에서 llama-server 를 찾지 못했습니다"
    rm -rf "$LLAMA_DIR"; mv "$LLAMA_DIR.tmp" "$LLAMA_DIR"
    say "llama-server 설치: $LLAMA_BIN"
  fi
fi
[ -x "$LLAMA_BIN" ] || die "llama-server 실행 파일이 없습니다: $LLAMA_BIN"
export LD_LIBRARY_PATH="$(dirname "$LLAMA_BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ------------------------------------------------------------------ 4. 모델
if [ -z "$MODEL_PATH" ]; then
  MODEL_PATH="$W/models/$(basename "$MODEL_URL")"
  if [ ! -s "$MODEL_PATH" ] || [ -f "$MODEL_PATH.part" ]; then
    FREE_KB="$(df -Pk "$W/models" | awk 'NR==2{print $4}')"
    [ "${FREE_KB:-0}" -gt $(( MODEL_SIZE / 1024 + 512000 )) ] || die "디스크 공간 부족: 모델에 약 4.7GB 필요"
    say "모델 내려받기 (Qwen2.5-7B-Instruct Q4_K_M, 4.7GB)"
    fetch "$MODEL_URL" "$MODEL_PATH"
  fi
fi
[ -s "$MODEL_PATH" ] || die "모델 파일이 없습니다: $MODEL_PATH"
SZ="$(stat -c %s "$MODEL_PATH")"
if [ "$MODEL_PATH" = "$W/models/$(basename "$MODEL_URL")" ] && [ "$SZ" != "$MODEL_SIZE" ]; then
  die "모델 크기 불일치 ($SZ / $MODEL_SIZE 바이트) — rm '$MODEL_PATH' 후 다시 실행"
fi

# ------------------------------------------------------------------ 5. 역할별 준비
if [ "$ROLE" = server ]; then
  SMI="$(command -v nvidia-smi)"
  if [ "$(id -u)" = 0 ]; then
    warn "root 로 실행 중입니다. 일반 사용자로 실행하고 nvidia-smi 만 sudo 허용하는 방식을 권장합니다"
  elif sudo -n -l "$SMI" >/dev/null 2>&1; then
    say "전력 한도 제어: sudo nvidia-smi 허용됨"
  else
    say "3080 은 'nvidia-smi -pl' 로 전력 한도를 걸 수 있지만 root 권한이 필요합니다."
    say "  추가할 규칙: $USER ALL=(root) NOPASSWD: $SMI  (/etc/sudoers.d/w2b-nvidia-smi)"
    if yesno "sudoers 규칙을 추가할까요? (안 하면 duty 방식으로 동작)" Y; then
      TMP="$(mktemp)"; echo "$USER ALL=(root) NOPASSWD: $SMI" > "$TMP"
      if sudo visudo -cf "$TMP" >/dev/null; then
        sudo install -m 440 -o root -g root "$TMP" /etc/sudoers.d/w2b-nvidia-smi && say "추가 완료"
      else warn "규칙 검증 실패 — 추가하지 않았습니다"; fi
      rm -f "$TMP"
    fi
  fi
else
  for ps in /sys/class/power_supply/*; do
    [ -f "$ps/type" ] && [ "$(cat "$ps/type")" = Mains ] && [ -f "$ps/online" ] && \
      { [ "$(cat "$ps/online")" = 1 ] || warn "충전기가 연결되지 않았습니다 — 배터리 구동 시 GPU 전력이 크게 제한됩니다"; }
  done
  if command -v powerprofilesctl >/dev/null; then
    PROF="$(powerprofilesctl get 2>/dev/null || true)"
    if [ -n "$PROF" ] && [ "$PROF" != performance ]; then
      if yesno "전원 프로파일이 '$PROF' 입니다. performance 로 바꿀까요?" Y; then
        powerprofilesctl set performance && say "전원 프로파일: performance" || warn "변경 실패"
      fi
    fi
  fi
fi

# ------------------------------------------------------------------ 6. 점검 → 실행
ARGS=(--orch "$ORCH" --id "$NODE_ID" --system-w "$SYSTEM_W" --gpu-index "$GPU_INDEX"
      --llama-bin "$LLAMA_BIN" --gguf "$MODEL_PATH")
[ -n "$TOKEN" ]    && ARGS+=(--token "$TOKEN")
[ -n "$SOLAR_KW" ] && ARGS+=(--solar-kw "$SOLAR_KW")

if [ "$SKIP_CHECK" != 1 ]; then
  say "점검 실행 (모델 로딩 때문에 수십 초 걸릴 수 있음)"
  if ! python3 "$ROOT/agent.py" "${ARGS[@]}" --check; then
    [ "$CHECK_ONLY" = 1 ] && exit 1
    yesno "점검에 실패 항목이 있습니다. 그래도 시작할까요?" N || exit 1
  fi
fi
[ "$CHECK_ONLY" = 1 ] && exit 0

trap 'echo; say "종료합니다"; exit 0' INT TERM
say "노드 $NODE_ID 시작 → $ORCH   (종료: Ctrl+C)"
while true; do
  set +e; python3 "$ROOT/agent.py" "${ARGS[@]}"; code=$?; set -e
  [ "$code" = 0 ] && break
  warn "에이전트 종료(코드 $code) — 5초 뒤 재시작"
  sleep 5
done
