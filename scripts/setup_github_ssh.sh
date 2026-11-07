#!/usr/bin/env bash
# GitHub push 할 때 아이디/토큰을 묻지 않도록 SSH 키를 설정 (Linux, 한 번만 실행)
#   1) ~/.ssh/id_ed25519 가 없으면 생성 (있으면 그대로 사용)
#   2) GitHub CLI(gh)로 로그인 → 공개키를 계정에 등록   (gh 가 없으면 설치 제안, 거절 시 웹 등록 안내)
#   3) github.com 호스트 키를 공식 지문과 대조해 known_hosts 에 추가
#   4) 이 저장소의 origin 을 https → ssh 주소로 변경, 접속 확인
# 사용: ./scripts/setup_github_ssh.sh
set -euo pipefail

say()  { printf '\033[1;34m[ssh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ssh] 주의:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[ssh] 오류:\033[0m %s\n' "$*" >&2; exit 1; }
yesno() { local a; read -r -p "  $1 [Y/n]: " a || true; [[ "${a:-Y}" =~ ^[Yy] ]]; }

[ -t 0 ] || die "터미널에서 직접 실행하세요 (GitHub 로그인 과정이 대화형입니다)"
command -v ssh-keygen >/dev/null || die "openssh 가 없습니다 (Arch: sudo pacman -S openssh / Ubuntu: sudo apt install openssh-client)"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="$HOME/.ssh/id_ed25519"
TITLE="$(whoami)@$(hostname)"

# ------------------------------------------------------------------ 1. 키
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
if [ -f "$KEY" ]; then
  say "기존 키 사용: $KEY"
else
  say "SSH 키 생성: $KEY (암호 없음 — push 할 때 아무것도 묻지 않음)"
  ssh-keygen -q -t ed25519 -N "" -f "$KEY" -C "$TITLE"
fi
PUB="$KEY.pub"

# ------------------------------------------------------------------ 2. GitHub 에 공개키 등록
manual_register() {
  say "아래 공개키를 복사해서 https://github.com/settings/ssh/new 에 붙여넣고 저장하세요 (Title: $TITLE)"
  echo; cat "$PUB"; echo
  read -r -p "  등록을 마쳤으면 Enter: " _ || true
}
if ! command -v gh >/dev/null; then
  if command -v pacman >/dev/null && yesno "GitHub CLI(gh)를 설치할까요? (sudo pacman -S github-cli)"; then
    sudo pacman -S --needed github-cli
  elif command -v apt-get >/dev/null && yesno "GitHub CLI(gh)를 설치할까요? (sudo apt install gh)"; then
    sudo apt-get install -y gh
  fi
fi
if command -v gh >/dev/null; then
  if ! gh auth status -h github.com >/dev/null 2>&1; then
    say "GitHub 로그인: 화면의 8자리 코드를 복사 → 브라우저에서 로그인 후 입력"
    gh auth login -h github.com -p ssh -s admin:public_key -w --skip-ssh-key
  elif ! gh auth status -h github.com 2>&1 | grep -q 'admin:public_key'; then
    say "키 등록 권한(admin:public_key) 추가 승인이 필요합니다"
    gh auth refresh -h github.com -s admin:public_key
  fi
  say "GitHub 계정: $(gh api user -q .login)"
  if out="$(gh ssh-key add "$PUB" --title "$TITLE" 2>&1)"; then
    say "공개키 등록 완료 (Title: $TITLE)"
  elif echo "$out" | grep -qi 'already'; then
    say "이미 등록된 키입니다"
  else
    warn "gh 로 등록 실패: $out"
    manual_register
  fi
else
  manual_register
fi

# ------------------------------------------------------------------ 3. github.com 호스트 키 확인
# GitHub 공식 Ed25519 지문 (docs.github.com "GitHub's SSH key fingerprints")
GH_FP="SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU"
KH="$HOME/.ssh/known_hosts"
if ! ssh-keygen -F github.com -f "$KH" >/dev/null 2>&1; then
  TMP="$(mktemp)"
  ssh-keyscan -t ed25519 github.com 2>/dev/null > "$TMP" || true
  FP="$(ssh-keygen -lf "$TMP" 2>/dev/null | awk '{print $2}')"
  if [ "$FP" = "$GH_FP" ]; then
    cat "$TMP" >> "$KH"; chmod 600 "$KH"; say "github.com 호스트 키 확인·등록 ($FP)"
  else
    rm -f "$TMP"; die "github.com 호스트 키 지문이 공식 값과 다릅니다 (받은 값: ${FP:-없음}). 네트워크를 확인하세요"
  fi
  rm -f "$TMP"
fi

# ------------------------------------------------------------------ 4. 저장소 주소 전환 + 확인
if git -C "$REPO_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  URL="$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || true)"
  if [[ "$URL" =~ ^https://([^@/]+@)?github\.com/([^/]+)/([^/]+)$ ]]; then
    NEW="git@github.com:${BASH_REMATCH[2]}/${BASH_REMATCH[3]%.git}.git"
    git -C "$REPO_DIR" remote set-url origin "$NEW"
    say "origin 변경: $URL → $NEW"
  elif [ -n "$URL" ]; then
    say "origin: $URL"
  fi
fi

out="$(ssh -T -o BatchMode=yes git@github.com 2>&1 || true)"
if echo "$out" | grep -q "successfully authenticated"; then
  say "성공: $(echo "$out" | grep -o 'Hi [^!]*!')  이제 git push 할 때 아무것도 묻지 않습니다"
else
  die "SSH 접속 확인 실패: $out"
fi
