#!/usr/bin/env bash
# RTX 3080 서버(Linux)용: git clone 후 ./start_server_3080.sh
# 처음 한 번 설정을 묻고, llama.cpp·모델을 받아 점검한 뒤 노드를 켭니다.  (설명: README.md)
W2B_ROLE=server exec "$(dirname "$0")/scripts/node_linux.sh" "$@"
