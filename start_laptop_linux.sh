#!/usr/bin/env bash
# RTX 4060 Laptop(Linux)용: git clone 후 ./start_laptop_linux.sh
# 처음 한 번 설정을 묻고, llama.cpp·모델을 받아 점검한 뒤 노드를 켭니다.  (설명: README.md)
W2B_ROLE=laptop exec "$(dirname "$0")/scripts/node_linux.sh" "$@"
