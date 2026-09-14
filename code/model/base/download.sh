#!/bin/bash
# download.sh - 下载 Qwen2.5-14B-Instruct (魔搭ModelScope，国内高速)
set -e

# 升级modelscope，老版本cli会有bug
pip install -U modelscope

# 下载，增加并发参数提升速度
modelscope download \
    --model Qwen/Qwen2.5-14B-Instruct \
    --local_dir ./qwen2.5-14b-instruct \
