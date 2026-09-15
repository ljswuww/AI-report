#!/bin/bash
# download.sh - Download Qwen2.5-14B-Instruct (ModelScope, fast in China)
set -e

# Upgrade modelscope; older CLI versions have bugs
pip install -U modelscope

# Download, with concurrency parameters to improve speed
modelscope download \
    --model Qwen/Qwen2.5-14B-Instruct \
    --local_dir ./qwen2.5-14b-instruct \