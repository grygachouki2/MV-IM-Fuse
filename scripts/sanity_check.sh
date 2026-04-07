#!/bin/bash
# Sanity Check 启动脚本
set -e

PRETRAINED=${1:-""}

echo "========== Sanity Check =========="

if [ -n "${PRETRAINED}" ]; then
    echo "使用预训练权重: ${PRETRAINED}"
    python sanity_check.py \
        --pretrained_imfuse ${PRETRAINED} \
        --num_mamba_blocks 1 \
        --num_attn_blocks 1 \
        --drop_path 0.1
else
    echo "使用随机初始化"
    python sanity_check.py \
        --num_mamba_blocks 1 \
        --num_attn_blocks 1 \
        --drop_path 0.1
fi

echo "Sanity Check 完成!"
