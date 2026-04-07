#!/bin/bash
# E2 Hybrid-1 (MV-Mixer × 1 + Attn × 1) 三阶段训练脚本
#
# 用法:
#   bash scripts/train_e2_hybrid1.sh <STAGE> <DATAPATH> <PRETRAINED_CKPT>
#
# 示例:
#   bash scripts/train_e2_hybrid1.sh 1 /path/to/BraTS2023_npy /path/to/imfuse_best.pth
#   bash scripts/train_e2_hybrid1.sh 2 /path/to/BraTS2023_npy
#   bash scripts/train_e2_hybrid1.sh 3 /path/to/BraTS2023_npy

set -e

STAGE=${1:-1}
DATAPATH=${2:?"请提供数据路径"}
PRETRAINED=${3:-""}

SAVEPATH="output/e2_hybrid1"
COMMON_ARGS="
    --datapath ${DATAPATH}
    --dataname BRATS2023
    --savepath ${SAVEPATH}
    --lr 2e-4
    --weight_decay 3e-5
    --batch_size 1
    --seed 999
    --num_mamba_blocks 1
    --num_attn_blocks 1
    --drop_path 0.1
    --mamba_skip
    --first_skip
    --val_interval 10
"

if [ "${STAGE}" = "1" ]; then
    echo "========== Stage 1: Warm-up MV-Mixer (50 epochs) =========="
    STAGE_ARGS="--stage 1 --stage1_epochs 50"
    if [ -n "${PRETRAINED}" ]; then
        STAGE_ARGS="${STAGE_ARGS} --pretrained_imfuse ${PRETRAINED}"
    fi
    python train_hybrid.py ${COMMON_ARGS} ${STAGE_ARGS}

elif [ "${STAGE}" = "2" ]; then
    echo "========== Stage 2: Fine-tune Token Encoder (100 epochs) =========="
    python train_hybrid.py ${COMMON_ARGS} \
        --stage 2 \
        --stage2_epochs 100 \
        --stage_resume ${SAVEPATH}/stage1_final.pth

elif [ "${STAGE}" = "3" ]; then
    echo "========== Stage 3: End-to-end Fine-tune (300 epochs) =========="
    python train_hybrid.py ${COMMON_ARGS} \
        --stage 3 \
        --stage3_epochs 300 \
        --stage_resume ${SAVEPATH}/stage2_final.pth

else
    echo "错误: STAGE 必须为 1, 2, 或 3"
    exit 1
fi

echo "Stage ${STAGE} 完成!"
