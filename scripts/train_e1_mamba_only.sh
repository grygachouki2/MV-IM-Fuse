#!/bin/bash
# E1 Mamba-only (MV-Mixer × 2) 三阶段训练脚本
set -e

STAGE=${1:-1}
DATAPATH=${2:?"请提供数据路径"}
PRETRAINED=${3:-""}

SAVEPATH="output/e1_mamba_only"
COMMON_ARGS="
    --datapath ${DATAPATH}
    --dataname BRATS2023
    --savepath ${SAVEPATH}
    --lr 2e-4
    --weight_decay 3e-5
    --batch_size 1
    --seed 999
    --num_mamba_blocks 2
    --num_attn_blocks 0
    --drop_path 0.1
    --mamba_mlp_ratio 4.0
    --mamba_layer_scale 1e-5
    --warmup_epochs 10
    --mamba_skip
    --first_skip
    --val_interval 10
"

if [ "${STAGE}" = "1" ]; then
    echo "========== E1 Stage 1: Warm-up MV-Mixer (50 epochs) =========="
    STAGE_ARGS="--stage 1 --stage1_epochs 50"
    if [ -n "${PRETRAINED}" ]; then
        STAGE_ARGS="${STAGE_ARGS} --pretrained_imfuse ${PRETRAINED}"
    fi
    python train_hybrid.py ${COMMON_ARGS} ${STAGE_ARGS}

elif [ "${STAGE}" = "2" ]; then
    echo "========== E1 Stage 2: Fine-tune (100 epochs) =========="
    python train_hybrid.py ${COMMON_ARGS} \
        --stage 2 \
        --stage2_epochs 100 \
        --stage_resume ${SAVEPATH}/stage1_final.pth

elif [ "${STAGE}" = "3" ]; then
    echo "========== E1 Stage 3: End-to-end (300 epochs) =========="
    python train_hybrid.py ${COMMON_ARGS} \
        --stage 3 \
        --stage3_epochs 300 \
        --stage_resume ${SAVEPATH}/stage2_final.pth
fi
