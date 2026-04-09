#!/bin/bash
# 15 种缺失模态组合全面评估脚本
set -e

RESUME=${1:?"请提供模型检查点路径"}
DATAPATH=${2:?"请提供数据路径"}
SAVEPATH=${3:-"output/eval_results"}

# 模型配置 (根据实验版本修改)
NUM_MAMBA=${4:-1}
NUM_ATTN=${5:-1}

echo "========== 全面评估 =========="
echo "Checkpoint: ${RESUME}"
echo "Config: MV-Mixer×${NUM_MAMBA} + Attn×${NUM_ATTN}"

python test_hybrid.py \
    --dataname BRATS2023 \
    --datapath ${DATAPATH} \
    --savepath ${SAVEPATH} \
    --resume ${RESUME} \
    --num_mamba_blocks ${NUM_MAMBA} \
    --num_attn_blocks ${NUM_ATTN} \
    --drop_path 0.1 \
    --mamba_skip \
    --first_skip

echo "评估完成! 结果保存在 ${SAVEPATH}"
