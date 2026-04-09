# IMFuseHybrid 阶段训练修复说明

## 已实施修复

根据 `docs/stage1_loss_stall_diagnosis.md` 中的诊断，本次已落地以下改动：

1. 分离 MV-Mixer 和 Attention 的结构超参数
   - `HybridTokenEncoder` 新增 `mamba_mlp_ratio` / `attn_mlp_ratio`
   - `HybridTokenEncoder` 新增 `mamba_layer_scale` / `attn_layer_scale`
   - `IMFuseHybrid`、`train_hybrid.py`、`test_hybrid.py` 已同步支持这些参数

2. 修复 Stage 1 的优化目标
   - Stage 1 不再把 `prm_loss` 纳入反向传播
   - `full_loss` 仍保留完整的 `fuse + sep + prm` 统计，便于对照训练曲线

3. 为 Stage 1/2 增加 warmup
   - 新增 `--warmup_epochs`
   - 未显式设置时，Stage 1/2 默认使用 `min(10, stage_epochs // 5)` 的线性 warmup

4. Stage 2 支持直接从 IM-Fuse 预训练初始化
   - `train_hybrid.py --stage 2 --pretrained_imfuse ...` 现在可直接加载 IM-Fuse 预训练权重

5. 处理无 Attention block 的权重迁移边界情况
   - `num_attn_blocks == 0` 时不再尝试映射不存在的 Attention block 权重

## 默认推荐配置

当前脚本默认采用以下更稳妥的设置：

- `mamba_mlp_ratio=4.0`
- `attn_mlp_ratio=8.0`
- `mamba_layer_scale=1e-5`
- `attn_layer_scale=0.0`（即不启用 attention residual layer scale）
- `warmup_epochs=10`

这些设置的目标是：

- 让新插入的 MV-Mixer 初始时近似残差旁路，避免破坏预训练特征流
- 保留 Attention block 对 IM-Fuse FFN 权重的完整复用能力
- 降低随机初始化 MLP 对 Stage 1 的噪声注入

## 推荐启动方式

### 方案 A：修复后的三阶段训练

```bash
bash scripts/train_e2_hybrid1.sh 1 <DATAPATH> <IMFUSE_CKPT>
bash scripts/train_e2_hybrid1.sh 2 <DATAPATH>
bash scripts/train_e2_hybrid1.sh 3 <DATAPATH>
```

### 方案 B：从 Stage 2 直接联合训练

```bash
python train_hybrid.py \
    --datapath <DATAPATH> \
    --dataname BRATS2023 \
    --savepath <OUTPUT_DIR> \
    --stage 2 \
    --stage2_epochs 150 \
    --pretrained_imfuse <IMFUSE_CKPT> \
    --num_mamba_blocks 1 \
    --num_attn_blocks 1 \
    --mamba_mlp_ratio 4.0 \
    --attn_mlp_ratio 8.0 \
    --mamba_layer_scale 1e-5 \
    --warmup_epochs 10 \
    --mamba_skip \
    --first_skip
```

## 变更文件

- `mambavision_mixer.py`
- `IMFuse_hybrid.py`
- `train_hybrid.py`
- `test_hybrid.py`
- `scripts/train_e1_stage1.sh`
- `scripts/train_e1_mamba_only.sh`
- `scripts/train_e2_hybrid1.sh`
- `scripts/train_e3_hybrid2.sh`