# 训练显存与算力优化说明

本文档说明 2026-04-09 对训练链路做的性能优化，覆盖 `train.py`、`train_poly.py`、`train_hybrid.py` 以及所有相关自注意力模块。

## 本次优化解决了什么

### 1. 可选 AMP 混合精度训练

- 三个训练脚本都新增了 `--amp` 开关。
- 支持 `--amp_dtype fp16|bf16`。
- 当选择 `fp16` 时自动启用 `GradScaler`。
- 当选择 `bf16` 时不使用 `GradScaler`；若当前 GPU 不支持 `bf16`，会自动回退到 `fp16` 并打印警告。

### 2. 使用 PyTorch fused SDPA，自动命中 Flash Attention

- `IMFuse.py`
- `IMFuse_no1skip.py`
- `IMFuse_hybrid.py`
- `mambavision_mixer.py`

以上文件中的自注意力实现，已从手写的 `q @ k^T -> softmax -> attn @ v` 路径切换为 `torch.nn.functional.scaled_dot_product_attention`。

在 PyTorch 2.5.1 + CUDA 环境下，这允许运行时自动选择：

- Flash Attention kernel
- memory-efficient attention kernel
- math fallback kernel

是否允许 Flash kernel 由 `--flash_attention/--no-flash_attention` 控制，默认允许。

### 3. 修复 epoch 统计造成的隐性显存占用

原训练代码将 `loss`、`dice_loss`、`cross_loss` 等带计算图的 tensor 直接累加到 epoch 统计变量中，例如：

```python
loss_epoch += loss
```

这会让整轮训练的计算图持续被引用，导致：

- GPU 显存长期不释放
- Python 侧内存增长
- AMP 带来的节省被部分抵消

现在改为只累加 `.item()` 后的标量值，图会在每个 iteration 结束后正常释放。

### 4. 开启更适合固定输入尺寸的 CUDA 后端优化

新增并默认启用：

- `--tf32`：允许 Ampere 及以上 GPU 使用 TF32 tensor core
- `--cudnn_benchmark`：固定输入尺寸时自动选择更快的卷积算法
- `--matmul_precision high`：设置 PyTorch 的 float32 matmul precision hint

## 新增命令行参数

三个训练脚本统一新增以下参数：

```bash
--amp
--amp_dtype {fp16,bf16}
--flash_attention / --no-flash_attention
--tf32 / --no-tf32
--cudnn_benchmark / --no-cudnn_benchmark
--matmul_precision {highest,high,medium}
```

默认行为：

- AMP 默认关闭
- Flash Attention 默认允许
- TF32 默认开启
- cuDNN benchmark 默认开启
- `matmul_precision=high`

## 推荐启动方式

### 1. 通用推荐

适合大多数支持 Tensor Core 的 NVIDIA GPU：

```bash
python train_poly.py \
  --datapath /root/MV-IM-Fuse/dataset/BRATS2023_npy \
  --dataname BRATS2023 \
  --savepath /root/MV-IM-Fuse/checkpoints/exp_amp \
  --amp \
  --amp_dtype fp16
```

### 2. 若 GPU 对 BF16 支持较好

例如 A100 / H100 一类卡：

```bash
python train_hybrid.py \
  --datapath /root/MV-IM-Fuse/dataset/BRATS2023_npy \
  --dataname BRATS2023 \
  --savepath /root/MV-IM-Fuse/checkpoints/hybrid_amp_bf16 \
  --stage 1 \
  --amp \
  --amp_dtype bf16
```

### 3. 若需要排查数值稳定性问题

先关闭 AMP，仅保留 fused attention 和 TF32：

```bash
python train.py \
  --datapath /root/MV-IM-Fuse/dataset/BRATS2023_npy \
  --dataname BRATS2023 \
  --savepath /root/MV-IM-Fuse/checkpoints/baseline_fp32
```

若怀疑 Flash kernel 与当前驱动 / CUDA 组合存在兼容问题，可显式关闭：

```bash
python train.py \
  --datapath /root/MV-IM-Fuse/dataset/BRATS2023_npy \
  --dataname BRATS2023 \
  --savepath /root/MV-IM-Fuse/checkpoints/no_flash \
  --amp \
  --no-flash_attention
```

## Checkpoint 兼容性

现在训练脚本会额外保存 `scaler_dict`：

- `train.py` 的 `model_last.pth` / `best.pth`
- `train_poly.py` 的 `model_last.pth` / `best.pth`
- `train_hybrid.py` 的 `stage*_last.pth` / `stage*_best.pth` / `stage*_final.pth`

恢复训练时：

- 如果 checkpoint 中存在 `scaler_dict`，会自动恢复 AMP scaler 状态。
- 老 checkpoint 没有该字段时也能继续加载，不影响兼容性。

## 预期收益

实际收益取决于 GPU、batch size 和 missing-modality 组合，但通常可以期待：

- AMP: 明显降低 activation 显存
- fused SDPA/Flash Attention: 降低 attention 显存占用并提升吞吐
- 标量化 epoch 统计: 避免每个 epoch 内显存和内存逐步膨胀
- TF32 + cuDNN benchmark: 在固定 patch 尺寸 `128^3` 下提升卷积和 matmul 速度

## 已知限制

- Flash Attention 是否真正被命中，由 GPU 架构、CUDA 版本、head size、dtype 和 PyTorch 后端共同决定。
- `scaled_dot_product_attention` 已接入，但如果运行条件不满足，PyTorch 会自动回退到 memory-efficient 或 math kernel。
- `mamba_ssm` 路径本身没有被替换为更激进的 fused 实现；本次优化重点放在训练精度策略和 attention kernel。

## 建议的后续排查顺序

如果仍然显存紧张，建议按以下顺序排查：

1. 先启用 `--amp --amp_dtype fp16`
2. 确认没有意外关闭 `--flash_attention`
3. 降低 `batch_size`
4. 检查是否有额外的 TensorBoard 图像日志频率过高
5. 再考虑改 patch size 或做梯度累积