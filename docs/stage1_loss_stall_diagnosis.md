# IMFuseHybrid Stage 1 训练 Loss 不下降 — 诊断报告

## 1. 现象

Stage 1 训练 3 个 Epoch 后，Loss 基本不下降：

| Epoch | Total Loss | fuse_cross | fuse_dice | prm_cross | prm_dice |
|-------|-----------|-----------|-----------|-----------|----------|
| 1 | 6.0783 | 0.5435 | 0.7412 | 1.8234 | 2.9702 |
| 2 | 5.8977 | 0.5163 | 0.7348 | 1.6977 | 2.9489 |
| 3 | 5.8569 | 0.5086 | 0.7336 | 1.6688 | 2.9459 |

关键特征：
- `fuse_dice ≈ 0.74` — 锁定在 4 类分割的随机预测基线（1 - 1/num_cls = 0.75）
- `prm_dice ≈ 2.95` — 4 个金字塔层 × ~0.74 per level，同样是随机基线
- `fuse_cross` 和 `prm_cross` 有微弱下降，但极其缓慢
- 总 Loss 的 50% 以上来自 `prm_dice`（不可优化的常量偏移）

## 2. 根因分析

### 2.1 【致命】无 Layer Scale → 随机初始化的 MV-Mixer 立即破坏预训练特征流

**这是最核心的问题。**

当前配置 `--hybrid_layer_scale 0.0` → 代码中 `layer_scale=None` → `gamma_1 = gamma_2 = 1.0`（标量，非可训练参数）。

HybridBlock 的前向传播：
```python
x = x + self.gamma_1 * self.mixer(self.norm1(x))   # gamma_1 = 1.0
x = x + self.gamma_2 * self.mlp(self.norm2(x))      # gamma_2 = 1.0
```

**问题**：Block 0（MV-Mixer）是全新随机初始化的，包括：
- `MambaVisionMixer3D`：SSM 分支 + Gate 分支 → 输出随机噪声
- `MLP`（512→4096→512）：~4.2M 参数/模态，Kaiming 初始化 → 输出量级大

这些随机输出以 **满权重（gamma=1.0）** 加到残差流上：
```
x_in (pretrained 好的 512-dim features)
  → 经过 MV-Mixer block 0：x_out = x_in + random_noise_1 + random_noise_2
  → 到达 frozen Attention block 1：处理的是已被噪声破坏的 tokens
  → 整个下游 pipeline 得到的都是垃圾特征
```

**对比 MambaVision 原论文**：MambaVision 是从头训练 300 epochs，所有层同时学习，没有预训练→微调场景。本项目是将 MV-Mixer 插入到已预训练的 IM-Fuse pipeline 中，必须保证初始时新模块对特征流的影响接近零。

**类比**：这等价于在一个训练好的 ResNet 中间随机插入一层全连接层，不做任何衰减，直接端到端——预训练的所有层输出立刻失效。

### 2.2 【致命】MV-Mixer Block 的 MLP 占可训练参数 88%，且全部随机初始化

Stage 1 可训练参数分析：

| 组件 | 参数量/模态 | 4 模态总计 | 占比 |
|------|-----------|-----------|------|
| MambaVisionMixer3D (SSM) | ~554K | ~2.2M | 11.6% |
| MLP (512→4096→512) | ~4.2M | ~16.8M | 88.3% |
| LayerNorm × 2 | ~2K | ~8K | <0.1% |
| **总可训练参数** | **~4.76M** | **~19.0M** | **22.8% of model** |

Block 0 的 MLP 使用了 `mlp_ratio=8.0`（512→4096→512），与 IM-Fuse 原始 Transformer 的 FFN 维度一致。但关键区别是：**IM-Fuse 的 FFN 权重被映射到了 Block 1（Attention block）的 MLP**，而 Block 0 的 MLP 是全新的，无预训练权重。

这 16.8M 个随机参数的输出通过残差连接（gamma=1.0）叠加到 token 特征上，造成的扰动远大于 SSM 分支本身。

### 2.3 【严重】梯度路径过长，信号衰减严重

从 Loss 到可训练的 MV-Mixer Block 0 的梯度路径：

```
Loss
  ↓ backward through
Decoder_fuse (frozen, 含 RFM5→RFM1 + 多级 Conv + Upsample)
  ↓
multimodal_decode_conv (frozen, Conv1×1)
  ↓
Multimodal Transformer (frozen, Self-Attention + FFN)
  ↓
MambaFusionLayer (frozen, SSM-based 融合层)
  ↓
Tokenize + Masker (无参数，但涉及 reshape/indexing)
  ↓
flair_decode_conv → reshape (frozen)
  ↓
HybridTokenEncoder Block 1 — Attention (frozen, pretrained)
  ↓
HybridTokenEncoder Block 0 — MV-Mixer (TRAINABLE ← 梯度最终到达这里)
```

至少经过 **8 个冻结模块组**，其中 SSM-based MambaFusionLayer 因其序列扫描特性，特别容易导致梯度消失。

### 2.4 【严重】Loss 组成不合理：被不可优化的常量项主导

当前 Stage 1 的 loss：
```python
loss = fuse_loss + prm_loss    # sep_loss 已正确排除
```

但 `prm_loss = prm_cross + prm_dice`，其中 `prm_dice ≈ 3.0` 来自 4 个金字塔层的辅助预测。由于 Decoder_fuse 被冻结且接收破坏的特征，金字塔 loss 基本是常量，对优化没有帮助——但它占总 loss 的约 50%，严重稀释了有用的梯度信号。

### 2.5 【中等】训练策略问题

- **Poly LR + 仅 50 epochs**：`lr × (1 - epoch/50)^0.9` 衰减较快，在模型还没学会有效特征时 lr 已经很小了
- **batch_size=1**：梯度噪声极大，对本已困难的优化进一步雪上加霜
- **RAdam 优化器**：对于这种 warmup 场景，RAdam 的自适应特性或许不如带 warmup 的 AdamW

## 3. 思路层面的问题

### 3.1 Stage 1 "仅训练 MV-Mixer" 的设计目标不可达

Stage 1 的意图是 warm up MV-Mixer，让它学会处理 tokens，然后在 Stage 2 再联合训练 Attention。但这个设计有根本矛盾：

1. MV-Mixer → Attention 的串联结构意味着：MV-Mixer 的输出要经过 frozen Attention 才能影响最终预测
2. frozen Attention 期望接收"pretrained 质量"的特征——但 MV-Mixer 还没学会
3. MV-Mixer 只能通过"frozen Attention 处理后的输出"间接获得梯度信号——信号太弱

**核心矛盾**：你要求 MV-Mixer 在完全不了解下游期望的情况下，独立学会产生"恰好能让 frozen Attention + frozen Decoder 正常工作"的输出。这个目标太苛刻了。

### 3.2 与 MambaVision 论文设计的差异

MambaVision 原论文是 4 阶段层次架构，每个阶段从头一起训练。**Hybrid pattern 是在 Stages 3/4 内，前 N/2 用 MV-Mixer，后 N/2 用 Self-Attention**。所有块同时训练 300 epochs。

本项目的 3 阶段迁移训练策略（先 MV-Mixer → 再加 Attention → 再全部）是原创设计，没有 MambaVision 论文的直接支持。分阶段冻结训练更适合 adapter/LoRA 等低秩插入场景，不太适合全规模新模块的插入。

## 4. 推荐修复方案

### 方案 A：最小改动修复

#### A1. 为 MV-Mixer blocks 启用 Layer Scale

修改 `HybridTokenEncoder`，对 Mamba blocks 和 Attention blocks 使用不同的 layer_scale：

```python
# mambavision_mixer.py - HybridTokenEncoder.__init__
for block_index in range(num_mamba_blocks):
    blocks.append(HybridBlock(
        ...,
        layer_scale=1e-5,   # MV-Mixer: 近零初始化
    ))
for block_index in range(num_attn_blocks):
    blocks.append(HybridBlock(
        ...,
        layer_scale=None,    # Attention: gamma=1.0, 保持预训练行为
    ))
```

**原理**：layer_scale=1e-5 使 MV-Mixer 初始输出被衰减到接近零，HybridTokenEncoder 初始行为等价于原始 Transformer（只有 Attention block 工作）。随着训练进行，gamma 逐渐增大，MV-Mixer 逐步介入。

需要同步修改 HybridTokenEncoder 构造函数，新增 `mamba_layer_scale` 和 `attn_layer_scale` 参数。

#### A2. Stage 1 移除金字塔 loss

```python
# train_hybrid.py 训练循环中
if stage == 1:
    loss = fuse_loss  # 仅用主预测 loss
else:
    loss = fuse_loss + prm_loss
```

**原理**：Stage 1 只训练 MV-Mixer，金字塔 loss 来自深层 decoder，信号到 MV-Mixer 时已严重衰减且含噪。仅用 `fuse_loss` 提供更干净的梯度。

#### A3. 添加 Learning Rate Warmup

```python
# 前 5-10 个 epoch 线性 warmup
warmup_epochs = min(10, stage_epochs // 5)
if epoch < warmup_epochs:
    warmup_factor = (epoch + 1) / warmup_epochs
    for pg in optimizer.param_groups:
        pg['lr'] = pg['lr'] * warmup_factor
```

### 方案 B：调整训练策略（如果方案 A 效果仍不理想）

#### B1. 合并 Stage 1 和 Stage 2

直接从 Stage 2 开始：同时训练 MV-Mixer + Attention blocks。这样两者可以协同学习，避免 MV-Mixer 必须独立适应 frozen Attention 的困境。

```bash
python train_hybrid.py \
    --stage 2 \
    --stage2_epochs 150 \
    --pretrained_imfuse checkpoints/model_last.pth \
    --hybrid_layer_scale 1e-5 \
    ...
```

注意：需要修改 `train_hybrid.py` 使 Stage 2 也支持从 IM-Fuse 预训练权重初始化（当前仅 Stage 1 支持 `--pretrained_imfuse`）。

#### B2. 降低 MV-Mixer MLP ratio

当前 `mlp_ratio=8.0` 是为了复用 IM-Fuse FFN 权重（匹配 attention block 的 MLP）。但 MV-Mixer block 的 MLP 无预训练权重，使用 `mlp_ratio=4.0` 甚至 `2.0` 可减少随机噪声量：

| mlp_ratio | MLP 参数/模态 | 效果 |
|-----------|-------------|------|
| 8.0 | 4.2M | 当前值，随机噪声巨大 |
| 4.0 | 2.1M | 减半参数和噪声 |
| 2.0 | 1.1M | 进一步降低 |

注意：这不影响 attention block 的 MLP（仍可用 8.0 以复用预训练权重）。需要分别配置。

