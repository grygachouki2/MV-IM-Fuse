# 3D MambaVision-lite × IM-Fuse 实验计划

## 1. 项目背景与目标

<aside>
🎯

**核心假设**：在保持 IM-Fuse 的 I-MFB 融合、skip connection、multimodal Transformer、decoder、loss 和训练流程完全不变的前提下，仅将每个模态分支的 Intra-modal Transformer 替换为 **MambaVision-lite Hybrid Token Encoder**，即可稳定提升缺失模态脑肿瘤分割性能。

</aside>

### 1.1 研究动机

- **MambaVision** 论文证明：在视觉 backbone 中，浅层用 CNN 做快速特征提取、深层用 Mamba mixer + 最后层用 Self-Attention 的混合模式（**Late Attention**），能在精度和吞吐量上同时超越纯 Transformer 和纯 Mamba 模型
- **IM-Fuse** 论文证明：Mamba 的选择机制和长序列建模能力可以有效应对缺失模态下的多模态融合问题，其 **Interleaved Mamba Fusion Block (I-MFB)** 在 BraTS2023 上达到 SOTA
- 但 IM-Fuse 的单模态 token encoder 仍然沿用传统的 Intra-modal Transformer，**尚未利用 MambaVision 的混合 token mixing 策略来增强单模态表征**

### 1.2 实验目标

- [ ]  验证 MambaVision-lite Hybrid Token Encoder 对单模态 bottleneck 表征的增强效果
- [ ]  在 BraTS2023 的 15 种模态缺失组合下全面评估
- [ ]  特别关注 **T1c 缺失场景下 ET（Enhancing Tumor）** 的表现
- [ ]  确保显存增长 ≤ 15%、训练吞吐下降 ≤ 20%

---

## 2. 理论基础

### 2.1 MambaVision 核心设计

MambaVision 是第一个将 Mamba 与 Transformer 结合的视觉混合 backbone，其关键发现包括：

| **设计要素** | **具体方案** | **效果** |
| --- | --- | --- |
| 宏观架构 | 4 阶段层级结构：Stage 1-2 为 CNN ResBlock，Stage 3-4 为 Mamba + Transformer | 高分辨率快速提取 + 低分辨率全局建模 |
| MambaVision Mixer | 用 **regular conv** 替代 causal conv；增加 **对称 non-SSM 分支**；两路 **concat** 后投影 | ImageNet Top-1 从 80.5% → 82.3%（+1.8%） |
| Hybrid Pattern | 每 stage 的前 N/2 层用 MV-Mixer，后 N/2 层用 Self-Attention（**Last N/2**） | 优于 Random/First N/2/Mixed 等所有其他模式 |
| Self-Attention | 多头注意力 + 窗口机制（Stage 3: window=14, Stage 4: window=7） | 捕获全局上下文和长距离依赖 |

### 2.2 IM-Fuse 核心设计

IM-Fuse 利用 Mamba 作为多模态融合机制，在 BraTS2023 上达到 SOTA：

- **Hybrid Modality-specific Encoder**：Conv Encoder → Linear Tokenization → Intra-modal Transformer
- **Bernoulli Masking**：训练时随机置零缺失模态的特征
- **I-MFB（Interleaved Mamba Fusion Block）**：交错排列各模态 token 与 learnable token，通过 Mamba 融合
- **Multimodal Transformer**：融合后的表征再经 Transformer 建模长距离依赖
- **Conv Decoder**：对称 U-Net 解码器 + skip connections
- **Loss**：Dice + Weighted CE，附加 shared-weight 辅助 decoder

### 2.3 本方案的切入点

<aside>
💡

**改动范围极小**：仅替换 IM-Fuse 中每个模态分支的 Intra-modal Transformer（1 个 Transformer Block）→ MambaVision-lite Hybrid Token Encoder（MV-Mixer × k + Transformer × 1）。其余所有模块完全不变。

</aside>

---

## 3. 实验矩阵设计

### 3.1 模型版本定义

| **版本** | **Intra-modal Token Encoder 结构** | **目的** | **优先级** |
| --- | --- | --- | --- |
| **Baseline** | 原始 Intra-modal Transformer × 1 | 复现 IM-Fuse 基线 | 🔴 必做 |
| **E1: Mamba-only** | MV-Mixer × 2（无 Transformer） | 验证纯 Mamba mixer 是否足够 | 🟡 次要 |
| **E2: Hybrid-1** ⭐ | MV-Mixer × 1 + Transformer × 1 | **主方案**，最低成本验证 Late Attention | 🔴 必做 |
| **E3: Hybrid-2** | MV-Mixer × 2 + Transformer × 1 | 验证增加 mixer 层数的边际收益 | 🟡 次要 |

### 3.2 实验执行顺序

1. **Phase 0**：Sanity check（加载 IM-Fuse 检查点 → 构建 Hybrid 模型 → 单卡 overfit 20 case，验证前向/反向传播正确）
2. **Phase 1**：Baseline 确认（直接使用 IM-Fuse 检查点评估，确认基线性能）
3. **Phase 2**：**E2 (Hybrid-1) 三阶段迁移训练**
    - Stage 1: Warm-up MV-Mixer（~30–50 epochs，冻结预训练部分）
    - Stage 2: Fine-tune Token Encoder（~50–100 epochs，解冻 Hybrid Encoder Transformer）
    - Stage 3: End-to-end Fine-tune（~200–300 epochs，全部解冻）
    - ⏱ 每阶段结束后评估 val 曲线，若 Stage 1 无正面信号则暂停
4. **Phase 3**：E1 (Mamba-only) 和 E3 (Hybrid-2) 使用相同三阶段策略并行训练
5. **Phase 4**：全面评估 + 消融分析（含阶段间 checkpoint 对比）

---

## 4. 训练协议

### 4.1 数据集

| **项目** | **配置** |
| --- | --- |
| 数据集 | BraTS2023-GLI（1,251 volumes） |
| 划分 | 70% train / 10% val / 20% test（与 IM-Fuse 相同 split） |
| 模态 | FLAIR, T1, T1c, T2 |
| 分割目标 | ET（Enhancing Tumor）、TC（Tumor Core）、WT（Whole Tumor） |
| 预处理 | 与 mmFormer 一致（IM-Fuse 沿用） |
| 数据增强 | 与 mmFormer 一致 |

### 4.2 迁移学习策略

<aside>
🔑

**核心思路**：利用已有的 IM-Fuse 预训练检查点，通过 **三阶段渐进解冻 + 分组学习率**，让新增的 MV-Mixer 模块快速适配，同时保护已学习的特征，大幅减少总训练 epoch 并节省计算资源。

</aside>

#### 4.2.1 三阶段渐进解冻

| **阶段** | **Epochs** | **冻结模块** | **可训练模块** | **目的** |
| --- | --- | --- | --- | --- |
| **Stage 1: Warm-up MV-Mixer** | ~30–50 | Conv Encoder, Intra-modal Transformer, I-MFB, Multimodal Transformer, Decoder | **仅 MV-Mixer Block(s)** | 新增模块快速收敛，不破坏预训练特征 |
| **Stage 2: Fine-tune Token Encoder** | ~50–100 | Conv Encoder, I-MFB, Multimodal Transformer, Decoder | **MV-Mixer + Hybrid Encoder 内的 Transformer** | MV-Mixer 与 Late Attention 联合适配 |
| **Stage 3: End-to-end Fine-tune** | ~200–300 | （无） | **全部模块** | 端到端微调，全局优化 |

**总训练 epoch：~280–450**（对比原始从头训练 1000 epochs，**节省 55%–72%**）

#### 4.2.2 分组学习率设置

每个阶段为不同模块组设置差异化学习率，保护低层已学习特征，加速新模块适配：

**Stage 1: Warm-up MV-Mixer（~30–50 epochs）**

| **参数组** | **学习率** | **说明** |
| --- | --- | --- |
| MV-Mixer Blocks | 2 × 10⁻⁴（base LR） | 新增模块，正常学习率 |
| 其余所有模块 | 0（frozen） | 完全冻结，不更新 |

**Stage 2: Fine-tune Token Encoder（~50–100 epochs）**

| **参数组** | **学习率** | **说明** |
| --- | --- | --- |
| MV-Mixer Blocks | 1 × 10⁻⁴ | 已预热，适当降低 |
| Hybrid Encoder Transformer | 5 × 10⁻⁵（0.25× base） | 预训练权重，小心微调 |
| 其余所有模块 | 0（frozen） | 继续冻结 |

**Stage 3: End-to-end Fine-tune（~200–300 epochs）**

| **参数组** | **学习率** | **说明** |
| --- | --- | --- |
| MV-Mixer Blocks | 5 × 10⁻⁵ | 持续学习 |
| Hybrid Encoder Transformer | 2 × 10⁻⁵（0.1× base） | 精细调整 |
| I-MFB + Multimodal Transformer | 1 × 10⁻⁵（0.05× base） | 融合模块，极小学习率保护 |
| Conv Encoder | 5 × 10⁻⁶（0.025× base） | 底层特征已充分学习，几乎冻结 |
| Conv Decoder + Aux Decoder | 1 × 10⁻⁵（0.05× base） | 适应新表征 |


#### 4.2.3 各阶段 LR Scheduler

每个阶段内独立使用 Poly scheduler，阶段切换时重建 optimizer 和 scheduler.

---

## 5. 风险控制

### 5.1 风险：收益太小

**原因**：仅改动单模态 token encoder，未触及 fusion 主体

**控制**：

- 迁移学习 Stage 1（~30–50 epochs）结束后即可观察 val 曲线和 T1c-missing ET
- 若 Stage 1 后 MV-Mixer 输出无明显改善信号，暂停后续阶段
- 利用三阶段 checkpoint 做精细分析，定位瓶颈是否在新模块

### 5.2 风险：Mamba branch 训练不稳定

**原因**：3D 医学分割中 Mamba 在高分辨率/长序列下可能不稳定

**控制**：

- 第一版仅上 MV-Mixer × 1（Hybrid-1）
- Stage 1 中冻结预训练部分，新模块在稳定梯度环境中学习，降低不稳定风险
- 使用残差连接、LayerNorm、drop_path=0.1 的保守设置
- **先单卡 overfit 20 case** 确认训练稳定后再启动三阶段训练
- 每阶段保存 checkpoint，若某阶段出现训练崩溃可回退到上一阶段继续

### 5.3 风险：阶段切换时 loss 震荡

**原因**：解冻新参数时梯度突变，可能导致 loss spike

**控制**：

- 解冻预训练模块时使用极低学习率（0.05×–0.25× base LR）
- 阶段切换时重建 optimizer（清除 momentum 状态），避免旧动量干扰
- 可选：Stage 2/3 开始的前 5 epochs 使用 linear warmup 平滑过渡

### 5.4 风险：关键缺失场景退化

**原因**：缺失模态补偿不仅由 backbone 决定

**控制**：

- 必须单独监控 missing T1c → ET Dice
- 若此处退化，首选不是继续改 backbone，而是后续考虑加轻量补偿项
