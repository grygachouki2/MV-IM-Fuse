# MambaVision-lite × IM-Fuse 详细实施方案

> 基于实验计划，结合 MambaVision 和 MV-IM-Fuse 两个仓库的源码分析，制定代码级别的实施方案。

---

## 目录

1. [现状分析：被替换模块的精确定义](#1-现状分析被替换模块的精确定义)
2. [新模块设计：MambaVision-lite Hybrid Token Encoder](#2-新模块设计mambavision-lite-hybrid-token-encoder)
3. [文件变更清单](#3-文件变更清单)
4. [Step 1：实现 MambaVision-lite 模块](#4-step-1实现-mambavision-lite-模块)
5. [Step 2：创建 IMFuse_hybrid 模型文件](#5-step-2创建-imfuse_hybrid-模型文件)
6. [Step 3：实现三阶段迁移训练脚本](#6-step-3实现三阶段迁移训练脚本)
7. [Step 4：检查点加载与权重迁移](#7-step-4检查点加载与权重迁移)
8. [Step 5：Sanity Check 流程](#8-step-5sanity-check-流程)
9. [Step 6：评估与消融分析](#9-step-6评估与消融分析)
10. [显存与吞吐量估算](#10-显存与吞吐量估算)
11. [完整实施时间线](#11-完整实施时间线)

---

## 1. 现状分析：被替换模块的精确定义

### 1.1 当前 IntraFormer 架构（IMFuse.py）

每个模态分支的 Intra-modal Token Encoder 包含以下组件：

```
x5 (B, 128, 8, 8, 8)                     # Conv Encoder 第5层输出
  │
  ├─ encode_conv: Conv3d(128 → 512, k=1)  # 通道投影
  │
  ├─ reshape: (B, 512, 8,8,8) → (B, 512, 512)  # 空间展平为 token 序列
  │
  ├─ + pos_embedding: (1, 512, 512)        # 可学习位置编码
  │
  ├─ Transformer(depth=1):                 # 单层 Transformer Block
  │   ├─ Residual(PreNormDrop(SelfAttention(dim=512, heads=8)))
  │   └─ Residual(PreNorm(FeedForward(512, 4096)))
  │
  ├─ reshape: (B, 512, 512) → (B, 512, 8,8,8)  # 恢复空间维度
  │
  └─ decode_conv: Conv3d(512 → 128, k=1)  # 通道反投影
```

**关键参数**（定义在 `IMFuse.py` 全局变量）：
| 参数 | 值 | 位置 |
|------|------|------|
| `transformer_basic_dims` | 512 | token 维度 |
| `mlp_dim` | 4096 | FFN 隐藏层维度 |
| `num_heads` | 8 | 注意力头数 |
| `depth` | 1 | Transformer 层数 |
| `patch_size` | 8 | 空间分辨率（8³ = 512 tokens） |

**每个模态独立拥有的组件**（以 `flair` 为例）：
- `self.flair_encode_conv` — Conv3d(128, 512, k=1)
- `self.flair_pos` — nn.Parameter(1, 512, 512)
- `self.flair_transformer` — Transformer(dim=512, depth=1, heads=8, mlp_dim=4096)
- `self.flair_decode_conv` — Conv3d(512, 128, k=1)

### 1.2 需要保持不变的模块

| 模块 | 文件位置 | 说明 |
|------|---------|------|
| Conv Encoder × 4 | `IMFuse.py: Encoder` | 5层3D CNN，每个模态独立 |
| MaskModal | `IMFuse.py: MaskModal` | Bernoulli 掩码 |
| Tokenize / TokenizeSep | `IMFuse.py` | Skip connection 的 token 化 |
| MambaFusionLayer / CatLayer | `IMFuse.py` | I-MFB 跨模态融合（Mamba） |
| Multimodal Transformer | `IMFuse.py` | InterFormer 全局注意力 |
| Decoder_fuse / Decoder_sep | `IMFuse.py` | 解码器 + RFM |
| fusion_prenorm | `layers.py` | 多模态融合层 |
| 全部 Loss 函数 | `utils/criterions.py` | Dice + Weighted CE |
| 数据管线 | `data/` 全部文件 | 不修改 |

---

## 2. 新模块设计：MambaVision-lite Hybrid Token Encoder

### 2.1 设计原则

从 MambaVision 的 `Block` 类（`mamba_vision.py:494-531`）中提取核心设计，适配 IM-Fuse 的 3D 医学分割场景：

1. **保持输入输出接口不变**：输入 `(B, 512, 512)` token 序列 + 位置编码，输出 `(B, 512, 512)`
2. **Hybrid Pattern**：MV-Mixer × k + Transformer × 1（Last Attention）
3. **3D 适配**：无需 window partition（序列长度仅 512，远小于 2D 视觉任务）

### 2.2 MambaVisionMixer3D 设计

从 `MambaVisionMixer`（`mamba_vision.py:310-430`）适配 3D 场景。两个关键区别：

| 维度 | 原始 MambaVision | MambaVision-lite 3D |
|------|------------------|---------------------|
| 输入形状 | (B, H×W, C) 2D patches | (B, 8³, 512) = (B, 512, 512) 3D patches |
| SSM d_state | 8 | 16（3D空间关系更复杂） |
| SSM d_conv | 3 | 3（保持不变） |
| expand | 1 | 1（控制显存） |
| 序列长度 | 196~784 (window) | 512 (8³) |

**MambaVision Mixer 原始架构**（提取自 `mamba_vision.py`）：
```python
# 双分支设计（SSM + Gate）
xz = Linear(d_model → d_inner)(x)              # 投影扩展
x, z = xz.chunk(2, dim=1)                       # 分两路
x = SiLU(Conv1d_x(x, k=3, groups, padding=same))  # SSM 分支: 深度卷积
z = SiLU(Conv1d_z(z, k=3, groups, padding=same))  # Gate 分支: 深度卷积

# SSM 分支执行 selective scan
x_dbl = x_proj(x)  → [Δ, B, C]
y = selective_scan_fn(x, Δ, A, B, C, D)

# 合并双分支
output = Linear(d_inner → d_model)(cat([y, z], dim=1))
```

### 2.3 三种实验配置对应的 Hybrid Encoder 结构

#### E1: Mamba-only（MV-Mixer × 2）
```
Input (B, 512, 512)
  ├─ Block 0: LayerNorm → MambaVisionMixer3D → DropPath → (+Residual) → LayerNorm → MLP → DropPath → (+Residual)
  ├─ Block 1: LayerNorm → MambaVisionMixer3D → DropPath → (+Residual) → LayerNorm → MLP → DropPath → (+Residual)
  └─ Output (B, 512, 512)
```

#### E2: Hybrid-1（MV-Mixer × 1 + Transformer × 1）⭐ 主方案
```
Input (B, 512, 512)
  ├─ Block 0: LayerNorm → MambaVisionMixer3D → DropPath → (+Residual) → LayerNorm → MLP → DropPath → (+Residual)
  ├─ Block 1: LayerNorm → SelfAttention(8 heads) → DropPath → (+Residual) → LayerNorm → MLP → DropPath → (+Residual)
  └─ Output (B, 512, 512)
```

#### E3: Hybrid-2（MV-Mixer × 2 + Transformer × 1）
```
Input (B, 512, 512)
  ├─ Block 0: LayerNorm → MambaVisionMixer3D → DropPath → (+Residual) → LayerNorm → MLP → DropPath → (+Residual)
  ├─ Block 1: LayerNorm → MambaVisionMixer3D → DropPath → (+Residual) → LayerNorm → MLP → DropPath → (+Residual)
  ├─ Block 2: LayerNorm → SelfAttention(8 heads) → DropPath → (+Residual) → LayerNorm → MLP → DropPath → (+Residual)
  └─ Output (B, 512, 512)
```

### 2.4 与原始 IM-Fuse Transformer 的对比

| 维度 | 原始 Intra-modal Transformer | Hybrid-1 (E2) |
|------|-------------------------------|----------------|
| 总层数 | 1 (Transformer only) | 2 (Mamba + Transformer) |
| Token Mixing | Self-Attention 全连接 | SSM 顺序扫描 + Self-Attention |
| FFN | 512→4096→512, GELU, Dropout | 512→2048→512, GELU（MLP ratio=4） |
| 位置编码 | 加到输入（每层加一次） | 仅加到输入（第一层之前） |
| 残差连接 | fn(x) + x | x + γ·DropPath(mixer(LN(x))) |
| Layer Scale | 无 | 可选（推荐 1e-5 初始化） |
| Drop Path | 无 | 0.1 |
| 参数量增量 | — | +~1.6M per modality |

---

## 3. 文件变更清单

### 3.1 新建文件

| 文件路径 | 用途 |
|---------|------|
| `mambavision_mixer.py` | MambaVisionMixer3D + HybridBlock 独立模块 |
| `IMFuse_hybrid.py` | 替换 IntraFormer 后的完整模型 |
| `train_hybrid.py` | 三阶段迁移训练脚本 |
| `scripts/sanity_check.sh` | Sanity check 启动脚本 |
| `scripts/train_e2_hybrid1.sh` | E2 Hybrid-1 训练启动脚本 |
| `scripts/eval_all_missing.sh` | 15种缺失组合全面评估脚本 |

### 3.2 不修改的文件

```
layers.py              # 不动
utils/criterions.py    # 不动
utils/lr_scheduler.py  # 不动（复用 LR_Scheduler + poly 模式）
utils/parser.py        # 不动
utils/initialization.py # 不动
data/*                  # 全部不动
predict.py             # 不动
test.py                # 不动
IMFuse.py              # 不动（保持基线可用）
IMFuse_no1skip.py      # 不动
train_poly.py          # 不动
```

---

## 4. Step 1：实现 MambaVision-lite 模块

### 4.1 文件：`mambavision_mixer.py`

```python
"""
MambaVision-lite 3D Hybrid Token Encoder
从 MambaVision 论文提取核心 SSM mixer，适配 IM-Fuse 3D 医学分割。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from torch.cuda.amp import autocast


class MambaVisionMixer3D(nn.Module):
    """
    从 MambaVision 的 MambaVisionMixer 适配而来。
    双分支设计：SSM 分支 + Gate 分支，最后 concat 投影。

    与原始 mamba_ssm.Mamba 的关键区别：
    1. 使用 regular conv（非 causal conv），padding='same'
    2. 增加对称的 non-SSM 分支（z 分支独立 conv + SiLU）
    3. 两路 concat 后统一投影

    参数:
        d_model: 输入/输出特征维度 (默认 512)
        d_state: SSM 状态维度 (默认 16)
        d_conv: 深度卷积核大小 (默认 3)
        expand: 扩展因子 (默认 1, 控制 d_inner = expand * d_model)
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=1,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入投影：d_model → d_inner (split into x, z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)

        # SSM 参数投影
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)

        # Δ 初始化
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner // 2) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # A 矩阵
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n", d=self.d_inner // 2
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D 前馈
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))
        self.D._no_weight_decay = True

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # 双分支深度卷积 (regular conv, padding='same')
        self.conv1d_x = nn.Conv1d(
            self.d_inner // 2, self.d_inner // 2,
            kernel_size=d_conv, groups=self.d_inner // 2,
            bias=conv_bias
        )
        self.conv1d_z = nn.Conv1d(
            self.d_inner // 2, self.d_inner // 2,
            kernel_size=d_conv, groups=self.d_inner // 2,
            bias=conv_bias
        )

    @autocast(enabled=False)
    def forward(self, hidden_states):
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.float()

        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())

        # SSM 分支
        x = F.silu(F.conv1d(x, self.conv1d_x.weight, self.conv1d_x.bias, padding='same', groups=self.d_inner // 2))
        # Gate 分支
        z = F.silu(F.conv1d(z, self.conv1d_z.weight, self.conv1d_z.bias, padding='same', groups=self.d_inner // 2))

        # SSM 计算
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x, dt, A, B, C, self.D.float(),
            z=None, delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True, return_last_state=None
        )

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class HybridBlock(nn.Module):
    """
    统一的 Hybrid Block，可选 MambaVisionMixer3D 或 SelfAttention 作为 token mixer。
    结构: LN → Mixer → DropPath(γ₁·) → (+Residual) → LN → MLP → DropPath(γ₂·) → (+Residual)

    这正是 MambaVision 论文中 Block 的设计（mamba_vision.py:494-531），
    但不使用 window partition（3D 序列长度仅 512）。
    """

    def __init__(self, dim, use_attention=False, num_heads=8,
                 mlp_ratio=4.0, drop_path=0.1, layer_scale=1e-5,
                 # MambaVisionMixer3D 参数
                 d_state=16, d_conv=3, expand=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        if use_attention:
            self.mixer = SelfAttention3D(dim, heads=num_heads)
        else:
            self.mixer = MambaVisionMixer3D(
                d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

        # Layer Scale
        if layer_scale is not None:
            self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))
        else:
            self.gamma_1 = 1.0
            self.gamma_2 = 1.0

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class SelfAttention3D(nn.Module):
    """与 IM-Fuse 原始 SelfAttention 保持一致的接口。"""

    def __init__(self, dim, heads=8, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HybridTokenEncoder(nn.Module):
    """
    MambaVision-lite Hybrid Token Encoder
    替换 IM-Fuse 中单个模态的 Intra-modal Transformer。

    接口不变：
        输入: tokens (B, 512, 512) + pos (1, 512, 512)
        输出: tokens (B, 512, 512)

    参数:
        dim: token 维度 (512)
        num_mamba_blocks: MV-Mixer 层数
        num_attn_blocks: Transformer 层数（放在最后，Late Attention）
        num_heads: 注意力头数
        mlp_ratio: MLP 隐藏层比例
        drop_path: Drop path rate
        d_state: SSM 状态维度
        d_conv: SSM 卷积核大小
        expand: SSM 扩展因子
    """

    def __init__(self, dim=512, num_mamba_blocks=1, num_attn_blocks=1,
                 num_heads=8, mlp_ratio=4.0, drop_path=0.1,
                 layer_scale=1e-5, d_state=16, d_conv=3, expand=1):
        super().__init__()
        total_depth = num_mamba_blocks + num_attn_blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path, total_depth)]

        blocks = []
        # 前 num_mamba_blocks 层: MambaVisionMixer3D
        for i in range(num_mamba_blocks):
            blocks.append(HybridBlock(
                dim=dim, use_attention=False, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop_path=dpr[i],
                layer_scale=layer_scale, d_state=d_state,
                d_conv=d_conv, expand=expand
            ))
        # 后 num_attn_blocks 层: SelfAttention (Late Attention)
        for i in range(num_attn_blocks):
            blocks.append(HybridBlock(
                dim=dim, use_attention=True, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop_path=dpr[num_mamba_blocks + i],
                layer_scale=layer_scale
            ))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, pos):
        """
        Args:
            x: (B, num_tokens, dim) — token 序列
            pos: (1, num_tokens, dim) — 可学习位置编码
        Returns:
            (B, num_tokens, dim)
        """
        x = x + pos  # 位置编码只加一次（与原始 Transformer 每层加不同，简化设计）
        for blk in self.blocks:
            x = blk(x)
        return x
```

### 4.2 模块参数量估算

以 E2 Hybrid-1（MV-Mixer × 1 + Transformer × 1, dim=512）为例：

| 子模块 | 参数量 |
|--------|--------|
| **MambaVisionMixer3D** | |
| — in_proj (512→512) | 262,144 |
| — x_proj (256→32+16+16=64) | 16,384 |
| — dt_proj (32→256) | 8,448 |
| — A_log (256×16) | 4,096 |
| — D (256) | 256 |
| — out_proj (512→512) | 262,144 |
| — conv1d_x (256, k=3) | 1,024 |
| — conv1d_z (256, k=3) | 1,024 |
| **小计 MV-Mixer** | **~555K** |
| **HybridBlock[0] MLP** (512→2048→512) | 2,098,176 |
| **HybridBlock[0] LN×2 + γ×2** | 2,048 |
| **SelfAttention3D** (512, 8 heads) | |
| — qkv (512→1536) | 786,432 |
| — proj (512→512) | 262,144 |
| **小计 Attention** | **~1,048K** |
| **HybridBlock[1] MLP** | 2,098,176 |
| **HybridBlock[1] LN×2 + γ×2** | 2,048 |
| **总计 per modality** | **~5.8M** |
| **4 个模态** | **~23.2M** |

**对比原始 Intra-modal Transformer**（每个模态约 4.7M）：

| 配置 | 每模态参数量 | 4 模态总增量 |
|------|------------|-------------|
| 原始 Transformer × 1 | ~4.7M | 基线 |
| E2: Hybrid-1 (Mamba×1 + Attn×1) | ~5.8M | +4.4M (+23%) |
| E1: Mamba-only (Mamba×2) | ~5.3M | +2.4M (+13%) |
| E3: Hybrid-2 (Mamba×2 + Attn×1) | ~8.4M | +14.8M (+79%) |

---

## 5. Step 2：创建 IMFuse_hybrid 模型文件

### 5.1 文件：`IMFuse_hybrid.py`

基于 `IMFuse.py` 创建，仅修改 IntraFormer 部分。变更对比：

```diff
 # 新增 import
+from mambavision_mixer import HybridTokenEncoder

 class IMFuseHybrid(nn.Module):
-    def __init__(self, num_cls=4, interleaved_tokenization=False, mamba_skip=False):
+    def __init__(self, num_cls=4, interleaved_tokenization=False, mamba_skip=False,
+                 num_mamba_blocks=1, num_attn_blocks=1, drop_path=0.1):
         ...
         ########### IntraFormer
         # encode_conv / decode_conv / pos 保持不变

-        self.flair_transformer = Transformer(embedding_dim=transformer_basic_dims,
-                                              depth=depth, heads=num_heads, mlp_dim=mlp_dim)
-        self.t1ce_transformer = Transformer(...)
-        self.t1_transformer = Transformer(...)
-        self.t2_transformer = Transformer(...)
+        self.flair_hybrid_encoder = HybridTokenEncoder(
+            dim=transformer_basic_dims,
+            num_mamba_blocks=num_mamba_blocks,
+            num_attn_blocks=num_attn_blocks,
+            num_heads=num_heads,
+            mlp_ratio=4.0,
+            drop_path=drop_path,
+        )
+        self.t1ce_hybrid_encoder = HybridTokenEncoder(...)
+        self.t1_hybrid_encoder = HybridTokenEncoder(...)
+        self.t2_hybrid_encoder = HybridTokenEncoder(...)
```

**forward 方法中的变更**：

```diff
         ########### IntraFormer
-        flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
-        t1ce_intra_token_x5 = self.t1ce_transformer(t1ce_token_x5, self.t1ce_pos)
-        t1_intra_token_x5 = self.t1_transformer(t1_token_x5, self.t1_pos)
-        t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)
+        flair_intra_token_x5 = self.flair_hybrid_encoder(flair_token_x5, self.flair_pos)
+        t1ce_intra_token_x5 = self.t1ce_hybrid_encoder(t1ce_token_x5, self.t1ce_pos)
+        t1_intra_token_x5 = self.t1_hybrid_encoder(t1_token_x5, self.t1_pos)
+        t2_intra_token_x5 = self.t2_hybrid_encoder(t2_token_x5, self.t2_pos)
```

**其余所有代码保持与 `IMFuse.py` 完全一致。**

### 5.2 模型构造参数与实验版本映射

| 参数 | E1 (Mamba-only) | E2 (Hybrid-1) ⭐ | E3 (Hybrid-2) |
|------|----------------|-----------------|----------------|
| `num_mamba_blocks` | 2 | 1 | 2 |
| `num_attn_blocks` | 0 | 1 | 1 |
| `drop_path` | 0.1 | 0.1 | 0.1 |

---

## 6. Step 3：实现三阶段迁移训练脚本

### 6.1 文件：`train_hybrid.py`

基于 `train_poly.py` 扩展，核心新增功能：

1. **三阶段训练循环**
2. **分组学习率 + 参数冻结**
3. **阶段间 Checkpoint 管理**
4. **从 IM-Fuse 检查点加载预训练权重**

### 6.2 新增命令行参数

```python
# 模型配置
parser.add_argument('--num_mamba_blocks', default=1, type=int, help='MV-Mixer 层数')
parser.add_argument('--num_attn_blocks', default=1, type=int, help='Attention 层数')
parser.add_argument('--drop_path', default=0.1, type=float, help='Drop path rate')

# 三阶段训练
parser.add_argument('--stage', default=1, type=int, choices=[1, 2, 3], help='当前训练阶段')
parser.add_argument('--stage1_epochs', default=50, type=int, help='Stage 1 训练轮数')
parser.add_argument('--stage2_epochs', default=100, type=int, help='Stage 2 训练轮数')
parser.add_argument('--stage3_epochs', default=300, type=int, help='Stage 3 训练轮数')

# 预训练权重
parser.add_argument('--pretrained_imfuse', default=None, type=str, help='IM-Fuse 预训练检查点路径')
parser.add_argument('--stage_resume', default=None, type=str, help='当前阶段继续训练的检查点')
```

### 6.3 参数分组与冻结逻辑

```python
def get_param_groups(model, stage, args):
    """
    根据训练阶段返回不同的参数组，实现分组学习率 + 冻结。

    参数命名规则（DataParallel 后带 module. 前缀）：
    - module.{mod}_hybrid_encoder.blocks.0.mixer.* → MV-Mixer (block 0)
    - module.{mod}_hybrid_encoder.blocks.1.mixer.* → Attention (block 1, for Hybrid-1)
    - module.{mod}_hybrid_encoder.blocks.*.mlp.*   → MLP
    - module.{mod}_hybrid_encoder.blocks.*.norm*    → LayerNorm
    - module.{mod}_hybrid_encoder.blocks.*.gamma*   → Layer Scale
    - module.{mod}_encoder.*                        → Conv Encoder
    - module.mamba_fusion_layer*                     → I-MFB
    - module.multimodal_transformer.*               → InterFormer
    - module.decoder_fuse.* / module.decoder_sep.*  → Decoder
    """
    # 收集各组参数
    mv_mixer_params = []       # MV-Mixer blocks (新增)
    hybrid_attn_params = []    # Hybrid Encoder 中的 Attention blocks
    conv_encoder_params = []   # 4 个 Conv Encoder
    fusion_params = []         # I-MFB + Multimodal Transformer
    decoder_params = []        # Decoder_fuse + Decoder_sep
    other_params = []          # encode_conv, decode_conv, pos 等

    modalities = ['flair', 't1ce', 't1', 't2']

    for name, param in model.named_parameters():
        matched = False

        # MV-Mixer block 参数
        for mod in modalities:
            if f'{mod}_hybrid_encoder' in name:
                # 判断是 Mamba block 还是 Attention block
                # blocks.0 = Mamba (for Hybrid-1), blocks.1 = Attention
                if '.mixer.' in name:
                    # 解析 block index
                    block_idx = int(name.split('.blocks.')[1].split('.')[0])
                    if block_idx < args.num_mamba_blocks:
                        mv_mixer_params.append(param)
                    else:
                        hybrid_attn_params.append(param)
                else:
                    # norm, mlp, gamma 归属于所在 block
                    block_idx = int(name.split('.blocks.')[1].split('.')[0])
                    if block_idx < args.num_mamba_blocks:
                        mv_mixer_params.append(param)
                    else:
                        hybrid_attn_params.append(param)
                matched = True
                break

        if matched:
            continue

        # Conv Encoder
        if '_encoder.' in name and 'hybrid' not in name:
            conv_encoder_params.append(param)
        # I-MFB + Multimodal Transformer
        elif 'mamba_fusion' in name or 'multimodal_transformer' in name:
            fusion_params.append(param)
        # Decoder
        elif 'decoder_fuse' in name or 'decoder_sep' in name:
            decoder_params.append(param)
        # 其他（encode_conv, decode_conv, pos, tokenize 等）
        else:
            other_params.append(param)

    base_lr = args.lr  # 2e-4

    if stage == 1:
        # 仅训练 MV-Mixer，其余全冻结
        for p in hybrid_attn_params + conv_encoder_params + fusion_params + decoder_params + other_params:
            p.requires_grad = False
        return [
            {'params': mv_mixer_params, 'lr': base_lr, 'name': 'mv_mixer'},
        ]

    elif stage == 2:
        # 训练 MV-Mixer + Hybrid Encoder 的 Attention
        for p in conv_encoder_params + fusion_params + decoder_params + other_params:
            p.requires_grad = False
        for p in mv_mixer_params + hybrid_attn_params:
            p.requires_grad = True
        return [
            {'params': mv_mixer_params, 'lr': base_lr * 0.5, 'name': 'mv_mixer'},         # 1e-4
            {'params': hybrid_attn_params, 'lr': base_lr * 0.25, 'name': 'hybrid_attn'},   # 5e-5
        ]

    elif stage == 3:
        # 全部解冻
        for p in model.parameters():
            p.requires_grad = True
        return [
            {'params': mv_mixer_params, 'lr': base_lr * 0.25, 'name': 'mv_mixer'},         # 5e-5
            {'params': hybrid_attn_params, 'lr': base_lr * 0.1, 'name': 'hybrid_attn'},     # 2e-5
            {'params': fusion_params, 'lr': base_lr * 0.05, 'name': 'fusion'},              # 1e-5
            {'params': conv_encoder_params, 'lr': base_lr * 0.025, 'name': 'conv_encoder'}, # 5e-6
            {'params': decoder_params, 'lr': base_lr * 0.05, 'name': 'decoder'},            # 1e-5
            {'params': other_params, 'lr': base_lr * 0.05, 'name': 'other'},                # 1e-5
        ]
```

### 6.4 训练主循环伪代码

```python
def main():
    # 1. 构建模型
    model = IMFuseHybrid(
        num_cls=num_cls,
        interleaved_tokenization=args.interleaved_tokenization,
        mamba_skip=args.mamba_skip,
        num_mamba_blocks=args.num_mamba_blocks,
        num_attn_blocks=args.num_attn_blocks,
        drop_path=args.drop_path,
    )

    # 2. 加载 IM-Fuse 预训练权重（仅 Stage 1 首次启动时）
    if args.pretrained_imfuse and args.stage == 1 and args.stage_resume is None:
        load_imfuse_pretrained(model, args.pretrained_imfuse)

    model = torch.nn.DataParallel(model).cuda()

    # 3. 配置 optimizer（每阶段重建，清除 momentum）
    param_groups = get_param_groups(model, args.stage, args)
    optimizer = torch.optim.RAdam(param_groups, weight_decay=args.weight_decay)

    # 4. 配置 LR scheduler（每阶段独立 Poly）
    stage_epochs = getattr(args, f'stage{args.stage}_epochs')
    lr_schedule = LR_Scheduler(args.lr, stage_epochs, mode='poly')

    # 5. 如果是阶段内继续训练
    if args.stage_resume:
        checkpoint = torch.load(args.stage_resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    # 6. 训练循环（与 train_poly.py 基本一致）
    for epoch in range(start_epoch, stage_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        ...
        # 训练 epoch 代码复用 train_poly.py 逻辑
        ...

    # 7. 保存阶段最终 checkpoint
    save_stage_checkpoint(model, optimizer, args.stage, stage_epochs)
    print(f"Stage {args.stage} 完成，保存至 {ckpts}/stage{args.stage}_final.pth")
```

### 6.5 LR Scheduler 适配

每个阶段独立使用 Poly scheduler，但需要适配分组学习率：

```python
class GroupLR_Scheduler:
    """分组 Poly LR Scheduler"""

    def __init__(self, param_groups_config, num_epochs, mode='poly'):
        self.mode = mode
        self.num_epochs = num_epochs
        self.base_lrs = [pg['lr'] for pg in param_groups_config]

    def __call__(self, optimizer, epoch):
        for i, base_lr in enumerate(self.base_lrs):
            if self.mode == 'poly':
                now_lr = round(base_lr * (1 - epoch / self.num_epochs) ** 0.9, 8)
            optimizer.param_groups[i]['lr'] = now_lr
        return optimizer.param_groups[0]['lr']
```

---

## 7. Step 4：检查点加载与权重迁移

### 7.1 从 IM-Fuse 检查点加载到 IMFuseHybrid

IM-Fuse 和 IMFuseHybrid 的权重名称映射规则：

| IM-Fuse 权重名 | IMFuseHybrid 权重名 | 处理方式 |
|----------------|---------------------|---------|
| `module.flair_encoder.*` | `module.flair_encoder.*` | 直接复制 |
| `module.flair_encode_conv.*` | `module.flair_encode_conv.*` | 直接复制 |
| `module.flair_decode_conv.*` | `module.flair_decode_conv.*` | 直接复制 |
| `module.flair_pos` | `module.flair_pos` | 直接复制 |
| `module.flair_transformer.*` | ❌ 不存在 | 跳过（或映射到 hybrid_encoder 的 Attention block） |
| ❌ 不存在 | `module.flair_hybrid_encoder.blocks.0.*` | 随机初始化（MV-Mixer） |
| ❌ 不存在 | `module.flair_hybrid_encoder.blocks.1.*` | 可选：从 flair_transformer 迁移 |
| `module.mamba_fusion_layer*` | `module.mamba_fusion_layer*` | 直接复制 |
| `module.multimodal_transformer.*` | `module.multimodal_transformer.*` | 直接复制 |
| `module.decoder_fuse.*` | `module.decoder_fuse.*` | 直接复制 |
| `module.decoder_sep.*` | `module.decoder_sep.*` | 直接复制 |

### 7.2 权重迁移函数

```python
def load_imfuse_pretrained(model, checkpoint_path):
    """
    从 IM-Fuse 检查点加载权重到 IMFuseHybrid。
    新增的 MV-Mixer 参数保持随机初始化。
    可选：将原始 Transformer 权重映射到 Hybrid Encoder 的最后一个 Attention block。
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrained_dict = checkpoint['state_dict']

    model_dict = model.state_dict()
    modalities = ['flair', 't1ce', 't1', 't2']

    # 1. 直接复制所有名称匹配的权重
    transferred = {}
    skipped = []
    for k, v in pretrained_dict.items():
        # 去掉 'module.' 前缀（如果有）
        clean_k = k.replace('module.', '')

        if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
            transferred[clean_k] = v
        else:
            skipped.append(k)

    # 2. 将原始 Transformer 权重映射到 Hybrid Encoder 的 Attention block
    for mod in modalities:
        attn_block_idx = model.num_mamba_blocks  # Attention block 的 index

        # 原始 Transformer 映射规则：
        # {mod}_transformer.cross_attention_list.0.fn.fn.{qkv,proj}.weight
        #   → {mod}_hybrid_encoder.blocks.{attn_block_idx}.mixer.{qkv,proj}.weight
        # {mod}_transformer.cross_ffn_list.0.fn.fn.net.{0,3}.{weight,bias}
        #   → {mod}_hybrid_encoder.blocks.{attn_block_idx}.mlp.{0,2}.{weight,bias}

        prefix_old = f'{mod}_transformer'
        prefix_new = f'{mod}_hybrid_encoder.blocks.{attn_block_idx}'

        # SelfAttention 权重
        attn_mapping = {
            f'{prefix_old}.cross_attention_list.0.fn.norm.weight': f'{prefix_new}.norm1.weight',
            f'{prefix_old}.cross_attention_list.0.fn.norm.bias': f'{prefix_new}.norm1.bias',
            f'{prefix_old}.cross_attention_list.0.fn.fn.qkv.weight': f'{prefix_new}.mixer.qkv.weight',
            f'{prefix_old}.cross_attention_list.0.fn.fn.proj.weight': f'{prefix_new}.mixer.proj.weight',
            f'{prefix_old}.cross_attention_list.0.fn.fn.proj.bias': f'{prefix_new}.mixer.proj.bias',
        }

        # FFN 权重
        ffn_mapping = {
            f'{prefix_old}.cross_ffn_list.0.fn.norm.weight': f'{prefix_new}.norm2.weight',
            f'{prefix_old}.cross_ffn_list.0.fn.norm.bias': f'{prefix_new}.norm2.bias',
            f'{prefix_old}.cross_ffn_list.0.fn.fn.net.0.weight': f'{prefix_new}.mlp.0.weight',
            f'{prefix_old}.cross_ffn_list.0.fn.fn.net.0.bias': f'{prefix_new}.mlp.0.bias',
            f'{prefix_old}.cross_ffn_list.0.fn.fn.net.3.weight': f'{prefix_new}.mlp.2.weight',
            f'{prefix_old}.cross_ffn_list.0.fn.fn.net.3.bias': f'{prefix_new}.mlp.2.bias',
        }

        for old_key, new_key in {**attn_mapping, **ffn_mapping}.items():
            if old_key in pretrained_dict and new_key in model_dict:
                old_val = pretrained_dict[old_key]
                if old_val.shape == model_dict[new_key].shape:
                    transferred[new_key] = old_val
                else:
                    print(f"形状不匹配，跳过: {old_key} {old_val.shape} vs {new_key} {model_dict[new_key].shape}")

    # 3. 加载
    model_dict.update(transferred)
    model.load_state_dict(model_dict, strict=False)

    print(f"权重迁移完成: {len(transferred)}/{len(model_dict)} 参数已加载")
    print(f"跳过（原始模型独有）: {len(skipped)} 个参数")

    # 4. 报告未初始化的参数（应该只有 MV-Mixer 相关）
    uninit = [k for k in model_dict if k not in transferred]
    print(f"随机初始化（新增模块）: {len(uninit)} 个参数")
    for k in uninit:
        print(f"  - {k}")
```

### 7.3 权重映射注意事项

IM-Fuse 原始 Transformer 的 FFN 结构：
```python
# 原始: net.0 = Linear(512→4096), net.2 = Dropout, net.3 = Linear(4096→512), net.4 = Dropout
FeedForward.net = Sequential(Linear, GELU, Dropout, Linear, Dropout)
# 对应 index: 0       1     2        3       4
```

HybridBlock 的 MLP 结构：
```python
# 新: 0 = Linear(512→2048), 1 = GELU, 2 = Linear(2048→512)
self.mlp = Sequential(Linear, GELU, Linear)
# 对应 index: 0       1      2
```

**注意**：原始 FFN hidden_dim=4096，新 MLP hidden_dim=2048（mlp_ratio=4 且 expand=1 时 d_model 不变）。
如果 `mlp_ratio=8`（维持 4096），则 FFN 权重可以直接复制。否则 FFN 部分需要随机初始化。

**建议**：为 Hybrid Encoder 的 Attention block 设置 `mlp_ratio=8`，使 hidden_dim=4096 与原始一致，最大化权重复用。

---

## 8. Step 5：Sanity Check 流程

### 8.1 Phase 0: 前向/反向传播验证

```python
# sanity_check.py
"""
验证步骤:
1. 构建 IMFuseHybrid 模型（E2 配置）
2. 加载 IM-Fuse 预训练权重
3. 单张样本前向传播，检查输出形状
4. 反向传播，检查梯度流
5. 20 个样本 overfit 测试（~50 epochs），验证 loss 收敛
"""

def sanity_check():
    # 1. 构建模型
    model = IMFuseHybrid(
        num_cls=4,
        num_mamba_blocks=1,
        num_attn_blocks=1,
        drop_path=0.1,
    )

    # 2. 加载预训练
    load_imfuse_pretrained(model, 'path/to/imfuse_best.pth')
    model = model.cuda()

    # 3. 前向传播测试
    x = torch.randn(1, 4, 128, 128, 128).cuda()
    mask = torch.ones(1, 4).bool().cuda()

    model.is_training = True
    fuse_pred, sep_preds, prm_preds = model(x, mask)

    assert fuse_pred.shape == (1, 4, 128, 128, 128), f"fuse_pred shape error: {fuse_pred.shape}"
    assert len(sep_preds) == 4, f"sep_preds count error: {len(sep_preds)}"
    assert len(prm_preds) == 4, f"prm_preds count error: {len(prm_preds)}"
    print("✓ 前向传播形状正确")

    # 4. 反向传播测试
    target = torch.zeros(1, 4, 128, 128, 128).cuda()
    target[:, 0] = 1  # 全背景
    loss = criterions.dice_loss(fuse_pred, target, num_cls=4)
    loss.backward()

    # 检查 MV-Mixer 梯度
    for name, param in model.named_parameters():
        if 'hybrid_encoder.blocks.0.mixer' in name and param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    print("✓ 反向传播梯度正常")

    # 5. Overfit 测试
    print("开始 overfit 测试 (20 samples, 50 epochs)...")
    # ... 加载 20 个真实样本，训练 50 epoch
    # 验证 loss < 0.1 即通过
```

### 8.2 验证清单

| 检查项 | 通过条件 | 阻断? |
|--------|---------|-------|
| 模型构建无报错 | 无异常 | 是 |
| 前向传播输出形状 | (B,4,128,128,128) | 是 |
| 反向传播梯度存在 | 所有 requires_grad=True 的参数有梯度 | 是 |
| 梯度无 NaN/Inf | `torch.isnan().any() == False` | 是 |
| 20 sample overfit | 50 epoch 后 loss < 0.1 | 是（若不通过则排查） |
| 缺失模态前向传播 | 15 种 mask 组合均无报错 | 是 |
| 单步训练速度 | < 原始 1.5x | 否（记录即可） |
| 显存占用 | < 原始 1.2x | 否（记录即可） |

---

## 9. Step 6：评估与消融分析

### 9.1 评估指标

与 IM-Fuse 保持完全一致（`predict.py: softmax_output_dice_class4`）：

| 指标 | 定义 |
|------|------|
| WT Dice | Whole Tumor (label 1+2+3) |
| TC Dice | Tumor Core (label 1+3) |
| ET Dice | Enhancing Tumor (label 3) |
| ET++ Dice | ET with post-processing (< 500 voxels → 0) |

### 9.2 15 种缺失模态评估矩阵

评估脚本需对每种缺失组合独立计算 Dice：

```python
# 15 种 mask 组合（与 train_poly.py 中一致）
masks = [
    # 单模态可用 (4)
    [False, False, False, True],   # T2 only
    [False, True, False, False],   # T1c only
    [False, False, True, False],   # T1 only
    [True, False, False, False],   # FLAIR only
    # 双模态可用 (6)
    [False, True, False, True],    # T1c + T2
    [False, True, True, False],    # T1c + T1
    [True, False, True, False],    # FLAIR + T1
    [False, False, True, True],    # T1 + T2
    [True, False, False, True],    # FLAIR + T2
    [True, True, False, False],    # FLAIR + T1c
    # 三模态可用 (4)
    [True, True, True, False],     # FLAIR + T1c + T1
    [True, False, True, True],     # FLAIR + T1 + T2
    [True, True, False, True],     # FLAIR + T1c + T2
    [False, True, True, True],     # T1c + T1 + T2
    # 全模态 (1)
    [True, True, True, True],      # All
]
```

### 9.3 关键关注场景

根据实验计划，重点监控：

1. **T1c 缺失 → ET Dice**：mask index 0, 2, 3, 6, 7, 8, 11
2. **单模态可用场景**：index 0-3（最极端的缺失）
3. **全模态场景**：index 14（是否引入退化）

### 9.4 消融分析计划

| 消融实验 | 对比 | 验证假设 |
|---------|------|---------|
| E2 vs Baseline | Hybrid-1 vs 原始 Transformer | MV-Mixer 增强效果 |
| E1 vs E2 | Mamba-only vs Hybrid | Late Attention 是否必要 |
| E2 vs E3 | Hybrid-1 vs Hybrid-2 | 增加 Mamba 层数的边际收益 |
| Stage 1 checkpoint vs Baseline | MV-Mixer warm-up 后 | 新模块是否有正面信号 |
| Stage 2 checkpoint vs Stage 1 | 加入 Attention fine-tune | 两模块联合是否优于单模块 |

---

## 10. 显存与吞吐量估算

### 10.1 显存分析

以 batch_size=1, input=(1, 4, 128, 128, 128), float32 为基准：

**原始 IM-Fuse 的 IntraFormer 部分**（每个模态）：
- encode_conv 输出: 512×8×8×8 = 1MB
- Token: 512×512 = 1MB
- Transformer (depth=1):
  - QKV: 3×512×512 = 3MB
  - Attention: 512×512 = 1MB (可释放)
  - FFN: 512×4096 + 4096×512 = 8MB (可释放)
- 4 个模态: ~4 × ≈5MB = **~20MB 激活值**

**Hybrid-1 (E2) IntraFormer 部分**（每个模态）：
- encode_conv 输出: 同上 1MB
- Token: 同上 1MB
- MV-Mixer (Block 0):
  - in_proj: 512×512 = 1MB
  - conv1d_x/z: 0.5MB
  - selective_scan: 参与 SSM 状态展开， ~2MB
  - MLP: 512×2048 = 4MB
- Attention (Block 1):
  - QKV + Attention: ~4MB
  - MLP: 512×2048 = 4MB
- 4 个模态: ~4 × ≈12MB = **~48MB 激活值**

**增长**: ~28MB / ~20MB ≈ **+140%** 仅在 IntraFormer 部分。但 IntraFormer 仅占总模型显存的一小部分（<10%），因为 Conv Encoder + Decoder + Skip Mamba Fusion 占主体。

**总体估算**: 总显存增长 **~8-12%**，满足 ≤15% 的约束。

### 10.2 吞吐量分析

**序列长度 512 对于 Mamba SSM 是轻量级的**（Mamba 为线性复杂度）。主要增量来自多一个 Block 的前向/反向计算。

| 配置 | IntraFormer FLOPs (per modality) | 训练吞吐估计 |
|------|----------------------------------|-------------|
| Baseline | ~2.1 GFLOPs | 基线 |
| E2 Hybrid-1 | ~3.5 GFLOPs | ~83% 基线（-17%） |
| E1 Mamba-only | ~2.8 GFLOPs | ~87% 基线（-13%） |
| E3 Hybrid-2 | ~4.5 GFLOPs | ~75% 基线（-25%） |

E2 预计满足 ≤20% 下降的约束。E3 可能略超。

---

## 11. 完整实施时间线

### Phase 0: 代码实现 + Sanity Check

| 步骤 | 任务 | 产出 |
|------|------|------|
| 0.1 | 实现 `mambavision_mixer.py` | MambaVisionMixer3D, HybridBlock, HybridTokenEncoder |
| 0.2 | 实现 `IMFuse_hybrid.py` | IMFuseHybrid 模型类 |
| 0.3 | 实现 `train_hybrid.py` | 三阶段训练脚本 |
| 0.4 | 实现权重迁移函数 | `load_imfuse_pretrained()` |
| 0.5 | 前向/反向传播验证 | 形状、梯度、NaN 检查 |
| 0.6 | 20 sample overfit | Loss 收敛验证 |
| 0.7 | 15 种 mask 组合测试 | 无报错确认 |
| 0.8 | 记录显存 + 吞吐 | 确认满足约束 |

### Phase 1: Baseline 确认

| 步骤 | 任务 | 产出 |
|------|------|------|
| 1.1 | 加载 IM-Fuse 检查点 | 确认可正常推理 |
| 1.2 | 15 种 mask 全面评估 | 基线 Dice 数值 |
| 1.3 | 记录 T1c-missing ET Dice | 重点关注指标 |

### Phase 2: E2 Hybrid-1 三阶段训练 ⭐

| 步骤 | 任务 | Epochs | 产出 |
|------|------|--------|------|
| 2.1 | Stage 1: Warm-up MV-Mixer | 30-50 | `stage1_final.pth` |
| 2.2 | Stage 1 评估 | — | MV-Mixer 正面信号判断 |
| 2.3 | Stage 2: Fine-tune Token Encoder | 50-100 | `stage2_final.pth` |
| 2.4 | Stage 2 评估 | — | Hybrid 联合效果 |
| 2.5 | Stage 3: End-to-end Fine-tune | 200-300 | `stage3_final.pth` + `best.pth` |
| 2.6 | 全面评估 | — | 15 种 mask Dice 表格 |

### Phase 3: E1 + E3 平行实验

| 步骤 | 任务 | 产出 |
|------|------|------|
| 3.1 | E1 三阶段训练 | `e1_best.pth` |
| 3.2 | E3 三阶段训练 | `e3_best.pth` |
| 3.3 | E1/E3 全面评估 | 消融对比数据 |

### Phase 4: 分析与报告

| 步骤 | 任务 | 产出 |
|------|------|------|
| 4.1 | 跨阶段 checkpoint 对比 | 各阶段收益分析 |
| 4.2 | 关键场景深入分析 | T1c-missing ET 表现 |
| 4.3 | 效率对比 | 显存/吞吐/参数量表格 |
| 4.4 | 撰写实验报告 | 完整结果汇总 |

---

## 附录 A：启动脚本模板

### A.1 E2 Hybrid-1 Stage 1

```bash
#!/bin/bash
python train_hybrid.py \
    --datapath /path/to/BraTS2023 \
    --dataname BRATS2023 \
    --savepath output/e2_hybrid1 \
    --lr 2e-4 \
    --weight_decay 3e-5 \
    --batch_size 1 \
    --seed 999 \
    --num_mamba_blocks 1 \
    --num_attn_blocks 1 \
    --drop_path 0.1 \
    --stage 1 \
    --stage1_epochs 50 \
    --pretrained_imfuse /path/to/imfuse_best.pth \
    --mamba_skip \
    --first_skip
```

### A.2 E2 Hybrid-1 Stage 2（接 Stage 1）

```bash
python train_hybrid.py \
    --datapath /path/to/BraTS2023 \
    --dataname BRATS2023 \
    --savepath output/e2_hybrid1 \
    --lr 2e-4 \
    --weight_decay 3e-5 \
    --batch_size 1 \
    --seed 999 \
    --num_mamba_blocks 1 \
    --num_attn_blocks 1 \
    --drop_path 0.1 \
    --stage 2 \
    --stage2_epochs 100 \
    --stage_resume output/e2_hybrid1/stage1_final.pth \
    --mamba_skip \
    --first_skip
```

### A.3 E2 Hybrid-1 Stage 3（接 Stage 2）

```bash
python train_hybrid.py \
    --datapath /path/to/BraTS2023 \
    --dataname BRATS2023 \
    --savepath output/e2_hybrid1 \
    --lr 2e-4 \
    --weight_decay 3e-5 \
    --batch_size 1 \
    --seed 999 \
    --num_mamba_blocks 1 \
    --num_attn_blocks 1 \
    --drop_path 0.1 \
    --stage 3 \
    --stage3_epochs 300 \
    --stage_resume output/e2_hybrid1/stage2_final.pth \
    --mamba_skip \
    --first_skip
```

---

## 附录 B：依赖项检查

`train_hybrid.py` 需要的额外依赖（对比现有 `requirements.txt`）：

| 包名 | 用途 | 说明 |
|------|------|------|
| `mamba-ssm` | selective_scan_fn | 已安装（IM-Fuse 已使用） |
| `einops` | rearrange, repeat | 需要确认安装 |
| `timm` | DropPath | 需要确认安装 |

```bash
# 验证依赖
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba_ssm OK')"
python -c "from einops import rearrange; print('einops OK')"
python -c "from timm.models.layers import DropPath; print('timm OK')"
```

---

## 附录 C：关键代码引用索引

| 概念 | 源文件 | 行号范围 |
|------|--------|---------|
| MambaVisionMixer 原始实现 | `MambaVision/mambavision/models/mamba_vision.py` | 310-430 |
| Block (Hybrid switching) | `MambaVision/mambavision/models/mamba_vision.py` | 494-531 |
| MambaVisionLayer (window partition) | `MambaVision/mambavision/models/mamba_vision.py` | 535-620 |
| IM-Fuse IntraFormer 定义 | `MV-IM-Fuse/IMFuse.py` | 420-470 |
| IM-Fuse IntraFormer forward | `MV-IM-Fuse/IMFuse.py` | 480-510 |
| IM-Fuse InterFormer (I-MFB) | `MV-IM-Fuse/IMFuse.py` | 530-560 |
| MaskModal | `MV-IM-Fuse/IMFuse.py` | 400-410 |
| Transformer (original) | `MV-IM-Fuse/IMFuse.py` | 345-385 |
| train_poly.py 训练循环 | `MV-IM-Fuse/train_poly.py` | 195-280 |
| 分组 LR Scheduler | `MV-IM-Fuse/utils/lr_scheduler.py` | 1-20 |
| Loss 函数 | `MV-IM-Fuse/utils/criterions.py` | 全文 |
| BraTS 数据集 | `MV-IM-Fuse/data/datasets_nii.py` | 全文 |
