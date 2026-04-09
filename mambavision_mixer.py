"""
MambaVision-lite 3D Hybrid Token Encoder
从 MambaVision 论文提取核心 SSM mixer，适配 IM-Fuse 3D 医学分割场景。

核心设计：
- MambaVisionMixer3D: 双分支 (SSM + Gate) mixer，regular conv + concat 投影
- HybridBlock: LN → Mixer → DropPath → Residual → LN → MLP → DropPath → Residual
- SelfAttention3D: 标准多头自注意力
- HybridTokenEncoder: MV-Mixer × k + Transformer × m (Late Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from torch.cuda.amp import autocast
from utils.perf import sdpa_attention


class MambaVisionMixer3D(nn.Module):
    """
    从 MambaVision 的 MambaVisionMixer 适配而来。
    双分支设计：SSM 分支 + Gate 分支，最后 concat 投影。

    与原始 mamba_ssm.Mamba 的关键区别：
    1. 使用 regular conv (非 causal conv)，padding='same'
    2. 增加对称的 non-SSM 分支 (z 分支独立 conv + SiLU)
    3. 两路 concat 后统一投影

    参数:
        d_model: 输入/输出特征维度 (默认 512)
        d_state: SSM 状态维度 (默认 16)
        d_conv: 深度卷积核大小 (默认 3)
        expand: 扩展因子 (默认 1, 控制 d_inner = expand * d_model)
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入投影：d_model → d_inner
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)

        # SSM 参数投影
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)

        # Δ 初始化 (与 MambaVision 一致)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner // 2) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # A 矩阵
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner // 2,
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
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
        )

    @autocast(enabled=False)
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, L, D) token 序列
        Returns:
            (B, L, D) 相同形状
        """
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.float()

        _, seqlen, _ = hidden_states.shape

        # 投影 + 分支
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())

        # SSM 分支: regular conv + SiLU
        x = F.silu(
            F.conv1d(
                input=x,
                weight=self.conv1d_x.weight,
                bias=self.conv1d_x.bias,
                padding='same',
                groups=self.d_inner // 2,
            )
        )
        # Gate 分支: regular conv + SiLU
        z = F.silu(
            F.conv1d(
                input=z,
                weight=self.conv1d_z.weight,
                bias=self.conv1d_z.bias,
                padding='same',
                groups=self.d_inner // 2,
            )
        )

        # SSM 计算
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )

        # 合并双分支
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class SelfAttention3D(nn.Module):
    """标准多头自注意力，与 IM-Fuse 原始 SelfAttention 保持一致接口。"""

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
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = sdpa_attention(
            q,
            k,
            v,
            attn_drop_p=self.attn_drop.p,
            training=self.training,
            scale=self.scale,
        ).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HybridBlock(nn.Module):
    """
    统一的 Hybrid Block，可选 MambaVisionMixer3D 或 SelfAttention 作为 token mixer。
    结构: LN → Mixer → DropPath(γ₁·) → (+Residual) → LN → MLP → DropPath(γ₂·) → (+Residual)

    对应 MambaVision 论文中 Block 的设计 (mamba_vision.py:494-531)。
    """

    def __init__(
        self,
        dim,
        use_attention=False,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path=0.1,
        layer_scale=1e-5,
        d_state=16,
        d_conv=3,
        expand=1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        if use_attention:
            self.mixer = SelfAttention3D(dim, heads=num_heads)
        else:
            self.mixer = MambaVisionMixer3D(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
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


class HybridTokenEncoder(nn.Module):
    """
    MambaVision-lite Hybrid Token Encoder
    替换 IM-Fuse 中单个模态的 Intra-modal Transformer。

    接口不变：
        输入: tokens (B, 512, 512) + pos (1, 512, 512)
        输出: tokens (B, 512, 512)

    参数:
        dim: token 维度 (512)
        num_mamba_blocks: MV-Mixer block 数量
        num_attn_blocks: Self-Attention block 数量 (放在 Mamba blocks 之后, Late Attention)
        num_heads: 注意力头数
        mlp_ratio: 共享 MLP 隐藏层扩展比（向后兼容）
        mamba_mlp_ratio: MV-Mixer block 的 MLP 扩展比
        attn_mlp_ratio: Attention block 的 MLP 扩展比
        drop_path: DropPath 概率
        layer_scale: 共享 Layer Scale 初始值（向后兼容）
        mamba_layer_scale: MV-Mixer block 的 Layer Scale 初始值
        attn_layer_scale: Attention block 的 Layer Scale 初始值
        d_state: SSM 状态维度
        d_conv: SSM 深度卷积核大小
        expand: SSM 扩展因子
    """

    def __init__(
        self,
        dim=512,
        num_mamba_blocks=1,
        num_attn_blocks=1,
        num_heads=8,
        mlp_ratio=4.0,
        mamba_mlp_ratio=None,
        attn_mlp_ratio=None,
        drop_path=0.1,
        layer_scale=1e-2,
        mamba_layer_scale=None,
        attn_layer_scale=None,
        d_state=16,
        d_conv=3,
        expand=1,
    ):
        super().__init__()
        self.num_mamba_blocks = num_mamba_blocks
        self.num_attn_blocks = num_attn_blocks
        total_depth = num_mamba_blocks + num_attn_blocks
        drop_path_rates = torch.linspace(0, drop_path, total_depth).tolist() if total_depth > 0 else []
        mamba_mlp_ratio = mlp_ratio if mamba_mlp_ratio is None else mamba_mlp_ratio
        attn_mlp_ratio = mlp_ratio if attn_mlp_ratio is None else attn_mlp_ratio
        mamba_layer_scale = layer_scale if mamba_layer_scale is None else mamba_layer_scale
        attn_layer_scale = layer_scale if attn_layer_scale is None else attn_layer_scale

        blocks = []
        # Mamba blocks (前半部分)
        for block_index in range(num_mamba_blocks):
            blocks.append(
                HybridBlock(
                    dim=dim,
                    use_attention=False,
                    num_heads=num_heads,
                    mlp_ratio=mamba_mlp_ratio,
                    drop_path=drop_path_rates[block_index],
                    layer_scale=mamba_layer_scale,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )
        # Attention blocks (后半部分, Late Attention)
        for block_index in range(num_attn_blocks):
            blocks.append(
                HybridBlock(
                    dim=dim,
                    use_attention=True,
                    num_heads=num_heads,
                    mlp_ratio=attn_mlp_ratio,
                    drop_path=drop_path_rates[num_mamba_blocks + block_index],
                    layer_scale=attn_layer_scale,
                )
            )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, pos):
        """
        Args:
            x: (B, num_tokens, dim) token 序列
            pos: (1, num_tokens, dim) 位置编码
        Returns:
            (B, num_tokens, dim)
        """
        # 位置编码仅在输入时加一次
        x = x + pos
        for block in self.blocks:
            x = block(x)
        return x
