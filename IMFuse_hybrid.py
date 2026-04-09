"""
IMFuse_hybrid: MambaVision-lite × IM-Fuse 混合模型

基于 IMFuse.py，仅将每个模态分支的 Intra-modal Transformer 替换为
HybridTokenEncoder (MV-Mixer × k + Transformer × m)。
其余所有模块 (Conv Encoder, I-MFB, Multimodal Transformer, Decoder, Loss) 完全不变。
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from layers import general_conv3d_prenorm, fusion_prenorm
from mamba_ssm import Mamba
from torch.cuda.amp import autocast
from utils.initialization import InitWeights_He
from mambavision_mixer import HybridTokenEncoder
from utils.perf import sdpa_attention

# ============ 全局参数 (与 IMFuse.py 完全一致) ============
basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8
input_patch_size = 128


# ============ 以下类直接从 IMFuse.py 复用，保持不动 ============

class MambaTrans(nn.Module):
    def __init__(self, channels):
        super(MambaTrans, self).__init__()
        self.mamba = Mamba(
            d_model=channels,
            d_state=min(channels, 256),
            d_conv=4,
            expand=2,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, channels)

    def forward(self, x):
        x = self.mamba(self.norm1(x)) + x
        x = self.head(self.norm2(x)) + x
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.mamba = MambaTrans(dim)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        x_mamba = self.mamba(x)
        return x_mamba


class MambaFusionLayer(nn.Module):
    def __init__(self, dim, num_tokens_fused_representation=None):
        super().__init__()
        self.dim = dim
        self.num_tokens_fused_representation = num_tokens_fused_representation
        self.fused_tokens = nn.Parameter(torch.randn(1, self.num_tokens_fused_representation, dim))
        self.mamba_layer = MambaLayer(dim)

    def forward(self, x):
        B = x.size(0)
        fused_tokens = self.fused_tokens.repeat(B, 1, 1)
        x_fused = torch.cat([x, fused_tokens], dim=1)
        x_mamba = self.mamba_layer(x_fused)
        x_mamba = x_mamba[:, -self.num_tokens_fused_representation:, :]
        return x_mamba


class MambaFusionCatLayer(nn.Module):
    def __init__(self, dim, num_tokens_fused_representation=None):
        super().__init__()
        self.dim = dim
        self.num_tokens_fused_representation = num_tokens_fused_representation
        self.fused_tokens = nn.Parameter(torch.randn(1, self.num_tokens_fused_representation, dim))
        self.mamba_layer = MambaLayer(dim)

    def forward(self, x):
        B = x[0].size(0)
        fused_tokens = self.fused_tokens.repeat(B, 1, 1)
        x = torch.stack([*x, fused_tokens], dim=2)
        x = x.view(B, -1, self.dim)
        x = self.mamba_layer(x)
        x = x[:, 4::5, :]
        return x


class Tokenize(nn.Module):
    def __init__(self, dims, num_modals=4):
        super(Tokenize, self).__init__()
        self.dims = dims
        self.num_modals = num_modals

    def forward(self, x):
        flair_intra_x, t1ce_intra_x, t1_intra_x, t2_intra_x = torch.chunk(x, self.num_modals, dim=1)
        multimodal_token_x = torch.cat((
            flair_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),
            t1ce_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),
            t1_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),
            t2_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),
        ), dim=1)
        return multimodal_token_x


class TokenizeSep(nn.Module):
    def __init__(self, dims, num_modals=4):
        super(TokenizeSep, self).__init__()
        self.dims = dims
        self.num_modals = num_modals

    def forward(self, x):
        flair_intra_x, t1ce_intra_x, t1_intra_x, t2_intra_x = torch.chunk(x, self.num_modals, dim=1)
        return (
            flair_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),
            t1ce_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),
            t1_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),
            t2_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),
        )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))
        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))
        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))
        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))
        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))
        return x1, x2, x3, x4, x5


class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()
        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))
        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))
        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))
        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))
        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))
        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)
        return pred


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4, mamba_skip=False, first_skip=True):
        super(Decoder_fuse, self).__init__()
        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        if first_skip:
            self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        else:
            self.d1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims * 16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims * 8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims * 4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims * 2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims * 16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims * 8, num_cls=1 if mamba_skip else num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims * 4, num_cls=1 if mamba_skip else num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims * 2, num_cls=1 if mamba_skip else num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims * 1, num_cls=1 if mamba_skip else num_cls) if first_skip else None
        self.mamba_skip = mamba_skip
        self.first_skip = first_skip

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        if self.first_skip:
            de_x1 = self.RFM1(x1)
            de_x1 = torch.cat((de_x1, de_x2), dim=1)
            de_x1 = self.d1_out(self.d1_c2(de_x1))
        else:
            de_x1 = self.d1_out(self.d1_c2(de_x2))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


# ============ 原始 Transformer 相关类 (用于 InterFormer, 保持不变) ============

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
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


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )
        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)

    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


# ============ IMFuseHybrid: 核心模型 ============

class IMFuseHybrid(nn.Module):
    """
    MambaVision-lite × IM-Fuse 混合模型。
    
    与原始 IMFuse 的唯一区别：
    - flair_transformer / t1ce_transformer / t1_transformer / t2_transformer
      → 替换为 flair_hybrid_encoder / t1ce_hybrid_encoder / t1_hybrid_encoder / t2_hybrid_encoder
    
    新增参数:
        num_mamba_blocks: MV-Mixer block 数量 (默认 1)
        num_attn_blocks: Self-Attention block 数量 (默认 1)  
        drop_path: DropPath 概率 (默认 0.1)
        hybrid_mlp_ratio: Hybrid Encoder MLP 扩展比 (默认 8.0，可完整复用 IM-Fuse FFN 权重)
    """

    def __init__(
        self,
        num_cls=4,
        interleaved_tokenization=False,
        mamba_skip=False,
        first_skip=True,
        num_mamba_blocks=1,
        num_attn_blocks=1,
        drop_path=0.1,
        hybrid_mlp_ratio=8.0,
        hybrid_layer_scale=0.0,
    ):
        super(IMFuseHybrid, self).__init__()
        self.interleaved_tokenization = interleaved_tokenization
        self.num_mamba_blocks = num_mamba_blocks
        self.num_attn_blocks = num_attn_blocks

        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        if self.interleaved_tokenization:
            TokenizerClass = TokenizeSep
            MambaFusionLayerClass = MambaFusionCatLayer
        else:
            TokenizerClass = Tokenize
            MambaFusionLayerClass = MambaFusionLayer

        ########### IntraFormer → HybridTokenEncoder
        self.flair_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)

        self.flair_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t1ce_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t1_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t2_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)

        self.flair_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.fused_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))

        # ★ 核心变更：用 HybridTokenEncoder 替换原始 Transformer
        hybrid_kwargs = dict(
            dim=transformer_basic_dims,
            num_mamba_blocks=num_mamba_blocks,
            num_attn_blocks=num_attn_blocks,
            num_heads=num_heads,
            mlp_ratio=hybrid_mlp_ratio,
            drop_path=drop_path,
            layer_scale=hybrid_layer_scale if hybrid_layer_scale > 0 else None,
        )
        self.flair_hybrid_encoder = HybridTokenEncoder(**hybrid_kwargs)
        self.t1ce_hybrid_encoder = HybridTokenEncoder(**hybrid_kwargs)
        self.t1_hybrid_encoder = HybridTokenEncoder(**hybrid_kwargs)
        self.t2_hybrid_encoder = HybridTokenEncoder(**hybrid_kwargs)
        ########### IntraFormer

        ########### InterFormer (完全不变)
        self.mamba_fusion_layer = MambaFusionLayer(dim=transformer_basic_dims, num_tokens_fused_representation=patch_size ** 3)
        self.multimodal_transformer = Transformer(
            embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals
        )
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16 * num_modals, kernel_size=1, padding=0)
        ########### InterFormer

        self.masker = MaskModal()

        ######## Skip Connections (完全不变)
        if mamba_skip and not first_skip:
            self.tokenize = nn.ModuleList([
                TokenizerClass(dims=16, num_modals=num_modals),
                TokenizerClass(dims=32, num_modals=num_modals),
                TokenizerClass(dims=64, num_modals=num_modals),
                TokenizerClass(dims=512, num_modals=num_modals),
            ])
            self.mamba_fusion_layers = nn.ModuleList([
                MambaFusionLayerClass(dim=16, num_tokens_fused_representation=64 ** 3),
                MambaFusionLayerClass(dim=32, num_tokens_fused_representation=32 ** 3),
                MambaFusionLayerClass(dim=64, num_tokens_fused_representation=16 ** 3),
                MambaFusionLayerClass(dim=512, num_tokens_fused_representation=8 ** 3),
            ])
        else:
            self.tokenize = nn.ModuleList([
                TokenizerClass(dims=8, num_modals=num_modals),
                TokenizerClass(dims=16, num_modals=num_modals),
                TokenizerClass(dims=32, num_modals=num_modals),
                TokenizerClass(dims=64, num_modals=num_modals),
                TokenizerClass(dims=512, num_modals=num_modals),
            ])
            self.mamba_fusion_layers = nn.ModuleList([
                MambaFusionLayerClass(dim=8, num_tokens_fused_representation=128 ** 3),
                MambaFusionLayerClass(dim=16, num_tokens_fused_representation=64 ** 3),
                MambaFusionLayerClass(dim=32, num_tokens_fused_representation=32 ** 3),
                MambaFusionLayerClass(dim=64, num_tokens_fused_representation=16 ** 3),
                MambaFusionLayerClass(dim=512, num_tokens_fused_representation=8 ** 3),
            ])
        ########

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls, mamba_skip=mamba_skip, first_skip=first_skip)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.is_training = False
        self.mamba_skip = mamba_skip
        self.first_skip = first_skip

        self.apply(InitWeights_He(1e-2))

    def forward(self, x, mask):
        # Extract features from different layers
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])

        ########### IntraFormer → HybridTokenEncoder
        flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)

        # ★ 核心变更：使用 HybridTokenEncoder
        flair_intra_token_x5 = self.flair_hybrid_encoder(flair_token_x5, self.flair_pos)
        t1ce_intra_token_x5 = self.t1ce_hybrid_encoder(t1ce_token_x5, self.t1ce_pos)
        t1_intra_token_x5 = self.t1_hybrid_encoder(t1_token_x5, self.t1_pos)
        t2_intra_token_x5 = self.t2_hybrid_encoder(t2_token_x5, self.t2_pos)

        flair_intra_x5 = flair_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
        ########### IntraFormer

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1), mask)

        ########### Mamba Skip
        if self.mamba_skip:
            if self.first_skip:
                x1 = self.tokenize[0](x1)
                x1 = self.mamba_fusion_layers[0](x1)
                x1 = x1.view(x.size(0), input_patch_size, input_patch_size, input_patch_size, basic_dims).permute(0, 4, 1, 2, 3).contiguous()

                x2 = self.tokenize[1](x2)
                x2 = self.mamba_fusion_layers[1](x2)
                x2 = x2.view(x.size(0), input_patch_size // 2, input_patch_size // 2, input_patch_size // 2, basic_dims * 2).permute(0, 4, 1, 2, 3).contiguous()

                x3 = self.tokenize[2](x3)
                x3 = self.mamba_fusion_layers[2](x3)
                x3 = x3.view(x.size(0), input_patch_size // 4, input_patch_size // 4, input_patch_size // 4, basic_dims * 4).permute(0, 4, 1, 2, 3).contiguous()

                x4 = self.tokenize[3](x4)
                x4 = self.mamba_fusion_layers[3](x4)
                x4 = x4.view(x.size(0), input_patch_size // 8, input_patch_size // 8, input_patch_size // 8, basic_dims * 8).permute(0, 4, 1, 2, 3).contiguous()
            else:
                x2 = self.tokenize[0](x2)
                x2 = self.mamba_fusion_layers[0](x2)
                x2 = x2.view(x.size(0), input_patch_size // 2, input_patch_size // 2, input_patch_size // 2, basic_dims * 2).permute(0, 4, 1, 2, 3).contiguous()

                x3 = self.tokenize[1](x3)
                x3 = self.mamba_fusion_layers[1](x3)
                x3 = x3.view(x.size(0), input_patch_size // 4, input_patch_size // 4, input_patch_size // 4, basic_dims * 4).permute(0, 4, 1, 2, 3).contiguous()

                x4 = self.tokenize[2](x4)
                x4 = self.mamba_fusion_layers[2](x4)
                x4 = x4.view(x.size(0), input_patch_size // 8, input_patch_size // 8, input_patch_size // 8, basic_dims * 8).permute(0, 4, 1, 2, 3).contiguous()
        #######

        ########### MambaFusion + InterFormer (完全不变)
        multimodal_token_x5 = self.tokenize[-1](x5_intra)
        fused_multimodal = self.mamba_fusion_layers[-1](multimodal_token_x5)
        multimodal_pos = self.fused_pos.repeat(x.size(0), 1, 1)
        multimodal_inter_token_x5 = self.multimodal_transformer(fused_multimodal, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(
            multimodal_inter_token_x5.view(
                multimodal_inter_token_x5.size(0), patch_size, patch_size, patch_size, transformer_basic_dims
            ).permute(0, 4, 1, 2, 3).contiguous()
        )
        x5_inter = multimodal_inter_x5

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer

        if self.is_training:
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds
        return fuse_pred
