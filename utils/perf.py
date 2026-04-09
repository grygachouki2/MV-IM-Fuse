from contextlib import nullcontext

import torch
import torch.nn.functional as F


AMP_DTYPES = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def resolve_amp_dtype(name):
    if name not in AMP_DTYPES:
        raise ValueError(f'Unsupported AMP dtype: {name}')
    return AMP_DTYPES[name]


def get_effective_amp_dtype(name):
    amp_dtype = resolve_amp_dtype(name)
    if amp_dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        return torch.float16, 'fp16'
    return amp_dtype, name


def create_grad_scaler(amp_enabled, amp_dtype):
    scaler_enabled = amp_enabled and torch.cuda.is_available() and amp_dtype == torch.float16
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def get_autocast_context(amp_enabled, amp_dtype):
    if not amp_enabled or not torch.cuda.is_available():
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=amp_dtype)


def configure_torch_runtime(
    enable_tf32=True,
    cudnn_benchmark=True,
    matmul_precision='high',
    flash_attention=True,
):
    runtime = {
        'cuda_available': torch.cuda.is_available(),
        'flash_attention_requested': flash_attention,
    }

    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = cudnn_benchmark
        runtime['cudnn_benchmark'] = torch.backends.cudnn.benchmark
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = enable_tf32
            runtime['cudnn_allow_tf32'] = torch.backends.cudnn.allow_tf32

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision(matmul_precision)
        runtime['matmul_precision'] = matmul_precision

    if torch.cuda.is_available() and hasattr(torch.backends, 'cuda'):
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = enable_tf32
            runtime['matmul_allow_tf32'] = torch.backends.cuda.matmul.allow_tf32

        _set_sdp_flag('flash', flash_attention)
        _set_sdp_flag('mem_efficient', True)
        _set_sdp_flag('math', True)

        runtime['flash_sdp_enabled'] = _get_sdp_flag('flash')
        runtime['mem_efficient_sdp_enabled'] = _get_sdp_flag('mem_efficient')
        runtime['math_sdp_enabled'] = _get_sdp_flag('math')

    return runtime


def sdpa_attention(q, k, v, attn_drop_p=0.0, training=False, scale=None):
    if hasattr(F, 'scaled_dot_product_attention'):
        dropout_p = attn_drop_p if training else 0.0
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            scale=scale,
        )

    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    if attn_drop_p > 0.0 and training:
        attn = F.dropout(attn, p=attn_drop_p)
    return attn @ v


def _set_sdp_flag(name, enabled):
    setter = getattr(torch.backends.cuda, f'enable_{name}_sdp', None)
    if setter is not None:
        setter(enabled)


def _get_sdp_flag(name):
    getter = getattr(torch.backends.cuda, f'{name}_sdp_enabled', None)
    if getter is None:
        return None
    return getter()