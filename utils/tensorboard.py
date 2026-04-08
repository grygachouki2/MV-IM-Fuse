import os

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


DEFAULT_MODALITY_NAMES = ('flair', 't1ce', 't1', 't2')


def add_tensorboard_args(parser):
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--no_tensorboard', dest='tensorboard', action='store_false', help='Disable TensorBoard logging')
    parser.set_defaults(tensorboard=True)
    parser.add_argument('--tensorboard_dir', default=None, type=str, help='Override TensorBoard log directory')
    parser.add_argument('--tb_log_interval', default=10, type=int, help='Log training scalars every N optimization steps')
    parser.add_argument('--tb_image_interval', default=200, type=int, help='Log training preview slices every N optimization steps; 0 disables image previews')
    parser.add_argument('--tb_histogram_interval', default=10, type=int, help='Log weight and gradient histograms every N epochs; 0 disables histograms')
    parser.add_argument('--tb_histogram_limit', default=20, type=int, help='Maximum number of trainable parameters to include in each histogram logging event; 0 logs all trainable parameters')
    parser.add_argument('--tb_flush_secs', default=30, type=int, help='TensorBoard flush interval in seconds')


def create_tensorboard_writer(args, run_name, subdir=None, purge_step=None):
    if not getattr(args, 'tensorboard', True):
        return None
    if SummaryWriter is None:
        raise ImportError(
            'TensorBoard logging is enabled, but the tensorboard package is not installed. '
            'Install it with `pip install tensorboard` or rerun with --no_tensorboard.'
        )

    base_dir = getattr(args, 'tensorboard_dir', None) or os.path.join(args.savepath, 'tensorboard')
    log_dir = os.path.join(base_dir, subdir) if subdir else base_dir
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(
        log_dir=log_dir,
        flush_secs=getattr(args, 'tb_flush_secs', 30),
        purge_step=purge_step,
    )
    writer.add_text('run/name', run_name, 0)
    writer.add_text('run/args', _format_args(args), 0)
    writer.add_text('run/log_dir', log_dir, 0)
    return writer


def log_scalars(writer, scalars, step):
    if writer is None:
        return
    for tag, value in scalars.items():
        scalar = _to_scalar(value)
        if scalar is None:
            continue
        writer.add_scalar(tag, scalar, step)


def log_learning_rates(writer, optimizer, step, prefix='lr'):
    if writer is None:
        return
    for index, param_group in enumerate(optimizer.param_groups):
        writer.add_scalar(f'{prefix}/group_{index}', float(param_group['lr']), step)


def log_mask_stats(writer, mask, step, prefix='mask', modality_names=None):
    if writer is None or mask is None:
        return

    mask_cpu = mask.detach().float().cpu()
    if mask_cpu.ndim != 2:
        return

    modality_names = modality_names or DEFAULT_MODALITY_NAMES
    log_scalars(
        writer,
        {
            f'{prefix}/modalities_present_mean': mask_cpu.sum(dim=1).mean().item(),
            f'{prefix}/modalities_missing_mean': (mask_cpu.size(1) - mask_cpu.sum(dim=1)).mean().item(),
        },
        step,
    )

    for index in range(mask_cpu.size(1)):
        name = modality_names[index] if index < len(modality_names) else f'modality_{index}'
        writer.add_scalar(f'{prefix}/present_ratio/{name}', mask_cpu[:, index].mean().item(), step)


def log_parameter_histograms(writer, model, step, max_parameters=20, prefix='model'):
    if writer is None:
        return

    logged = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if max_parameters and logged >= max_parameters:
            break

        clean_name = name.replace('.', '/')
        writer.add_histogram(f'{prefix}/weights/{clean_name}', parameter.detach().float().cpu(), step)
        if parameter.grad is not None:
            writer.add_histogram(f'{prefix}/grads/{clean_name}', parameter.grad.detach().float().cpu(), step)
        logged += 1


def log_preview_batch(writer, tag, inputs, target, prediction, step, modality_names=None):
    if writer is None or inputs is None or target is None or prediction is None:
        return

    modality_names = modality_names or DEFAULT_MODALITY_NAMES
    input_sample = inputs[0].detach().float().cpu()
    target_sample = target[0].detach().float().cpu()
    prediction_sample = prediction[0].detach().float().cpu()

    num_modalities = min(input_sample.size(0), len(modality_names))
    for index in range(num_modalities):
        writer.add_image(
            f'{tag}/input/{modality_names[index]}',
            _normalize_slice(input_sample[index]).unsqueeze(0),
            step,
        )

    target_slice = _label_slice(target_sample)
    prediction_slice = _label_slice(prediction_sample)
    writer.add_image(f'{tag}/target', target_slice.unsqueeze(0), step)
    writer.add_image(f'{tag}/prediction', prediction_slice.unsqueeze(0), step)


def compute_global_norm(parameters, use_grad=False):
    total = 0.0
    for parameter in parameters:
        tensor = parameter.grad if use_grad else parameter.data
        if tensor is None:
            continue
        total += torch.sum(tensor.detach().float() ** 2).item()
    return total ** 0.5


def _format_args(args):
    return '\n'.join(f'{key}: {value}' for key, value in sorted(vars(args).items()))


def _to_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_slice(volume_slice):
    image = _middle_slice(volume_slice)
    image = image - image.min()
    max_value = image.max()
    if max_value > 0:
        image = image / max_value
    return image


def _label_slice(volume):
    if volume.ndim == 4:
        volume = torch.argmax(volume, dim=0)
    image = _middle_slice(volume.float())
    max_value = image.max()
    if max_value > 0:
        image = image / max_value
    return image


def _middle_slice(volume):
    if volume.ndim != 3:
        raise ValueError(f'Expected a 3D volume, but got shape {tuple(volume.shape)}')
    return volume[:, :, volume.shape[-1] // 2]