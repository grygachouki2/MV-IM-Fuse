"""
三阶段迁移训练脚本 for IMFuseHybrid (MambaVision-lite × IM-Fuse)

基于 train_poly.py 扩展，核心新增：
1. 三阶段训练循环 (Warm-up MV-Mixer → Fine-tune Token Encoder → End-to-end)
2. 分组学习率 + 参数冻结
3. 阶段间 Checkpoint 管理
4. 从 IM-Fuse 检查点加载预训练权重
"""

import argparse
import os
import time
import logging
import numpy as np
import wandb
import torch
import torch.optim
from utils.random_seed import setup_seed
from IMFuse_hybrid import IMFuseHybrid
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii, Brats_loadall_val_nii
from data.data_utils import init_fn
from utils import Parser, criterions
from utils.parser import setup
from utils.lr_scheduler import MultiEpochsDataLoader
from predict import AverageMeter, test_softmax


# ============ 命令行参数 ============

parser = argparse.ArgumentParser()

# 基础训练参数 (与 train_poly.py 一致)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--dataname', default='BRATS2023', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float, help='Base learning rate')
parser.add_argument('--weight_decay', default=3e-5, type=float)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=999, type=int)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--interleaved_tokenization', action='store_true', default=False)
parser.add_argument('--mamba_skip', action='store_true', default=False)
parser.add_argument('--first_skip', action='store_true', default=False)

# 模型配置
parser.add_argument('--num_mamba_blocks', default=1, type=int, help='MV-Mixer block 数量')
parser.add_argument('--num_attn_blocks', default=1, type=int, help='Attention block 数量')
parser.add_argument('--drop_path', default=0.1, type=float, help='Drop path rate')
parser.add_argument('--hybrid_mlp_ratio', default=4.0, type=float, help='Hybrid Encoder MLP ratio')

# 三阶段训练
parser.add_argument('--stage', default=1, type=int, choices=[1, 2, 3], help='当前训练阶段')
parser.add_argument('--stage1_epochs', default=50, type=int)
parser.add_argument('--stage2_epochs', default=100, type=int)
parser.add_argument('--stage3_epochs', default=300, type=int)

# 预训练权重
parser.add_argument('--pretrained_imfuse', default=None, type=str, help='IM-Fuse 预训练检查点路径')
parser.add_argument('--stage_resume', default=None, type=str, help='当前阶段继续训练的检查点')

# 验证频率
parser.add_argument('--val_interval', default=10, type=int, help='验证间隔 (epochs)')

path = os.path.dirname(__file__)


# ============ 分组学习率调度器 ============

class GroupPolyLR:
    """分组 Poly LR Scheduler，每个参数组有独立的 base LR。"""

    def __init__(self, base_lrs, num_epochs):
        self.base_lrs = base_lrs
        self.num_epochs = num_epochs

    def __call__(self, optimizer, epoch):
        factor = np.power(1 - np.float32(epoch) / np.float32(self.num_epochs), 0.9)
        for i, base_lr in enumerate(self.base_lrs):
            now_lr = round(base_lr * factor, 8)
            optimizer.param_groups[i]['lr'] = now_lr
        return optimizer.param_groups[0]['lr']


# ============ 参数分组与冻结 ============

def get_param_groups(model, stage, base_lr):
    """
    根据训练阶段返回不同的参数组，实现分组学习率 + 冻结。

    Stage 1: 仅训练 MV-Mixer blocks
    Stage 2: MV-Mixer + Hybrid Encoder 中的 Attention blocks
    Stage 3: 全部解冻，分组差异化学习率
    """
    modalities = ['flair', 't1ce', 't1', 't2']

    # 分类参数
    mv_mixer_params = []       # 新增 MV-Mixer blocks
    hybrid_attn_params = []    # Hybrid Encoder 中的 Attention blocks + MLP/Norm
    conv_encoder_params = []   # 4 个 Conv Encoder
    fusion_params = []         # I-MFB + Multimodal Transformer
    decoder_params = []        # Decoder_fuse + Decoder_sep
    other_params = []          # encode_conv, decode_conv, pos 等

    for name, param in model.named_parameters():
        matched = False

        # MV-Mixer blocks (Mamba mixer部分)
        for mod in modalities:
            # blocks.0, blocks.1 等 (前 num_mamba_blocks 个是 Mamba blocks)
            if f'{mod}_hybrid_encoder' in name:
                # 获取 block index
                parts = name.split('.')
                block_idx = None
                for pi, p in enumerate(parts):
                    if p == 'blocks' and pi + 1 < len(parts):
                        try:
                            block_idx = int(parts[pi + 1])
                        except ValueError:
                            pass
                        break

                if block_idx is not None:
                    # 获取模型中 num_mamba_blocks 数值
                    if hasattr(model, 'module'):
                        n_mamba = model.module.num_mamba_blocks
                    else:
                        n_mamba = model.num_mamba_blocks

                    if block_idx < n_mamba:
                        mv_mixer_params.append(param)
                    else:
                        hybrid_attn_params.append(param)
                else:
                    hybrid_attn_params.append(param)
                matched = True
                break

        if matched:
            continue

        # Conv Encoder
        for mod in modalities:
            if f'{mod}_encoder' in name:
                conv_encoder_params.append(param)
                matched = True
                break
        if matched:
            continue

        # Fusion: mamba_fusion_layer*, multimodal_transformer
        if 'mamba_fusion_layer' in name or 'multimodal_transformer' in name or 'multimodal_decode_conv' in name:
            fusion_params.append(param)
            continue

        # Decoder
        if 'decoder_fuse' in name or 'decoder_sep' in name:
            decoder_params.append(param)
            continue

        # Other (encode_conv, decode_conv, pos, masker, tokenize)
        other_params.append(param)

    if stage == 1:
        # 仅训练 MV-Mixer
        for p in hybrid_attn_params + conv_encoder_params + fusion_params + decoder_params + other_params:
            p.requires_grad = False
        for p in mv_mixer_params:
            p.requires_grad = True

        param_groups = [
            {'params': mv_mixer_params, 'lr': base_lr},
        ]
        base_lrs = [base_lr]

    elif stage == 2:
        # MV-Mixer + Hybrid Attention
        for p in conv_encoder_params + fusion_params + decoder_params + other_params:
            p.requires_grad = False
        for p in mv_mixer_params + hybrid_attn_params:
            p.requires_grad = True

        param_groups = [
            {'params': mv_mixer_params, 'lr': base_lr * 0.5},
            {'params': hybrid_attn_params, 'lr': base_lr * 0.25},
        ]
        base_lrs = [base_lr * 0.5, base_lr * 0.25]

    elif stage == 3:
        # 全部解冻
        for p in mv_mixer_params + hybrid_attn_params + conv_encoder_params + fusion_params + decoder_params + other_params:
            p.requires_grad = True

        param_groups = [
            {'params': mv_mixer_params, 'lr': base_lr * 0.25},
            {'params': hybrid_attn_params, 'lr': base_lr * 0.1},
            {'params': fusion_params, 'lr': base_lr * 0.05},
            {'params': conv_encoder_params, 'lr': base_lr * 0.025},
            {'params': decoder_params, 'lr': base_lr * 0.05},
            {'params': other_params, 'lr': base_lr * 0.05},
        ]
        base_lrs = [base_lr * 0.25, base_lr * 0.1, base_lr * 0.05, base_lr * 0.025, base_lr * 0.05, base_lr * 0.05]

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # 过滤空参数组
    filtered_groups = []
    filtered_lrs = []
    for pg, lr in zip(param_groups, base_lrs):
        if len(pg['params']) > 0:
            filtered_groups.append(pg)
            filtered_lrs.append(lr)

    return filtered_groups, filtered_lrs


# ============ 权重迁移 ============

def load_imfuse_pretrained(model, checkpoint_path):
    """
    从 IM-Fuse 检查点加载权重到 IMFuseHybrid。
    - 名称匹配的权重直接复制
    - 原始 Transformer 权重映射到 Hybrid Encoder 最后一个 Attention block
    - 新增 MV-Mixer 参数保持随机初始化
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrained_dict = checkpoint['state_dict']

    model_dict = model.state_dict()
    modalities = ['flair', 't1ce', 't1', 't2']

    transferred = {}
    skipped = []

    # 1. 直接复制所有名称匹配的权重
    for k, v in pretrained_dict.items():
        # 去掉 'module.' 前缀
        clean_k = k.replace('module.', '') if k.startswith('module.') else k

        if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
            transferred[clean_k] = v
        else:
            skipped.append(k)

    # 2. 映射原始 Transformer 权重到 Hybrid Encoder 的最后一个 Attention block
    if hasattr(model, 'module'):
        n_mamba = model.module.num_mamba_blocks
    else:
        n_mamba = model.num_mamba_blocks
    attn_block_idx = n_mamba  # Attention block 紧跟 Mamba blocks

    for mod in modalities:
        old_prefix = f'{mod}_transformer'
        new_prefix = f'{mod}_hybrid_encoder.blocks.{attn_block_idx}'

        # 映射 SelfAttention 权重
        # 原始: cross_attention_list.0.fn.fn.{qkv,proj}.{weight,bias}
        # 新: blocks.{idx}.mixer.{qkv,proj}.{weight,bias}
        attn_mapping = {
            f'{old_prefix}.cross_attention_list.0.fn.fn.qkv.weight':
                f'{new_prefix}.mixer.qkv.weight',
            f'{old_prefix}.cross_attention_list.0.fn.fn.proj.weight':
                f'{new_prefix}.mixer.proj.weight',
            f'{old_prefix}.cross_attention_list.0.fn.fn.proj.bias':
                f'{new_prefix}.mixer.proj.bias',
        }

        # 映射 LayerNorm 权重
        # 原始 PreNormDrop: cross_attention_list.0.fn.norm.{weight,bias}
        # 新: blocks.{idx}.norm1.{weight,bias}
        attn_mapping.update({
            f'{old_prefix}.cross_attention_list.0.fn.norm.weight':
                f'{new_prefix}.norm1.weight',
            f'{old_prefix}.cross_attention_list.0.fn.norm.bias':
                f'{new_prefix}.norm1.bias',
        })

        # FFN 的 LayerNorm
        # 原始 PreNorm: cross_ffn_list.0.fn.norm.{weight,bias}
        # 新: blocks.{idx}.norm2.{weight,bias}
        attn_mapping.update({
            f'{old_prefix}.cross_ffn_list.0.fn.norm.weight':
                f'{new_prefix}.norm2.weight',
            f'{old_prefix}.cross_ffn_list.0.fn.norm.bias':
                f'{new_prefix}.norm2.bias',
        })

        # FFN 权重映射 (仅当维度匹配时)
        # 原始: cross_ffn_list.0.fn.fn.net.{0,3}.{weight,bias}  (dim→4096→dim)
        # 新: blocks.{idx}.mlp.{0,2}.{weight,bias}  (dim→mlp_hidden→dim)
        ffn_mapping = {
            f'{old_prefix}.cross_ffn_list.0.fn.fn.net.0.weight':
                f'{new_prefix}.mlp.0.weight',
            f'{old_prefix}.cross_ffn_list.0.fn.fn.net.0.bias':
                f'{new_prefix}.mlp.0.bias',
            f'{old_prefix}.cross_ffn_list.0.fn.fn.net.3.weight':
                f'{new_prefix}.mlp.2.weight',
            f'{old_prefix}.cross_ffn_list.0.fn.fn.net.3.bias':
                f'{new_prefix}.mlp.2.bias',
        }

        for old_key, new_key in attn_mapping.items():
            # 处理 module. 前缀
            full_old = f'module.{old_key}'
            if full_old in pretrained_dict:
                src_key = full_old
            elif old_key in pretrained_dict:
                src_key = old_key
            else:
                continue

            if new_key in model_dict and pretrained_dict[src_key].shape == model_dict[new_key].shape:
                transferred[new_key] = pretrained_dict[src_key]

        for old_key, new_key in ffn_mapping.items():
            full_old = f'module.{old_key}'
            if full_old in pretrained_dict:
                src_key = full_old
            elif old_key in pretrained_dict:
                src_key = old_key
            else:
                continue

            if new_key in model_dict and pretrained_dict[src_key].shape == model_dict[new_key].shape:
                transferred[new_key] = pretrained_dict[src_key]

    # 3. 加载
    model_dict.update(transferred)
    model.load_state_dict(model_dict, strict=True)

    print(f"权重迁移完成: {len(transferred)}/{len(model_dict)} 参数已加载")
    print(f"跳过 (原始模型独有): {len(skipped)} 个参数")

    # 4. 报告未初始化的参数
    uninit = [k for k in model_dict if k not in transferred]
    print(f"随机初始化 (新增模块): {len(uninit)} 个参数")
    for k in uninit[:20]:
        print(f"  - {k}")
    if len(uninit) > 20:
        print(f"  ... 共 {len(uninit)} 个")


def load_stage_checkpoint(model, checkpoint_path):
    """加载阶段检查点到 IMFuseHybrid (完整模型权重)。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint


# ============ 主函数 ============

def main():
    args = parser.parse_args()
    setup(args, 'training')
    args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
    args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

    ckpts = args.savepath
    os.makedirs(ckpts, exist_ok=True)

    # Modality missing masks
    masks = [
        [False, False, False, True], [False, True, False, False],
        [False, False, True, False], [True, False, False, False],
        [False, True, False, True], [False, True, True, False],
        [True, False, True, False], [False, False, True, True],
        [True, False, False, True], [True, True, False, False],
        [True, True, True, False], [True, False, True, True],
        [True, True, False, True], [False, True, True, True],
        [True, True, True, True],
    ]
    masks_torch = torch.from_numpy(np.array(masks))
    mask_name = [
        't2', 't1c', 't1', 'flair',
        't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
        'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
        'flairt1cet1t2',
    ]

    # Seed
    setup_seed(args.seed)

    # Print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25 - len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    # Stage config
    stage = args.stage
    stage_epochs = getattr(args, f'stage{stage}_epochs')

    # Validation check epochs
    if stage_epochs <= 50:
        val_interval = max(5, args.val_interval)
    else:
        val_interval = args.val_interval
    val_check = list(range(val_interval, stage_epochs + 1, val_interval))
    if stage_epochs not in val_check:
        val_check.append(stage_epochs)
    print(f"Stage {stage}: {stage_epochs} epochs, val at {val_check}")

    # Init wandb
    slurm_job_id = os.getenv("SLURM_JOB_ID", "local")
    exp_name = f'E{args.num_mamba_blocks}M{args.num_attn_blocks}A'
    wandb_name = f'{args.dataname}_Hybrid_{exp_name}_S{stage}_jobid{slurm_job_id}'
    wandb.init(
        project="SegmentationMM",
        name=wandb_name,
        id=wandb_name,
        mode='online',
        resume="allow",
        config={
            "architecture": f"IMFuseHybrid_{exp_name}",
            "stage": stage,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "stage_epochs": stage_epochs,
            "num_mamba_blocks": args.num_mamba_blocks,
            "num_attn_blocks": args.num_attn_blocks,
            "drop_path": args.drop_path,
        },
    )

    # Dataset
    if args.dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print('dataset is error')
        exit(0)

    if args.dataname in ['BRATS2023', 'BRATS2020', 'BRATS2015']:
        train_file = 'datalist/train.txt'
        test_file = 'datalist/test15splits2.csv'
        val_file = 'datalist/val15splits.csv'
    elif args.dataname == 'BRATS2018':
        test_file = 'datalist/Brats18_test15splits.csv'
        val_file = 'datalist/Brats18_val15splits.csv'
        train_file = 'datalist/train3.txt'

    train_set = Brats_loadall_nii(
        transforms=args.train_transforms,
        root=args.datapath,
        num_cls=num_cls,
        train_file=train_file,
    )
    test_set = Brats_loadall_test_nii(
        transforms=args.test_transforms,
        root=args.datapath,
        test_file=test_file,
    )
    val_set = Brats_loadall_val_nii(
        transforms=args.test_transforms,
        root=args.datapath,
        num_cls=num_cls,
        val_file=val_file,
    )
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn,
    )
    test_loader = MultiEpochsDataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
    )
    val_loader = MultiEpochsDataLoader(
        dataset=val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
    )

    # ============ 构建模型 ============
    model = IMFuseHybrid(
        num_cls=num_cls,
        interleaved_tokenization=args.interleaved_tokenization,
        mamba_skip=args.mamba_skip,
        num_mamba_blocks=args.num_mamba_blocks,
        num_attn_blocks=args.num_attn_blocks,
        drop_path=args.drop_path,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
    )
    print(model)

    # ============ 加载权重 ============
    val_Dice_best = -999999
    start_epoch = 0

    if args.stage_resume is not None:
        # 阶段内继续训练 或 从上一阶段的 checkpoint 开始新阶段
        checkpoint = load_stage_checkpoint(model, args.stage_resume)
        loaded_stage = checkpoint.get('stage', stage)
        if loaded_stage == stage:
            # 同阶段继续
            start_epoch = checkpoint['epoch'] + 1
            val_Dice_best = checkpoint.get('val_Dice_best', -999999)
            print(f"继续 Stage {stage}, 从 epoch {start_epoch}")
        else:
            # 新阶段 (从上一阶段 checkpoint 开始)
            start_epoch = 0
            print(f"从 Stage {loaded_stage} checkpoint 开始 Stage {stage}")
    elif args.pretrained_imfuse is not None and stage == 1:
        # Stage 1: 从 IM-Fuse 预训练加载
        load_imfuse_pretrained(model, args.pretrained_imfuse)

    model = torch.nn.DataParallel(model).cuda()

    # ============ 配置优化器 ============
    param_groups, base_lrs = get_param_groups(model, stage, args.lr)

    # 打印参数组信息
    total_trainable = 0
    total_frozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += param.numel()
        else:
            total_frozen += param.numel()
    print(f"Trainable params: {total_trainable:,}, Frozen params: {total_frozen:,}")
    for i, pg in enumerate(param_groups):
        n_params = sum(p.numel() for p in pg['params'])
        print(f"  Group {i}: lr={pg['lr']:.2e}, params={n_params:,}")

    optimizer = torch.optim.RAdam(param_groups, weight_decay=args.weight_decay)
    lr_schedule = GroupPolyLR(base_lrs, stage_epochs)

    # 如果是同阶段继续，恢复 optimizer 状态
    if args.stage_resume is not None and checkpoint.get('stage', stage) == stage:
        try:
            optimizer.load_state_dict(checkpoint['optim_dict'])
            print("Optimizer state restored")
        except Exception as e:
            print(f"Warning: Could not restore optimizer state: {e}")

    # ============ 训练循环 ============
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info(f'############# Stage {stage} Training ############')
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)

    for epoch in range(start_epoch, stage_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        b = time.time()
        model.train()
        model.module.is_training = True

        prm_cross_loss_epoch = 0.0
        prm_dice_loss_epoch = 0.0
        fuse_cross_loss_epoch = 0.0
        fuse_dice_loss_epoch = 0.0
        sep_cross_loss_epoch = 0.0
        sep_dice_loss_epoch = 0.0
        loss_epoch = 0.0

        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            fuse_pred, sep_preds, prm_preds = model(x, mask)

            # Fuse loss
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss
            fuse_cross_loss_epoch += fuse_cross_loss
            fuse_dice_loss_epoch += fuse_dice_loss

            # Sep loss
            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss
            sep_cross_loss_epoch += sep_cross_loss
            sep_dice_loss_epoch += sep_dice_loss

            # Pyramid loss
            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss
            prm_cross_loss_epoch += prm_cross_loss
            prm_dice_loss_epoch += prm_dice_loss

            # Total loss
            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss
            else:
                loss = fuse_loss + sep_loss + prm_loss
            loss_epoch += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            msg = 'Stage {}, Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format(
                stage, (epoch + 1), stage_epochs, (i + 1), iter_per_epoch, loss.item()
            )
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            logging.info(msg)

        logging.info('train time per epoch: {}'.format(time.time() - b))

        # Log metrics
        wandb.log({
            f"stage{stage}/epoch": epoch,
            f"stage{stage}/loss": loss_epoch.cpu().detach().item() / iter_per_epoch,
            f"stage{stage}/fusecross": fuse_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
            f"stage{stage}/fusedice": fuse_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
            f"stage{stage}/sepcross": sep_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
            f"stage{stage}/sepdice": sep_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
            f"stage{stage}/prmcross": prm_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
            f"stage{stage}/prmdice": prm_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
            f"stage{stage}/learning_rate": step_lr,
        })

        # Save last checkpoint
        file_name = os.path.join(ckpts, f'stage{stage}_last.pth')
        torch.save({
            'epoch': epoch,
            'stage': stage,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_Dice_best': val_Dice_best,
        }, file_name)

        # Validation
        if (epoch + 1) in val_check or args.debug:
            print(f'Stage {stage}, Epoch {epoch + 1}: validating ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    val_loader, model, dataname=args.dataname
                )
            val_WT, val_TC, val_ET, val_ETpp = dice_score
            logging.info(
                'Validate Stage {} Epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ETpp = {:.2}, loss = {:.2}'.format(
                    stage, epoch, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item(), seg_loss.cpu().item()
                )
            )
            val_dice = (val_ET + val_WT + val_TC) / 3
            wandb.log({
                f"stage{stage}/val_epoch": epoch,
                f"stage{stage}/val_ET": val_ET.item(),
                f"stage{stage}/val_ETpp": val_ETpp.item(),
                f"stage{stage}/val_WT": val_WT.item(),
                f"stage{stage}/val_TC": val_TC.item(),
                f"stage{stage}/val_Dice": val_dice.item(),
            })

            if val_dice > val_Dice_best:
                val_Dice_best = val_dice.item()
                print(f'Stage {stage}: new best val_Dice = {val_Dice_best:.4f}, saving ...')
                file_name = os.path.join(ckpts, f'stage{stage}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'stage': stage,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'val_Dice_best': val_Dice_best,
                }, file_name)

            # Testing
            print('testing ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    test_loader, model, dataname=args.dataname
                )
            test_WT, test_TC, test_ET, test_ETpp = dice_score
            logging.info(
                'Testing Stage {} Epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ETpp = {:.2}'.format(
                    stage, epoch, test_WT.item(), test_TC.item(), test_ET.item(), test_ETpp.item()
                )
            )
            wandb.log({
                f"stage{stage}/test_WT": test_WT.item(),
                f"stage{stage}/test_TC": test_TC.item(),
                f"stage{stage}/test_ET": test_ET.item(),
                f"stage{stage}/test_ETpp": test_ETpp.item(),
            })

            model.train()
            model.module.is_training = True

    # Save stage final checkpoint
    file_name = os.path.join(ckpts, f'stage{stage}_final.pth')
    torch.save({
        'epoch': stage_epochs - 1,
        'stage': stage,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        'val_Dice_best': val_Dice_best,
    }, file_name)

    msg = 'Stage {} completed. Total time: {:.4f} hours'.format(stage, (time.time() - start) / 3600)
    logging.info(msg)
    print(msg)


if __name__ == '__main__':
    main()
