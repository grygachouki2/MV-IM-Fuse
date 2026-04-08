"""
Sanity Check 脚本 for IMFuseHybrid

验证流程:
1. 构建 IMFuseHybrid 模型 (E2 配置)
2. (可选) 加载 IM-Fuse 预训练权重
3. 前向传播: 检查输出形状
4. 反向传播: 检查梯度流
5. 15 种 mask 组合: 确认无报错
6. (可选) Overfit 测试: 小数据集上验证 loss 收敛
7. 显存与吞吐量测量
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from IMFuse_hybrid import IMFuseHybrid
from utils import criterions

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_imfuse', default=None, type=str, help='IM-Fuse 预训练检查点 (可选)')
parser.add_argument('--num_mamba_blocks', default=1, type=int)
parser.add_argument('--num_attn_blocks', default=1, type=int)
parser.add_argument('--drop_path', default=0.1, type=float)
parser.add_argument('--overfit_test', action='store_true', help='执行 overfit 测试 (需要数据)')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--overfit_epochs', default=50, type=int)
parser.add_argument('--overfit_samples', default=20, type=int)


def check_gradients(model, name_filter=None):
    """检查梯度是否存在 NaN/Inf。"""
    issues = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if name_filter and name_filter not in name:
                continue
            if torch.isnan(param.grad).any():
                issues.append(f"  NaN gradient: {name}")
            if torch.isinf(param.grad).any():
                issues.append(f"  Inf gradient: {name}")
    return issues


def count_parameters(model):
    """统计参数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    args = parser.parse_args()
    num_cls = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("IMFuseHybrid Sanity Check")
    print(f"Config: MV-Mixer×{args.num_mamba_blocks} + Attn×{args.num_attn_blocks}")
    print(f"Device: {device}")
    print("=" * 60)

    # ============ Step 1: 构建模型 ============
    print("\n[1/6] 构建模型...")
    model = IMFuseHybrid(
        num_cls=num_cls,
        interleaved_tokenization=False,
        mamba_skip=True,
        num_mamba_blocks=args.num_mamba_blocks,
        num_attn_blocks=args.num_attn_blocks,
        drop_path=args.drop_path,
    )
    total_params, trainable_params = count_parameters(model)
    print(f"  ✓ 模型构建成功")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    # ============ Step 2: 加载预训练权重 ============
    if args.pretrained_imfuse:
        print(f"\n[2/6] 加载 IM-Fuse 预训练权重...")
        try:
            from train_hybrid import load_imfuse_pretrained
            load_imfuse_pretrained(model, args.pretrained_imfuse)
            print("  ✓ 预训练权重加载成功")
        except Exception as e:
            print(f"  ✗ 预训练权重加载失败: {e}")
            raise e
    else:
        print("\n[2/6] 跳过预训练权重加载 (使用随机初始化)")

    model = model.to(device)
    model.is_training = True

    # ============ Step 3: 前向传播 ============
    print("\n[3/6] 前向传播测试...")
    x = torch.randn(1, 4, 128, 128, 128).to(device)
    mask = torch.ones(1, 4).bool().to(device)

    try:
        torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
        t0 = time.time()
        fuse_pred, sep_preds, prm_preds = model(x, mask)
        t1 = time.time()

        assert fuse_pred.shape == (1, num_cls, 128, 128, 128), \
            f"fuse_pred shape error: {fuse_pred.shape}"
        assert len(sep_preds) == 4, f"sep_preds count error: {len(sep_preds)}"
        for i, sp in enumerate(sep_preds):
            assert sp.shape == (1, num_cls, 128, 128, 128), \
                f"sep_pred[{i}] shape error: {sp.shape}"
        assert len(prm_preds) == 4, f"prm_preds count error: {len(prm_preds)}"

        fwd_time = t1 - t0
        if device == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  ✓ 前向传播成功 ({fwd_time:.2f}s, peak GPU: {peak_mem:.0f}MB)")
        else:
            print(f"  ✓ 前向传播成功 ({fwd_time:.2f}s)")

    except Exception as e:
        print(f"  ✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============ Step 4: 反向传播 ============
    print("\n[4/6] 反向传播测试...")
    try:
        target = torch.zeros(1, num_cls, 128, 128, 128).to(device)
        target[:, 0] = 1  # 全背景

        loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
        loss.backward()

        # 检查 MV-Mixer 梯度
        has_grad = False
        for name, param in model.named_parameters():
            if 'hybrid_encoder' in name and param.requires_grad:
                if param.grad is not None:
                    has_grad = True
                    break

        if not has_grad:
            print("  ✗ 警告: hybrid_encoder 参数无梯度!")
        else:
            issues = check_gradients(model)
            if issues:
                print("  ✗ 梯度存在问题:")
                for issue in issues:
                    print(issue)
            else:
                print("  ✓ 反向传播正常，梯度无 NaN/Inf")

    except Exception as e:
        print(f"  ✗ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    model.zero_grad()

    # ============ Step 5: 15 种 mask 组合 ============
    print("\n[5/6] 15 种缺失模态组合测试...")
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
    mask_names = [
        't2', 't1c', 't1', 'flair',
        't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
        'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
        'flairt1cet1t2',
    ]

    model.is_training = False
    model.eval()
    all_ok = True
    with torch.no_grad():
        for i, (m, name) in enumerate(zip(masks, mask_names)):
            mask_tensor = torch.tensor([m]).to(device)
            try:
                pred = model(x, mask_tensor)
                assert pred.shape == (1, num_cls, 128, 128, 128)
            except Exception as e:
                print(f"  ✗ mask[{i}] {name} 失败: {e}")
                all_ok = False

    if all_ok:
        print("  ✓ 全部 15 种 mask 组合通过")
    model.is_training = True

    # ============ Step 6: 显存与吞吐量 ============
    print("\n[6/6] 性能测量...")
    if device == 'cuda':
        model.train()
        model.is_training = True

        # Warmup
        mask_full = torch.ones(1, 4).bool().to(device)
        for _ in range(3):
            fuse_pred, sep_preds, prm_preds = model(x, mask_full)
            loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            loss.backward()
            model.zero_grad()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Measure
        times = []
        for _ in range(5):
            t0 = time.time()
            fuse_pred, sep_preds, prm_preds = model(x, mask_full)
            loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            loss.backward()
            model.zero_grad()
            torch.cuda.synchronize()
            times.append(time.time() - t0)

        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        avg_time = np.mean(times)
        print(f"  训练步骤平均时间: {avg_time:.3f}s")
        print(f"  峰值 GPU 显存: {peak_mem:.0f}MB ({peak_mem/1024:.2f}GB)")
    else:
        print("  跳过 (无 GPU)")

    # ============ 总结 ============
    print("\n" + "=" * 60)
    print("Sanity Check 完成!")
    print(f"  模型配置: MV-Mixer×{args.num_mamba_blocks} + Attn×{args.num_attn_blocks}")
    print(f"  参数量: {total_params:,} (可训练: {trainable_params:,})")
    print("=" * 60)

    # ============ 可选: Overfit 测试 ============
    if args.overfit_test:
        if args.datapath is None:
            print("\n跳过 overfit 测试: 未提供 --datapath")
            return

        print(f"\n[Bonus] Overfit 测试 ({args.overfit_samples} samples, {args.overfit_epochs} epochs)...")
        import data.transforms as data_transforms
        from data.datasets_nii import Brats_loadall_nii
        from data.data_utils import init_fn

        train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
        train_set = Brats_loadall_nii(
            transforms=train_transforms,
            root=args.datapath,
            num_cls=num_cls,
            train_file='datalist/train.txt',
        )

        # 取前 N 个样本
        from torch.utils.data import Subset, DataLoader
        indices = list(range(min(args.overfit_samples, len(train_set))))
        subset = Subset(train_set, indices)
        loader = DataLoader(subset, batch_size=1, shuffle=True, num_workers=2)

        model.train()
        model.is_training = True
        optimizer = torch.optim.RAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=2e-4, weight_decay=3e-5,
        )

        for epoch in range(args.overfit_epochs):
            epoch_loss = 0.0
            for data in loader:
                x_train, target_train, mask_train = data[:3]
                x_train = x_train.to(device)
                target_train = target_train.to(device)
                mask_train = mask_train.to(device)

                fuse_pred, sep_preds, prm_preds = model(x_train, mask_train)
                fuse_loss = (
                    criterions.softmax_weighted_loss(fuse_pred, target_train, num_cls=num_cls)
                    + criterions.dice_loss(fuse_pred, target_train, num_cls=num_cls)
                )
                sep_loss = torch.zeros(1).to(device)
                for sp in sep_preds:
                    sep_loss += criterions.softmax_weighted_loss(sp, target_train, num_cls=num_cls)
                    sep_loss += criterions.dice_loss(sp, target_train, num_cls=num_cls)

                loss = fuse_loss + sep_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{args.overfit_epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < 0.5:
            print(f"  ✓ Overfit 测试通过 (final loss: {avg_loss:.4f})")
        else:
            print(f"  ⚠ Overfit 测试: loss 未充分收敛 (final loss: {avg_loss:.4f})")


if __name__ == '__main__':
    main()
