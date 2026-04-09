"""
测试脚本 for IMFuseHybrid (MambaVision-lite × IM-Fuse)

对 15 种模态缺失组合进行全面评估。
基于 test.py 适配，使用 IMFuseHybrid 模型。
"""

import torch
import os
import argparse
import logging
import numpy as np
from predict import AverageMeter, test_softmax
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import MultiEpochsDataLoader
from utils.checkpoint import load_local_checkpoint
from IMFuse_hybrid import IMFuseHybrid

path = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', default='BRATS2023', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--test_file', default='datalist/test15splits.csv', type=str)
parser.add_argument('--datapath', default=os.path.join(path, 'dataset', 'BRATS2023_Training_npy'), type=str)
parser.add_argument('--interleaved_tokenization', action='store_true', default=False)
parser.add_argument('--mamba_skip', action='store_true', default=False)
parser.add_argument('--first_skip', action='store_true', default=False)
parser.add_argument('--num_mamba_blocks', default=1, type=int)
parser.add_argument('--num_attn_blocks', default=1, type=int)
parser.add_argument('--drop_path', default=0.1, type=float)
parser.add_argument('--hybrid_mlp_ratio', default=8.0, type=float)
parser.add_argument('--mamba_mlp_ratio', default=4.0, type=float)
parser.add_argument('--attn_mlp_ratio', default=8.0, type=float)
parser.add_argument('--hybrid_layer_scale', default=0.0, type=float)
parser.add_argument('--mamba_layer_scale', default=1e-5, type=float)
parser.add_argument('--attn_layer_scale', default=0.0, type=float)
parser.add_argument('--save_masks', action='store_true', default=False)

if __name__ == '__main__':
    args = parser.parse_args()

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
    mask_name = [
        't2', 't1c', 't1', 'flair',
        't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
        'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
        'flairt1cet1t2',
    ]

    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    datapath = args.datapath
    test_file = args.test_file
    save_path = args.savepath
    num_cls = 4
    dataname = args.dataname
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    # Setup logging
    os.makedirs(save_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        filename=os.path.join(save_path, f'test_{index}.log'),
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
    )

    model = IMFuseHybrid(
        num_cls=num_cls,
        interleaved_tokenization=args.interleaved_tokenization,
        mamba_skip=args.mamba_skip,
        first_skip=args.first_skip,
        num_mamba_blocks=args.num_mamba_blocks,
        num_attn_blocks=args.num_attn_blocks,
        drop_path=args.drop_path,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_layer_scale=args.hybrid_layer_scale,
        mamba_mlp_ratio=args.mamba_mlp_ratio,
        attn_mlp_ratio=args.attn_mlp_ratio,
        mamba_layer_scale=args.mamba_layer_scale if args.mamba_layer_scale > 0 else None,
        attn_layer_scale=args.attn_layer_scale if args.attn_layer_scale > 0 else None,
    )
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = load_local_checkpoint(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint['epoch'] + 1
    best_stage = checkpoint.get('stage', 'unknown')
    print(f"Loaded checkpoint: stage={best_stage}, epoch={best_epoch}")

    # Test all 15 missing modality combinations
    results = {}
    for i, (m, name) in enumerate(zip(masks, mask_name)):
        print(f"\n{'='*60}")
        print(f"Testing mask {i+1}/15: {name} ({m})")
        print(f"{'='*60}")

        mask_save_dir = os.path.join(save_path, name) if args.save_masks else save_path
        os.makedirs(mask_save_dir, exist_ok=True)

        with torch.no_grad():
            dice_score = test_softmax(
                test_loader,
                model,
                dataname=dataname,
                feature_mask=m,
                compute_loss=False,
                save_masks=args.save_masks,
                save_dir=mask_save_dir,
                index=index,
            )

        results[name] = dice_score
        WT, TC, ET, ETpp = dice_score
        print(f"  WT={WT:.4f}, TC={TC:.4f}, ET={ET:.4f}, ETpp={ETpp:.4f}")

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary (Stage {best_stage}, Epoch {best_epoch})")
    print(f"{'='*80}")
    print(f"{'Mask':<20} {'WT':>8} {'TC':>8} {'ET':>8} {'ETpp':>8}")
    print(f"{'-'*52}")

    all_wt, all_tc, all_et = [], [], []
    for name in mask_name:
        WT, TC, ET, ETpp = results[name]
        print(f"{name:<20} {WT:>8.4f} {TC:>8.4f} {ET:>8.4f} {ETpp:>8.4f}")
        all_wt.append(WT)
        all_tc.append(TC)
        all_et.append(ET)

    print(f"{'-'*52}")
    print(f"{'Average':<20} {np.mean(all_wt):>8.4f} {np.mean(all_tc):>8.4f} {np.mean(all_et):>8.4f}")

    # T1c-missing subset
    t1c_missing_indices = [0, 2, 3, 6, 7, 8, 11]
    t1c_et = [all_et[i] for i in t1c_missing_indices]
    t1c_names = [mask_name[i] for i in t1c_missing_indices]
    print(f"\nT1c-missing scenarios (ET Dice):")
    for n, e in zip(t1c_names, t1c_et):
        print(f"  {n}: {e:.4f}")
    print(f"  Average: {np.mean(t1c_et):.4f}")

    # Save results
    result_file = os.path.join(save_path, f'results_stage{best_stage}_epoch{best_epoch}.txt')
    with open(result_file, 'w') as f:
        f.write(f"Stage {best_stage}, Epoch {best_epoch}\n")
        f.write(f"{'Mask':<20} {'WT':>8} {'TC':>8} {'ET':>8} {'ETpp':>8}\n")
        for name in mask_name:
            WT, TC, ET, ETpp = results[name]
            f.write(f"{name:<20} {WT:>8.4f} {TC:>8.4f} {ET:>8.4f} {ETpp:>8.4f}\n")
        f.write(f"{'Average':<20} {np.mean(all_wt):>8.4f} {np.mean(all_tc):>8.4f} {np.mean(all_et):>8.4f}\n")
    print(f"\nResults saved to {result_file}")
