import os
import numpy as np
import medpy.io as medio
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser
join=os.path.join

def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print ('#' * 100)
        ecart = int((128-(xmax-xmin))/2)
        xmax = xmax+ecart+1
        xmin = xmin-ecart
    if xmin < 0:
        xmax-=xmin
        xmin=0
    return xmin, xmax

def crop(vol):
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3

    x_dim, y_dim, z_dim = tuple(vol.shape)
    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max

def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x

    return vol


def save_npy_atomic(output_path, array):
    tmp_path = output_path + '.tmp.npy'
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    np.save(tmp_path, array)
    os.replace(tmp_path, output_path)


def process_case(src_path, tar_path, file_name):
    case_id = file_name.split('/')[-1]
    vol_out = os.path.join(tar_path, 'vol', case_id + '_vol.npy')
    seg_out = os.path.join(tar_path, 'seg', case_id + '_seg.npy')
    vol_tmp = vol_out + '.tmp.npy'
    seg_tmp = seg_out + '.tmp.npy'

    if os.path.exists(vol_out) and os.path.exists(seg_out):
        return case_id, 'skipped', None

    for tmp_path in (vol_tmp, seg_tmp):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    for out_path in (vol_out, seg_out):
        if os.path.exists(out_path):
            os.remove(out_path)

    flair, _ = medio.load(os.path.join(src_path, file_name, case_id + '-t2f.nii.gz'))
    t1ce, _ = medio.load(os.path.join(src_path, file_name, case_id + '-t1c.nii.gz'))
    t1, _ = medio.load(os.path.join(src_path, file_name, case_id + '-t1n.nii.gz'))
    t2, _ = medio.load(os.path.join(src_path, file_name, case_id + '-t2w.nii.gz'))

    vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
    x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
    vol1 = normalize(vol[:, x_min:x_max, y_min:y_max, z_min:z_max])
    vol1 = vol1.transpose(1, 2, 3, 0)

    seg, _ = medio.load(os.path.join(src_path, file_name, case_id + '-seg.nii.gz'))
    seg = seg.astype(np.uint8)
    seg1 = seg[x_min:x_max, y_min:y_max, z_min:z_max]
    seg1[seg1 == 4] = 3

    save_npy_atomic(vol_out, vol1)
    save_npy_atomic(seg_out, seg1)
    return case_id, 'processed', vol1.shape


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument(
        '--workers',
        type=int,
        default=os.cpu_count() or 1,
        help='Number of parallel workers. Defaults to available CPU cores.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    src_path = args.input_path
    tar_path = args.output_path
    workers = max(1, args.workers)
    #src_path = '/work/grana_neuro/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    #tar_path = '/work/grana_neuro/missing_modalities/BRATS2023_Training_mmFormer_npy'

    name_list = sorted(
        file_name for file_name in os.listdir(src_path)
        if os.path.isdir(os.path.join(src_path, file_name))
    )
    if not os.path.exists(os.path.join(tar_path, 'vol')):
        os.makedirs(os.path.join(tar_path, 'vol'))

    if not os.path.exists(os.path.join(tar_path, 'seg')):
        os.makedirs(os.path.join(tar_path, 'seg'))

    total_cases = len(name_list)
    processed_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_case, src_path, tar_path, file_name): file_name
            for file_name in name_list
        }
        for index, future in enumerate(as_completed(futures), start=1):
            file_name = futures[future]
            try:
                case_id, status, shape = future.result()
            except Exception as exc:
                print(f'[{index}/{total_cases}] {file_name} failed: {exc}')
                raise

            if status == 'skipped':
                skipped_count += 1
                print(f'[{index}/{total_cases}] {case_id} skipped')
                continue

            processed_count += 1
            print(f'[{index}/{total_cases}] {case_id} processed {shape}')

    print(
        f'Finished preprocessing {total_cases} cases with {workers} workers: '
        f'{processed_count} processed, {skipped_count} skipped.'
    )
