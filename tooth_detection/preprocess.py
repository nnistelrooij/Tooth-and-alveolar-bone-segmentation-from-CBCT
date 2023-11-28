import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import numpy as np
import pickle
import h5py
from skimage.morphology import skeletonize_3d
from tqdm import tqdm


def read_data(data_dict):
    image = data_dict['intensities'].astype(np.float32)
    image[image < 500] = 500
    image[image > 2500] = 2500
    image = (image - 500)/(2500 - 500)

    labels = data_dict['labels']

    return image, labels


def random_crop(
    image, label, offset_cnt, offset_skl,
    num_images: int=5,
    output_size: Tuple[int, int, int]=(128, 128, 128),
):
    (w, h, d) = image.shape
    image_list, label_list, offset_cnt_list, offset_skl_list = [], [], [], []
    for _ in range(num_images):
        w1 = np.random.randint(0, w - output_size[0])
        h1 = np.random.randint(0, h - output_size[1])
        d1 = np.random.randint(0, d - output_size[2])
        print('print the random coord:', w1, h1, d1)

        label_list.append(label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
        image_list.append(image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
        offset_cnt_list.append(offset_cnt[:, w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
        offset_skl_list.append(offset_skl[:, w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
        
    return image_list, label_list, offset_cnt_list, offset_skl_list


def determine_centroid_offsets(tooth_mask, offsets):
    coords = np.column_stack(np.nonzero(tooth_mask))

    centroid = coords.mean(axis=0).astype(int)
    offsets[tooth_mask] = centroid - coords


def determine_skeleton_offsets(tooth_mask, offsets):
    skeleton = skeletonize_3d(tooth_mask)

    tooth_coords = np.column_stack(np.nonzero(tooth_mask))
    skeleton_coords = np.column_stack(np.nonzero(skeleton))

    if skeleton_coords.shape[0] == 0:
        return False

    distances = np.linalg.norm(skeleton_coords[None] - tooth_coords[:, None], axis=-1)
    skeleton_idxs = distances.argmin(axis=1)

    offsets[tooth_mask] = skeleton_coords[skeleton_idxs] - tooth_coords

    return True


def save_h5(inputs):
    data_dict, offset = inputs

    out_files = []
    image, labels = read_data(data_dict)

    offset_cnt = np.zeros((*image.shape, 3), dtype=np.int8)
    offset_skl = np.zeros((*image.shape, 3), dtype=np.int8)
    for label in data_dict['unique_labels']:
        tooth_mask = labels == label
        determine_centroid_offsets(tooth_mask, offset_cnt)
        success = determine_skeleton_offsets(tooth_mask, offset_skl)

        if not success:
            labels[tooth_mask] = 0
            offset_cnt[tooth_mask] = 0
            offset_skl[tooth_mask] = 0

    offset_cnt = np.transpose(offset_cnt, (3, 0, 1, 2))
    offset_skl = np.transpose(offset_skl, (3, 0, 1, 2))

    image_list, label_list, offset_cnt_list, offset_skl_list = random_crop(
        image, labels, offset_cnt, offset_skl,
    )

    for i in range(len(image_list)):
        print('---save file:', offset)
        out_file = out_dir / f'{offset}_roi.h5'
        with h5py.File(out_file, 'w') as f:
            f.create_dataset('image', data = image_list[i])
            f.create_dataset('label', data = label_list[i].astype(int))
            f.create_dataset('cnt_offset', data = offset_cnt_list[i])
            f.create_dataset('skl_offset', data = offset_skl_list[i])

        offset += 1
        out_files.append(out_file)

    return out_files
        

if __name__ == '__main__':
    root = Path('/home/mkaailab/Documents/CBCT/toothseg/data/train/')
    out_dir = root.parent / 'train_cui_cnt_skl'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(root / '630771885053147113.pkl', 'rb') as f:
        train_dicts = pickle.load(f)

    with open(root / '2072182520573095653.pkl', 'rb') as f:
        val_dicts = pickle.load(f)

    with mp.Pool(16) as p:
        train_files = []
        iterator = zip(
            train_dicts.values(),
            range(1, 5 * len(train_dicts) + 1, 5),
        )
        for out_files in tqdm(
            p.imap_unordered(save_h5, iterator),
            total=len(train_dicts),
        ):
            train_files.extend(out_files)

    with mp.Pool(16) as p:
        val_files = []
        offset = len(train_files) + 1
        iterator = zip(
            val_dicts.values(),
            range(offset, offset + 5 * len(val_dicts), 5),
        )
        for out_files in tqdm(
            p.imap_unordered(save_h5, iterator),
            total=len(val_dicts),
        ):
            val_files.extend(out_files)

    with open(out_dir / 'train_file.list', 'w') as f:
        for file in train_files:
            f.write(file.as_posix() + '\n')

    with open(out_dir / 'test_file.list', 'w') as f:
        for file in val_files:
            f.write(file.as_posix() + '\n')
