import multiprocessing as mp
from pathlib import Path
import pickle

import h5py
import numpy as np
from scipy import ndimage
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize_3d
from tqdm import tqdm

from toothseg.models.cluster import fast_search_cluster


def read_data(data_dict):
    image = data_dict['intensities'].astype(np.float32)
    image[image < 500] = 500
    image[image > 2500] = 2500
    image = (image - 500)/(2500 - 500)

    labels = data_dict['labels']

    return image, labels


def determine_slices(
    tooth_mask,
    patch_size=(96, 96, 96),
):
    slices = ndimage.find_objects(tooth_mask)[0]

    out = ()
    for slc, patch_size, dim in zip(slices, patch_size, tooth_mask.shape):
        diff = patch_size - (slc.stop - slc.start)
        diff = diff // 2, diff // 2 + diff % 2
        slc = slice(slc.start - diff[0], slc.stop + diff[1])
        diff = dim - min(slc.start, 0) - max(dim, slc.stop)
        slc = slice(slc.start + diff, slc.stop + diff)
        out += (slc,)

    return out


def find_centroid(tooth_mask):
    centroid = np.column_stack(np.nonzero(tooth_mask)).mean(axis=0).astype(int)

    out = np.zeros_like(tooth_mask)
    out[tuple(centroid)] = True

    return out


def find_skeleton(tooth_mask):
    skeleton = skeletonize_3d(tooth_mask)

    return skeleton > 0


def find_roots_components(
    tooth_mask, label,
):   
    tooth_mask = ndimage.binary_opening(
        tooth_mask,
        ndimage.generate_binary_structure(3, 1),
        iterations=1,
    )

    centroid = np.column_stack(np.nonzero(tooth_mask)).mean(0).astype(int)

    roots_mask = tooth_mask.copy()
    if label < 30:
        roots_mask[..., centroid[2] - 12:] = False
    else:
        roots_mask[..., :centroid[2] + 12] = False

    roots_coords = np.column_stack(np.nonzero(roots_mask))
    peak_voxels, cluster_idxs = fast_search_cluster(
        roots_coords[:, :-1], density_threshold=8, distance_threshold=10,
    )

    if peak_voxels.shape[0] > 3:
        k = 3

    root_coords = np.zeros((0, 3), dtype=int)
    for label in range(peak_voxels.shape[0]):
        coords = roots_coords[cluster_idxs == label]
        dists = np.linalg.norm(coords - centroid, axis=-1)
        root_coord = coords[dists.argmax()]
        
        root_coords = np.concatenate((root_coords, [root_coord]))
    
    out = np.zeros_like(tooth_mask)
    out[tuple(root_coords.T)] = True

    return out
    


def save_h5s(inputs):
    data_dict, offset = inputs

    image, labels = read_data(data_dict)

    out_files = []
    for label in data_dict['unique_labels']:
        tooth_mask = labels == label
        components, max_label = ndimage.label(tooth_mask)
        counts = ndimage.sum_labels(np.ones_like(components), components, range(max_label + 1))
        tooth_mask[(counts < 200)[components]] = False

        if not np.any(tooth_mask):
            continue

        tooth_slices = determine_slices(tooth_mask)
        tooth_patch = tooth_mask[tooth_slices]
        if not np.any(tooth_patch):
            continue

        
        centroid = find_centroid(tooth_patch)
        skeleton = find_skeleton(tooth_patch)
        boundary = find_boundaries(tooth_patch)
        keypoints = find_roots_components(tooth_patch, label)

        out_file = out_dir / f'{offset}_{label}_instseg.h5'
        with h5py.File(out_file, 'w') as f:
            f.create_dataset('image', data = image[tooth_slices])
            f.create_dataset('label', data = labels[tooth_slices] == label)
            f.create_dataset('centroid', data = centroid)
            f.create_dataset('skeleton', data = skeleton)
            f.create_dataset('boundary', data = boundary)
            f.create_dataset('keypoints', data = keypoints)

        out_files.append(out_file)

    return out_files, data_dict['scan_file']
        

if __name__ == '__main__':
    root = Path('/home/mkaailab/Documents/CBCT/toothseg/data/train/')
    out_dir = root.parent / 'train_cui_instseg'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(root / '684044217571278600.pkl', 'rb') as f:
        train_dicts = pickle.load(f)

    with open(root / '252220999800534621.pkl', 'rb') as f:
        val_dicts = pickle.load(f)

    train_files = []
    # for train_dict, offset in tqdm(zip(
    #     train_dicts.values(), range(1, len(train_dicts) + 1),
    # ), total=len(train_dicts)):
    #     out_files, scan_file = save_h5s((train_dict, offset))
    #     train_files.extend(out_files)
    with mp.Pool(16) as p:
        iterator = zip(train_dicts.values(), range(1, len(train_dicts) + 1))
        t = tqdm(p.imap_unordered(save_h5s, iterator), total=len(train_dicts))
        for out_files, scan_file in t:
            train_files.extend(out_files)
            t.set_description(scan_file)

    with mp.Pool(16) as p:
        val_files = []
        offset = len(train_files) + 1
        iterator = zip(val_dicts.values(), range(offset, len(val_dicts) + offset + 1))
        t = tqdm(p.imap_unordered(save_h5s, iterator), total=len(val_dicts))
        for out_files, scan_file in t:
            val_files.extend(out_files)
            t.set_description(scan_file)

    with open(out_dir / 'train_file.list', 'w') as f:
        for file in train_files:
            f.write(file.as_posix() + '\n')

    with open(out_dir / 'test_file.list', 'w') as f:
        for file in val_files:
            f.write(file.as_posix() + '\n')
