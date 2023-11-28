from pathlib import Path
import pickle

import h5py
from tqdm import tqdm


def save_h5s(data_dicts, offset: int=1):
    files = []
    for i, data_dict in tqdm(
        enumerate(data_dicts.values()),
        total=len(data_dicts),
    ):
        out_file = out_dir / f'{offset + i}_roi.h5'
        files.append(out_file)

        image = data_dict['intensities'].astype(float)
        image[image < 500] = 500
        image[image > 2500] = 2500
        image = (image - 500)/(2500 - 500)
            
        labels = data_dict['labels'] > 0

        with h5py.File(out_file, 'w') as f:
            f.create_dataset('image', data=image)
            f.create_dataset('label', data=labels)

    return files


if __name__ == '__main__':
    root = Path('/home/mkaailab/Documents/CBCT/toothseg/data/train/')
    out_dir = root.parent / 'train_cui'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(root / '630771885053147113.pkl', 'rb') as f:
        train_dicts = pickle.load(f)

    with open(root / '2072182520573095653.pkl', 'rb') as f:
        val_dicts = pickle.load(f)

    train_files = save_h5s(train_dicts)
    val_files = save_h5s(val_dicts, offset=len(train_files) + 1)

    with open(out_dir / 'train_file.list', 'w') as f:
        for file in train_files:
            f.write(file.as_posix() + '\n')

    with open(out_dir / 'test_file.list', 'w') as f:
        for file in val_files:
            f.write(file.as_posix() + '\n')
