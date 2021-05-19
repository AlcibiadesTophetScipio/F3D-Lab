import json
import torch
from pathlib import Path
import torch.utils.data as data_utils

from datasets.base import N3D_Data, dataset_normalize
from workspace import Handled_Data_IO

def get_dataset(dataset_dir,
                train_split_file,
                test_split_file):
    dataset_io = Handled_Data_IO(dir_config={'base_dir': dataset_dir, 'sub_dir': {}},
                                 split_file=train_split_file)
    data_files = {
        'len': len(dataset_io.filenames),
        'points': [f.with_suffix('.obj') for f in dataset_io.get_instance_filenames(dir_name=None)],
    }
    train_dataset = N3D_Data(files=data_files)

    dataset_io = Handled_Data_IO(dir_config={'base_dir': dataset_dir, 'sub_dir': {}},
                                 split_file=test_split_file)
    data_files = {
        'len': len(dataset_io.filenames),
        'points': [f.with_suffix('.obj') for f in dataset_io.get_instance_filenames(dir_name=None)],
    }
    test_dataset = N3D_Data(files=data_files)

    return train_dataset, test_dataset

if __name__ == '__main__':
    split_file = '/home/syao/Program/Source/New3D/data_split_config/dfaust_regis/dfaust_50002_train.json'
    experiment_dir = '/home/syao/Program/Datasets'

    base_dir = Path(experiment_dir)
    data_io_dir_config = {
        'base_dir': base_dir,
        'sub_dir': {},
    }
    data_io = Handled_Data_IO(dir_config=data_io_dir_config,
                              split_file=split_file)

    data_files = {
        'len': len(data_io.filenames),
        'points': [f.with_suffix('.obj') for f in data_io.get_instance_filenames(dir_name=None)],
                  }
    df_dataset = N3D_Data(files=data_files)
    print(df_dataset)
    # df_norm_1 = dataset_normalize(d_dataset=df_dataset, norm_type=0)
    # df_norm_2 = dataset_normalize(d_dataset=df_dataset, norm_type=1)
    # df_norm_3 = dataset_normalize(d_dataset=df_dataset, norm_type=2)

    df_loader = data_utils.DataLoader(
        df_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )

    iter_count = 0
    for data, indices in df_loader:
        iter_count += 1
        points = data['points']
        print(points.size())

    print(iter_count)
    print(len(df_loader))
    print('Done!')