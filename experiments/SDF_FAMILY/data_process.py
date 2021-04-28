import torch
import torch.utils.data as data_utils

from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import time

from workspace import Handled_Data_IO


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]

def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples

def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    # for continue sampling
    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
            self,
            files,
            subsample: int,
            load_ram = False
    ):
        self.subsample = subsample
        self.load_ram = load_ram
        try:
            self.files = files
        except Exception as e:
            print(e)

    def __len__(self):
        return self.files['len']

    def __getitem__(self, idx):
        sdf_file = self.files['sdf_files'][idx]
        norm_params_file = self.files['norm_params_file'][idx]

        if self.load_ram:
            sdf_samples = read_sdf_samples_into_ram(sdf_file)
        else:
            sdf_samples = unpack_sdf_samples(sdf_file, self.subsample)
        norm_params_np = np.load(norm_params_file)
        norm_params_scale = torch.tensor(norm_params_np['scale'], dtype=torch.float)
        norm_params_offset = torch.tensor(norm_params_np['offset'], dtype=torch.float)

        return {'sdf': sdf_samples,
                'norm': {'scale': norm_params_scale,
                         'offset': norm_params_offset},
                }, idx

if __name__ == '__main__':
    data_split_file = '/home/syao/Program/Source/New3D/data_split_config/dfaust_regis/dfaust_50002_test.json'
    data_dir = '/home/syao/Program/Experiments/N3D/SDF_FAMILY/Data'

    base_dir = Path(data_dir)
    sub_dir = {
        'sdf_samples_dir': 'SdfSamples',
        'sdf_surface_dir': 'SurfaceSamples',
        'sdf_norm_params_dir': 'NormalizationParameters',
    }
    data_io_dir_config = {
        'base_dir': base_dir,
        'sub_dir': sub_dir,
    }
    data_io = Handled_Data_IO(dir_config=data_io_dir_config,
                              split_file=data_split_file)

    print('Set dataset...')
    data_files = {
        'len': len(data_io.filenames),
        'sdf_files': [f.with_suffix('.npz') for f in data_io.get_instance_filenames('sdf_samples_dir')],
        'norm_params_file': [f.with_suffix('.npz') for f in data_io.get_instance_filenames('sdf_norm_params_dir')],
    }
    sm_dataset = SDFSamples(files=data_files,
                            subsample=16384)
    sm_loader = data_utils.DataLoader(
        sm_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        drop_last=True,
    )

    print('Iterate dataset...')
    torch.cuda.set_device('cuda:0')
    start_time = time.time()
    for data, indices in tqdm(sm_loader):
        sdf = data['sdf']
        norm = data['norm']
        import pdb; pdb.set_trace()
        pass

    end_time = time.time()
    print("{:.2f}".format(end_time - start_time))