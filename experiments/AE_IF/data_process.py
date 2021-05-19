import torch
import torch.utils.data as data_utils

import numpy as np

def unpack_samples(filename, subsample, dist_eps=None):
    npz = np.load(filename)
    data_tensor = torch.tensor(npz, dtype=torch.float)

    random_idx = (torch.rand(int(subsample)) * data_tensor.shape[0]).long()
    samples = torch.index_select(data_tensor,
                                 dim=0,
                                 index=random_idx)
    if dist_eps:
        dist_mask = samples[:, 3:-1] > dist_eps
        samples = samples[dist_mask.view(-1)]
        while len(samples) < subsample:
            random_idx = (torch.rand(int(subsample)) * data_tensor.shape[0]).long()
            samples_tmp = torch.index_select(data_tensor,
                                            dim=0,
                                            index=random_idx)
            dist_mask = samples_tmp[:, 3:-1] > dist_eps
            samples_tmp = samples_tmp[dist_mask.view(-1)]
            samples = torch.cat([samples, samples_tmp], dim=0)
        samples = samples[0:subsample]

    return samples

class MCIFSamples(torch.utils.data.Dataset):
    def __init__(
            self,
            files,
            manifold_subsample: int,
            nomanifold_subsample: int,
            voxel_idx=False
    ):
        self.manifold_subsample = manifold_subsample
        self.nomanifold_subsample = nomanifold_subsample
        self.voxel_idx = voxel_idx
        try:
            self.files = files
        except Exception as e:
            print(e)

    def __len__(self):
        return self.files['len']

    def __getitem__(self, idx):
        pnts_file = self.files['pnts_files'][idx]
        norm_params_file = self.files['norm_params_files'][idx]

        manifold_file = self.files['manifold_files'][idx]
        nomanifold_file = self.files['nomanifold_files'][idx]

        pnts = torch.tensor(np.load(pnts_file), dtype=torch.float)
        manifold_samples = unpack_samples(manifold_file, self.manifold_subsample)
        nomanifold_samples = unpack_samples(nomanifold_file, self.nomanifold_subsample, dist_eps=1e-6)

        norm_params_np = np.load(norm_params_file, allow_pickle=True)
        norm_params_max_scale = torch.tensor(norm_params_np.item()['scale_max'], dtype=torch.float)
        norm_params_min_scale = torch.tensor(norm_params_np.item()['scale_min'], dtype=torch.float)

        if self.voxel_idx:
            voxel_idx_file = self.files['voxel_idx_files'][idx]
            voxel_idx = torch.tensor(np.load(voxel_idx_file), dtype=torch.long)

            return {'pnts': pnts,
                    'manifold_samples': manifold_samples,
                    'nomanifold_samples': nomanifold_samples,
                    'norm': {'scale_max': norm_params_max_scale,
                             'scale_min': norm_params_min_scale},
                    'voxel_idx': voxel_idx,
                    }, idx
        else:

            return {'pnts': pnts,
                    'manifold_samples': manifold_samples,
                    'nomanifold_samples': nomanifold_samples,
                    'norm': {'scale_max': norm_params_max_scale,
                             'scale_min': norm_params_min_scale},
                    }, idx
