from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm
from psbody.mesh import Mesh

from scipy.spatial import cKDTree
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

import sys
import torch
rootpath=str("/home/syao/Program/Source/New3D")
sys.path.append(rootpath)
from workspace import Handled_Data_IO
from datasets.data_process import normalize_n1p1

def make_voxel_idx(source_dir, files, ws, N=256):
    for f in tqdm(files):
        f = Path(f[1:])
        source = (source_dir/f).with_suffix(f.suffix+'.obj')
        mcif_pnts = (ws.get_dir('mcif_pnts_dir')/f).with_suffix(f.suffix+'.npy')
        voxel_file = (ws.get_dir('mcif_voxel_idx_dir') / f).with_suffix(f.suffix + '.npy')
        if not mcif_pnts.is_file():
            print(f"Can't find pnts file of {f}")
            continue
        if voxel_file.is_file():
            continue
        else:
            voxel_file.parent.mkdir(parents=True, exist_ok=True)

        pnts = np.load(mcif_pnts)
        pnts_tree = cKDTree(pnts)

        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        samples = samples.numpy()
        correlated_idx = []
        for idx_samples in range(len(samples)):
            sample_query = samples[idx_samples,:-1]
            p_result = pnts_tree.query(sample_query)
            correlated_idx.append(p_result[1])
        voxel_idx = np.array(correlated_idx)
        np.save(voxel_file, voxel_idx)

if __name__ == '__main__':
    print(Path.cwd())
    logging.basicConfig(level=logging.INFO)

    rec_res=256
    train_data = False
    if train_data:
        data_split_file = '/home/syao/Program/Source/New3D/data_split_config/dfaust_regis/dfaust_50002_train.json'
    else:
        data_split_file = '/home/syao/Program/Source/New3D/data_split_config/dfaust_regis/dfaust_50002_test.json'

    data_dir = '/home/syao/Program/Experiments/N3D/AE_IF/Data'
    base_dir = Path(data_dir)
    sub_dir = {
        'mcif_pnts_dir': 'McifPnts',
        'mcif_mf_samples_dir': 'McifManifoldSamples',
        'mcif_nomf_samples_dir': 'McifNomanifoldSamples',
        'mcif_norm_params_dir': 'NormalizationParameters',
        'mcif_voxel_idx_dir': 'VoxelIndex',
    }
    data_io_dir_config = {
        'base_dir': base_dir,
        'sub_dir': sub_dir,
    }
    data_io = Handled_Data_IO(dir_config=data_io_dir_config,
                              split_file=data_split_file)

    dataset_dir = Path('/DATA')
    make_voxel_idx(source_dir=dataset_dir,
                   files=data_io.filenames,
                   ws=data_io.ws,
                   N=rec_res)