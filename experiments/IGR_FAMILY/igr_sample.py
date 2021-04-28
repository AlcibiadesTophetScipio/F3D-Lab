import trimesh
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm

from scipy.spatial import cKDTree
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

import sys
rootpath=str("/home/syao/Program/Source/New3D")
sys.path.append(rootpath)
from workspace import Handled_Data_IO
from datasets.data_process import normalize_center_scale

def igr_preprocess(source_dir, files, ws, sample_num=250000, sigma=0.2):
    for f in tqdm(files):
        f = Path(f[1:])
        source = (source_dir/f).with_suffix(f.suffix+'.ply')
        igr_pnts_nves = (ws.get_dir('igr_pnts_nvecs_dir')/f).with_suffix(f.suffix+'.npy')
        igr_norm_params = (ws.get_dir('igr_norm_params_dir')/f).with_suffix(f.suffix+'.npy')

        if igr_pnts_nves.is_file() and \
            igr_norm_params.is_file():
            continue

        igr_pnts_nves.parent.mkdir(parents=True,exist_ok=True)
        igr_norm_params.parent.mkdir(parents=True, exist_ok=True)

        mesh = trimesh.load(source)
        sample = trimesh.sample.sample_surface(mesh, sample_num)

        # point cloud normalizing
        scale = 1.0
        pnts, norm_params = normalize_center_scale(points=sample[0], scale=scale)

        normals = mesh.face_normals[sample[1]]
        point_set = np.hstack([pnts, normals])

        np.save(igr_pnts_nves, point_set)
        np.save(igr_norm_params, norm_params)

if __name__ == '__main__':
    print(Path.cwd())
    logging.basicConfig(level=logging.INFO)

    train_data = True
    if train_data:
        data_split_file = '/home/syao/Program/Source/New3D/data_split_config/dfaust_scans/dfaust_50002_train.json'
    else:
        data_split_file = '/home/syao/Program/Source/New3D/data_split_config/dfaust_scans/dfaust_50002_test.json'

    data_dir = '/home/syao/Program/Experiments/N3D/IGR_FAMILY/Data'
    base_dir = Path(data_dir)
    sub_dir = {
        'igr_pnts_nvecs_dir': 'IgrPntsNvecs',
        'igr_norm_params_dir': 'NormalizationParameters',
    }
    data_io_dir_config = {
        'base_dir': base_dir,
        'sub_dir': sub_dir,
    }
    data_io = Handled_Data_IO(dir_config=data_io_dir_config,
                              split_file=data_split_file)

    dataset_dir = Path('/DATA')
    igr_preprocess(source_dir=dataset_dir,
                   files=data_io.filenames,
                   ws=data_io.ws)