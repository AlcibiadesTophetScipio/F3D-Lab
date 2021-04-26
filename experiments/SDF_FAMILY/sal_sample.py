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

def sal_preprocess(source_dir, files, ws, sample_num=250000, sigma=0.2):
    for f in tqdm(files):
        f = Path(f[1:])
        source = (source_dir/f).with_suffix(f.suffix+'.ply')
        sal_pnts = (ws.get_dir('sal_pnts_dir')/f).with_suffix(f.suffix+'.npy')
        sal_dist_tringles = (ws.get_dir('sal_tringles_dir')/f).with_suffix(f.suffix+'.npy')
        sal_norm = (ws.get_dir('sal_norm_params_dir')/f).with_suffix(f.suffix+'.npy')

        if sal_pnts.is_file() and \
            sal_dist_tringles.is_file() and \
            sal_norm.is_file():
            continue

        sal_pnts.parent.mkdir(parents=True,exist_ok=True)
        sal_dist_tringles.parent.mkdir(parents=True, exist_ok=True)
        sal_norm.parent.mkdir(parents=True, exist_ok=True)

        mesh = trimesh.load(source)
        sample = trimesh.sample.sample_surface(mesh, sample_num)
        center = np.mean(sample[0], axis=0)

        pnts = sample[0] - np.expand_dims(center, axis=0)
        scale = 1
        pnts = pnts / scale
        np.save(sal_pnts, pnts)

        triangles = []
        for tri in mesh.triangles:
            a = Point_3((tri[0][0] - center[0]) / scale, (tri[0][1] - center[1]) / scale,
                        (tri[0][2] - center[2]) / scale)
            b = Point_3((tri[1][0] - center[0]) / scale, (tri[1][1] - center[1]) / scale,
                        (tri[1][2] - center[2]) / scale)
            c = Point_3((tri[2][0] - center[0]) / scale, (tri[2][1] - center[1]) / scale,
                        (tri[2][2] - center[2]) / scale)
            triangles.append(Triangle_3(a, b, c))
        tree = AABB_tree_Triangle_3_soup(triangles)

        sigmas = []
        ptree = cKDTree(pnts)
        i = 0
        for p in np.array_split(pnts, 100, axis=0):
            d = ptree.query(p, 51)
            sigmas.append(d[0][:, -1])

            i = i + 1

        sigmas = np.concatenate(sigmas)
        sigmas_big = sigma * np.ones_like(sigmas)

        sample = np.concatenate([pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                 pnts + np.expand_dims(sigmas_big, -1) * np.random.normal(0.0, 1.0, size=pnts.shape)],
                                axis=0)

        dists = []
        for np_query in sample:
            cgal_query = Point_3(np_query[0].astype(np.double), np_query[1].astype(np.double),
                                 np_query[2].astype(np.double))

            cp = tree.closest_point(cgal_query)
            cp = np.array([cp.x(), cp.y(), cp.z()])
            dist = np.sqrt(((cp - np_query) ** 2).sum(axis=0))

            dists.append(dist)
        dists = np.array(dists)
        np.save(sal_dist_tringles,
                np.concatenate([sample, np.expand_dims(dists, axis=-1)], axis=-1))

        np.save(sal_norm,
                {"center": center, "scale": scale})


if __name__ == '__main__':
    print(Path.cwd())
    logging.basicConfig(level=logging.INFO)

    train_data = True
    if train_data:
        data_split_file = '/home/syao/Program/Source/New3D/data_split_config/dfaust_scans/dfaust_50002_train.json'
    else:
        data_split_file = '/home/syao/Program/Source/New3D/data_split_config/dfaust_scans/dfaust_50002_test.json'

    data_dir = '/home/syao/Program/Experiments/N3D/SDF_FAMILY/Data'
    base_dir = Path(data_dir)
    sub_dir = {
        'sal_pnts_dir': 'SalPnts',
        'sal_tringles_dir': 'SalTringles',
        'sal_norm_params_dir': 'NormalizationParameters',
    }
    data_io_dir_config = {
        'base_dir': base_dir,
        'sub_dir': sub_dir,
    }
    data_io = Handled_Data_IO(dir_config=data_io_dir_config,
                              split_file=data_split_file)

    dataset_dir = Path('/DATA')
    sal_preprocess(source_dir=dataset_dir,
                   files=data_io.filenames,
                   ws=data_io.ws)