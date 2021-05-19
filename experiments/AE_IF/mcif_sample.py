import trimesh
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
rootpath=str("/home/syao/Program/Source/New3D")
sys.path.append(rootpath)
from workspace import Handled_Data_IO
from datasets.data_process import normalize_n1p1

def mcif_preprocess(source_dir, files, ws, sample_num=150000, sigma=0.2):
    for f in tqdm(files):
        f = Path(f[1:])
        source = (source_dir/f).with_suffix(f.suffix+'.obj')
        mcif_pnts = (ws.get_dir('mcif_pnts_dir')/f).with_suffix(f.suffix+'.npy')
        mcif_manifold_samples = (ws.get_dir('mcif_mf_samples_dir')/f).with_suffix(f.suffix+'.npy')
        mcif_nomanifold_samples = (ws.get_dir('mcif_nomf_samples_dir') / f).with_suffix(f.suffix + '.npy')
        mcif_norm_params = (ws.get_dir('mcif_norm_params_dir')/f).with_suffix(f.suffix+'.npy')

        mesh = trimesh.load(source)
        # template_mesh = Mesh(v=mesh.vertices, f=mesh.faces)
        # Normalizing on original vertices
        if mcif_pnts.is_file() and mcif_norm_params.is_file():
            pnts = np.load(mcif_pnts)
            norm_params = np.load(mcif_norm_params)
        else:
            mcif_norm_params.parent.mkdir(parents=True, exist_ok=True)
            mcif_pnts.parent.mkdir(parents=True, exist_ok=True)
            pnts, norm_params = normalize_n1p1(points=mesh.vertices, eps=0.01)
            np.save(mcif_pnts, pnts)
            np.save(mcif_norm_params, norm_params)

        # construct data pointer
        pnts_tree = cKDTree(pnts)
        mesh.vertices = pnts

        if not mcif_manifold_samples.is_file():
            mcif_manifold_samples.parent.mkdir(parents=True, exist_ok=True)
            # manifold sampling
            manifold_samples = trimesh.sample.sample_surface(mesh, sample_num*2)
            manifold_normals = mesh.face_normals[manifold_samples[1]]
            manifold_samples = manifold_samples[0]

            # get idx with original points
            manifold_idxs = []
            for p_query in manifold_samples:
                p_result = pnts_tree.query(p_query)
                manifold_idxs.append(p_result[1])
            manifold_idxs = np.expand_dims(np.array(manifold_idxs), axis=-1)
            manifold_data = np.hstack([manifold_samples, manifold_normals, manifold_idxs])

            # save manifold samples
            np.save(mcif_manifold_samples, manifold_data)

        if mcif_nomanifold_samples.is_file():
            continue
        else:
            mcif_nomanifold_samples.parent.mkdir(parents=True, exist_ok=True)

        # nomanifold sampling
        base_samples = trimesh.sample.sample_surface(mesh, sample_num)[0]
        umbs_tree = cKDTree(base_samples)
        sigmas = []
        for p in np.array_split(base_samples, 100, axis=0):
            d = umbs_tree.query(p, 51)
            sigmas.append(d[0][:, -1])
        sigmas = np.concatenate(sigmas)
        sigmas_big = sigma * np.ones_like(sigmas)
        nomanifold_samples = np.concatenate([
            base_samples + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=base_samples.shape),
            base_samples + np.expand_dims(sigmas_big, -1) * np.random.normal(0.0, 1.0, size=base_samples.shape)],
            axis=0)

        # construct a triangle soup for querying distance of nomanifold samples
        triangles = []
        for tri in mesh.triangles:
            a = Point_3(tri[0][0], tri[0][1], tri[0][2])
            b = Point_3(tri[1][0], tri[1][1], tri[1][2])
            c = Point_3(tri[2][0], tri[2][1], tri[2][2])
            triangles.append(Triangle_3(a, b, c))
        tris_tree = AABB_tree_Triangle_3_soup(triangles)

        # unsigned distance of nomanifold points
        dists = []
        for np_query in nomanifold_samples:
            cgal_query = Point_3(np_query[0].astype(np.double), np_query[1].astype(np.double),
                                 np_query[2].astype(np.double))
            cp = tris_tree.closest_point(cgal_query)
            cp = np.array([cp.x(), cp.y(), cp.z()])
            dist = np.sqrt(((cp - np_query) ** 2).sum(axis=0))
            dists.append(dist)
        nomanifold_dists = np.expand_dims(np.array(dists), axis=-1)

        # get idx with original points
        nomanifold_idxs = []
        for p_query in nomanifold_samples:
            p_result = pnts_tree.query(p_query)
            nomanifold_idxs.append(p_result[1])
        nomanifold_idxs = np.expand_dims(np.array(nomanifold_idxs), axis=-1)
        nomanifold_data = np.hstack([nomanifold_samples, nomanifold_dists, nomanifold_idxs])

        # save nomanifold samples
        np.save(mcif_nomanifold_samples, nomanifold_data)

if __name__ == '__main__':
    print(Path.cwd())
    logging.basicConfig(level=logging.INFO)

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
    }
    data_io_dir_config = {
        'base_dir': base_dir,
        'sub_dir': sub_dir,
    }
    data_io = Handled_Data_IO(dir_config=data_io_dir_config,
                              split_file=data_split_file)

    dataset_dir = Path('/DATA')
    mcif_preprocess(source_dir=dataset_dir,
                   files=data_io.filenames,
                   ws=data_io.ws)