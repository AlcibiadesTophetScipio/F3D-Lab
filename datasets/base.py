import numpy as np
import torch
from torch.utils.data import Dataset
import torch.utils.data as data_utils
from pathlib import Path
from psbody.mesh import Mesh
import pytorch3d

from mesh_tools.mesh_operator import get_vert_connectivity

def parser_3d_file(file, with_edge=False):
    file_suffix = Path(file).suffix
    if file_suffix == '.obj' or file_suffix == '.ply':
        mesh = Mesh(filename=str(file))
        mesh_verts = torch.Tensor(mesh.v)
        if not with_edge:
            return {'points': mesh_verts}
        else:
            adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
            edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
            return {'points': mesh_verts,
                    'edges': edge_index.type(torch.long)}
    else:
        return {}

class N3D_Data(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return self.files['len']

    def __getitem__(self, idx):
        instance_file = self.files['points'][idx]
        instance = parser_3d_file(instance_file)

        return {**instance,}, idx

def dataset_normalize(d_dataset, norm_type=0):
    data_loader = data_utils.DataLoader(
        d_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8
    )
    if norm_type==0:
        # across dataset for one point, return [3]
        d_data = [d['points'].view(-1, 3) for d, i in data_loader]
        d_data = torch.cat(d_data, dim=0)
        d_mean = d_data.mean(dim=0)
        d_std = d_data.std(dim=0)
        return {'mean':d_mean, 'std':d_std}
    elif norm_type==1:
        # across dataset for one instance, return [N, 3]
        d_data = [d['points'] for d, i in data_loader]
        d_data = torch.cat(d_data, dim=0)
        d_mean = d_data.mean(dim=0)
        d_std = d_data.std(dim=0)
        return {'mean': d_mean, 'std': d_std}
    elif norm_type==2:
        # instance norm independently, return [* ,N, 3]
        d_static = [[d['points'].mean(dim=1), d['points'].std(dim=1)]
                    for d,i in data_loader]
        d_mean = [m[0] for m in d_static]
        d_mean = torch.cat(d_mean, dim=0)
        d_std = [m[1] for m in d_static]
        d_std = torch.cat(d_std, dim=0)

        return {'mean': d_mean,
                'std': d_std}

if __name__ == '__main__':

    print('Done!')