import torch
import os.path as osp
from psbody.mesh import Mesh
import numpy as np
import scipy.sparse as sp

from mesh_tools.mesh_operator import get_vert_connectivity, \
                                vertex_quadrics, \
                                _get_sparse_transform, \
                                scipy_to_torch_sparse

def collapse_cost(Qv, v):
    '''
    Quadric loss
    :param Qv:
    :param v:
    :return:
    '''
    p = np.vstack((v.reshape(-1,1), np.array([1]).reshape(-1,1)))
    destroy_cost = p.T.dot(Qv).dot(p)

    return destroy_cost

def neighbour_distance(v, edge, idx):
    '''
    neighbour distance
    :param v: All nodes featrue
    :param edge: All edges connection [row, col]
    :param idx: self index
    :return: dict{idx:distance}
    '''
    x = v[idx]
    neighbour = edge.row[edge.col == idx]

    result = {}
    for i in neighbour:
        n = v[i]
        tmp = np.linalg.norm((x-n), ord=2)
        result[i] = tmp

    return sorted(result.items(), key=lambda x:x[1])

class Node_Str:
    '''
    mark: for iterate dropping
    dir: -1, means self connection
         -2, means keep nodes
         others, means drop self and merge to the other
    '''
    def __init__(self, g_idx, cost, neighbour):
        self.g_idx = g_idx
        self.cost = cost
        self.neighbour = neighbour
        self.mark = False
        self.dir = -1

def make_edges(idx_x, list_v, t_mesh_v):
    x = t_mesh_v[idx_x]
    d = {}
    for i in list_v:
        m = t_mesh_v[i]
        d[i]= np.linalg.norm((x-m), ord=2)
    neighbours = sorted(d.items(), key=lambda x: x[1])
    edges = [idx_x]
    for (k, v) in neighbours:
        if k != idx_x:
            edges.append(k)
        if len(edges)>=3:
            break

    return np.array(edges)

def simplify_mesh(template_mesh):
    v_quadrics = vertex_quadrics(template_mesh)

    vert_adj = sp.coo_matrix(get_vert_connectivity(template_mesh.v, template_mesh.f))
    # vert_adj = pd.DataFrame({'row': vert_adj.row, 'col': vert_adj.col})

    nodes_list = {}
    for i in range(len(template_mesh.v)):
        score = collapse_cost(v_quadrics[i], template_mesh.v[i])
        d_neighbour = neighbour_distance(template_mesh.v, vert_adj, i)
        nodes_list[i] = (Node_Str(g_idx=i, cost=score[0, 0], neighbour=d_neighbour))
        # print(score)
    nodes_relist = sorted(nodes_list.items(), key=lambda x: x[1].cost)

    for (k, v) in nodes_relist:
        if nodes_list[v.g_idx].mark == False:
            nodes_list[v.g_idx].mark = True
            for (j, l) in v.neighbour:
                if nodes_list[j].mark == False:
                    nodes_list[v.g_idx].dir = j
                    nodes_list[j].mark = True
                    nodes_list[j].dir = -2
                    break

    faces = template_mesh.f.copy()
    source = []
    target = []
    self_dir = []
    for (k, v) in nodes_list.items():
        if v.dir >= 0:
            source.append(v.g_idx)
            target.append(v.dir)
            np.place(faces, faces==v.g_idx, v.dir)
        elif v.dir == -1:
            self_dir.append(v.g_idx)

    # test
    # l1 = set(self_dir)
    # l2 = set(source)
    # l3 = set(target)
    # print(l1 & l2)
    # print(l1 & l3)
    # print(l2 & l3)
    # print(len(self_dir))

    a = faces[:, 0] == faces[:, 1]
    b = faces[:, 1] == faces[:, 2]
    c = faces[:, 2] == faces[:, 0]

    # remove degenerate faces
    def logical_or3(x, y, z):
        return np.logical_or(x, np.logical_or(y, z))

    faces_to_keep = np.logical_not(logical_or3(a, b, c))
    faces = faces[faces_to_keep, :].copy()

    # BUG fixed at 2020-10-20 13:21
    # Dropped nodes maybe destroy the triangle mesh, thus destroy the edges connection,
    # lead to mismatch the nodes number
    face_list = []
    for v in target:
        if v not in faces:
            new_edges = make_edges(v, target, template_mesh.v)
            face_list.append(new_edges)
            # print('merged!', v)

    for v in self_dir:
        if v not in faces:
            new_edges = make_edges(v, target, template_mesh.v)
            face_list.append(new_edges)
            # print('self!', v)

    if len(face_list) > 0:
        faces = np.vstack([faces, face_list])

    new_faces, mtx = _get_sparse_transform(faces, len(template_mesh.v))

    msg = {'merged':target, 'removed':source, 'self':self_dir}
    return (new_faces, mtx, msg)

def recalcul_v(v, inform):
    _v = v.copy()
    idx_merged = inform['merged']
    idx_removed = inform['removed']
    v[idx_merged] = (v[idx_merged] + v[idx_removed]) / 2
    return v

def step_calc(template_mesh):
    ds_f, ds_D, u_msg = simplify_mesh(template_mesh)
    _v = recalcul_v(template_mesh.v, u_msg)
    new_mesh_v = ds_D.dot(_v)
    new_mesh = Mesh(v=new_mesh_v, f=ds_f)
    return new_mesh

def trans_inform(inform, device):
    idx_merged = inform['merged']
    idx_removed = inform['removed']
    idx_self = inform['self']

    return {'merged': torch.tensor(idx_merged, dtype=torch.long, device=device),
            'removed': torch.tensor(idx_removed, dtype=torch.long, device=device),
            'self': torch.tensor(idx_self, dtype=torch.long, device=device)}

def bt_sampling(mesh):
    M, A, D, U = [], [], [], []
    A.append(get_vert_connectivity(mesh.v, mesh.f).tocoo())
    M.append(mesh)
    for i in range(20):
        if M[-1].v.shape[0] <= 20:
            break
        else:
            ds_f, ds_D, u_msg = simplify_mesh(M[-1])
            D.append(ds_D.tocoo())
            _v = recalcul_v(M[-1].v, u_msg)
            new_mesh_v = ds_D.dot(_v)
            new_mesh = Mesh(v=new_mesh_v, f=ds_f)
            M.append(new_mesh)
            A.append(get_vert_connectivity(new_mesh.v, new_mesh.f).tocoo())
            U.append(u_msg)

    return {'M':M, 'A':A, 'D':D, 'U':U}


if __name__ == '__main__':
    template_fp = osp.expanduser('~/Research/Datasets/templates/coma_template.obj')
    template_mesh = Mesh(filename=template_fp)

    mesh = template_mesh
    print(mesh.f.shape)
    print(mesh.v.shape)
    # for i in range(10):
    #     if mesh.v.shape[0] <= 20:
    #         break
    #     mesh = step_calc(mesh)
    #     print('##########')
    #     print(i + 1)
    #
    #     # mesh.write_ply(DATADIR + 'temp/{:d}.ply'.format(i+1))
    #     print(mesh.f.shape)
    #     print(mesh.v.shape)

    ms = bt_sampling(template_mesh)
    device = torch.device('cuda:0')
    D_t = [scipy_to_torch_sparse(d).to(device) for d in ms['D']]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in ms['A']]
    U_t = [trans_inform(u, device=device) for u in ms['U']]
    M = ms['M']
    num_nodes = [len(M[i].v) for i in range(len(M))]

    print('Done!')