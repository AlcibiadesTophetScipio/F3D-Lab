import torch
import trimesh


if __name__ == '__main__':
    mesh_sample_pth = '/home/syao/Program/Experiments/N3D/AE_SOLO/Data/TemplateSample/dfaust_comasample_comanet_v1.pth'
    mesh_sample = torch.load(mesh_sample_pth)

    M = mesh_sample['M']
    for i in range(len(M)):
        M[i].write_ply(f'./mesh_{i}.ply')

    print('Done')