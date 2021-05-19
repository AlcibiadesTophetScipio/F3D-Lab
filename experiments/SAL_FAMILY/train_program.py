import torch
from collections import defaultdict
from tqdm import tqdm
import trimesh
from scipy.spatial import cKDTree

from layers.loss.loss_sal import loss_sal
from mesh_tools.mesh_operator import convert_sdf_samples_to_ply


def loss_ic(mf_pred,
            nmf_pred,
            nmf_dist,
            latent_reg,
            loss_net=loss_sal(latent_lambda=1.0e-3)):
    loss_res = loss_net(nonmanifold_pnts_pred=nmf_pred,
                nonmanifold_gt=nmf_dist,
                latent_reg=latent_reg)
    loss_total = loss_res["loss"]

    if mf_pred is None:
        return loss_total, {
                            'dist_term': loss_res['dist_term'],
                            'latent_term': loss_res['latent_term'],
                            }

    sdf_term = mf_pred.abs().mean()
    loss_total += sdf_term
    return loss_total, {
        'dist_term': loss_res['dist_term'],
        'latent_term': loss_res['latent_term'],
        'sdf_term': sdf_term.item(),
    }

def train(data_loader, net, optimizer):
    net.train()
    loss = {'total': 0.0}
    metric = defaultdict(float)

    for data, indices in tqdm(data_loader):
        optimizer.zero_grad()

        mf_pnts = data['manifold_samples'][:, :, :3].cuda()
        nmf_pnts = data['nomanifold_samples'][:, :, :3].cuda()
        nmf_dist = data['nomanifold_samples'][:, :, 3:-1].cuda()

        outputs = net(nmf_pnts, mf_pnts)
        loss_total, loss_dict = loss_ic(mf_pred=outputs['manifold_pnts_pred'],
                                        nmf_pred=outputs['nonmanifold_pnts_pred'],
                                        nmf_dist=nmf_dist.view(-1),
                                        latent_reg=outputs["latent_reg"])

        loss_total.backward()
        optimizer.step()

        loss['total'] += loss_total.item()
        for k, v in loss_dict.items():
            metric[k] += v

    return loss, metric, len(data_loader)


def sdf_create_mesh(mf_pnts,
                    net,
                    N=256,
                    max_batch=32 ** 3,
                    threshold_value = 0.0,
                    filename=None, offset=None, scale=None):
    net.eval()
    mf_pnts = mf_pnts.cuda()
    latent_vec, _ = net.encoder(mf_pnts)

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
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

    num_samples = N ** 3

    samples.requires_grad = False
    head = 0
    while head < num_samples:
        # Nx3
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].cuda()

        local_samples = sample_subset.shape[0]
        latent_repeat = latent_vec.expand(local_samples, -1)
        inputs = torch.cat([latent_repeat, sample_subset], 1)
        sdf_pred = net.decoder(inputs)

        samples[head: min(head + max_batch, num_samples), 3] = (
            sdf_pred.squeeze().detach().cpu()
        )
        head += max_batch

    # import pdb; pdb.set_trace()
    if threshold_value > 0.0:
        idx_select = samples[:, 3] < threshold_value
        samples[idx_select, 3] = 0.0

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)
    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        filename,
        offset,
        scale,
    )
    return

def reconstruct(data_loader,
                net,
                rec_files,
                rec_res,
                ws,
                rec_num=-1,
                load_epoch=None,
                t_writer=None):
    if load_epoch:
        mesh_dir = ws.get_dir('reconstruction_meshes_dir') / '{:04}'.format(load_epoch)
    else:
        mesh_dir = ws.get_dir('reconstruction_meshes_dir')

    net.eval()
    if rec_num > 1:
        idx_select = torch.randint(len(data_loader), (rec_num,))
    else:
        idx_select = None

    with torch.no_grad():
        for data, idx in tqdm(data_loader):
            if (not idx_select is None) and (idx not in idx_select):
                continue
            mesh_file = mesh_dir / '{}.ply'.format(rec_files[idx][1:])
            if mesh_file.is_file():
                continue
            else:
                mesh_file.parent.mkdir(parents=True, exist_ok=True)

            scale_max = data['norm']['scale_max']
            scale_min = data['norm']['scale_min']
            sdf_create_mesh(mf_pnts=data['manifold_samples'][:, :, :3],
                            net=net,
                            filename=mesh_file,
                            N=rec_res,
                            max_batch=int(2 ** 18),
                            threshold_value=1e-3,
                            scale=2.0/(scale_max-scale_min),
                            offset=-(0.5*(scale_max-scale_min)+scale_min)
                            )
            if t_writer:
                rec_mesh = trimesh.load(mesh_file)
                name = '_'.join(rec_files[idx].split('/')[-3:])
                tag = '{}/rec'.format(name)
                vertices = torch.tensor(rec_mesh.vertices).unsqueeze(0)
                faces = torch.tensor(rec_mesh.faces).unsqueeze(0)
                t_writer.add_mesh(tag=tag,
                                  global_step=load_epoch,
                                  vertices=vertices,
                                  faces=faces)