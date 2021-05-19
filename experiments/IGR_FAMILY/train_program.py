import torch
from collections import defaultdict
from tqdm import tqdm
import trimesh
from scipy.spatial import cKDTree

from layers.loss.loss_igr import loss_igr, gradient
from mesh_tools.mesh_operator import convert_sdf_samples_to_ply

def loss_ic(
            mf_input,
            mf_pred,
            nmf_input,
            nmf_pred,
            latent,
            normals,
            loss_net=loss_igr(grad_lambda=1e-1,
                              normals_lambda=1.0,
                              latent_lambda=1e-3)):
    loss_res = loss_net(
        mnfld_pnts=mf_input,
        mnfld_pred=mf_pred,
        nonmnfld_pnts=nmf_input,
        nonmnfld_pred=nmf_pred,
        latent=latent,
        normals=normals)
    loss_total = loss_res["loss"]

    return loss_total, {
                        'sdf_term': loss_res['sdf_term'],
                        'grad_term': loss_res['grad_term'],
                        'normals_term': loss_res['normals_term'],
                        'latent_term': loss_res['latent_term'],
                        }


def compose_input(pnts, lat_vecs, indices=None):
    batch_size = pnts.shape[0]
    num_samp_per_scene = pnts.shape[1]
    xyz = pnts
    # xyz = pnts.reshape(-1, 3)

    if indices is None:
        batch_vecs = lat_vecs.repeat(1, num_samp_per_scene, 1)
    else:
        idx_select = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)
        # batch_vecs = lat_vecs(idx_select)
        batch_vecs = torch.index_select(lat_vecs, 0, idx_select).reshape(batch_size, num_samp_per_scene, -1)
    input = torch.cat([batch_vecs, xyz], dim=-1)
    return input

def train(data_loader, net, optimizer, lat_vecs):
    net.train()
    loss = {'total': 0.0}
    metric = defaultdict(float)

    for data, indices in tqdm(data_loader):
        optimizer.zero_grad()

        mf_pnts = data['manifold_samples'][:, :, :3].cuda()
        mf_normals = data['manifold_samples'][:, :, 3:-1].cuda()
        nmf_pnts = data['nomanifold_samples'][:, :, :3].cuda()
        indices = indices.cuda()

        # compose input
        mf_input = compose_input(pnts=mf_pnts, lat_vecs=lat_vecs, indices=indices)
        nmf_input = compose_input(pnts=nmf_pnts, lat_vecs=lat_vecs, indices=indices)

        # forward pass
        mf_input.requires_grad_()
        nmf_input.requires_grad_()

        mf_pred = net(mf_input)
        nmf_pred = net(nmf_input)

        loss_total, loss_dict = loss_ic(
            mf_input=mf_input,
            mf_pred=mf_pred,
            nmf_input=nmf_input,
            nmf_pred=nmf_pred,
            latent=lat_vecs,
            normals=mf_normals)
        loss_total.backward()
        optimizer.step()

        loss['total'] += loss_total.item()
        for k, v in loss_dict.items():
            metric[k] += v

    return loss, metric, len(data_loader)


def get_latent_code(decoder,
                    data,
                    latent_size,
                    num_iterations,
                    base_lr=5e-4):
    def local_adjust_lr(init_lr, optimizer, num_iter, dfactor, iter_interval):
        lr = init_lr * ((1.0/dfactor) ** (num_iter // iter_interval))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    dfactor = 10
    iter_interval = int(num_iterations / 2)

    latent_vec = torch.ones(1, 1, latent_size).normal_(mean=0, std=1.0/latent_size).cuda()
    latent_vec.requires_grad = True
    optimizer = torch.optim.Adam([latent_vec], lr=base_lr)

    decoder.eval()
    for i in range(num_iterations):
        optimizer.zero_grad()
        local_adjust_lr(base_lr, optimizer, i, dfactor, iter_interval)

        mf_pnts = data['manifold_samples'][:, :, :3].cuda()
        mf_normals = data['manifold_samples'][:, :, 3:-1].cuda()
        nmf_pnts = data['nomanifold_samples'][:, :, :3].cuda()

        # compose input
        mf_input = compose_input(pnts=mf_pnts, lat_vecs=latent_vec)
        nmf_input = compose_input(pnts=nmf_pnts, lat_vecs=latent_vec)

        # forward pass
        mf_input.requires_grad_()
        nmf_input.requires_grad_()

        mf_pred = decoder(mf_input)
        nmf_pred = decoder(nmf_input)

        loss_total, loss_dict = loss_ic(
            mf_input=mf_input,
            mf_pred=mf_pred,
            nmf_input=nmf_input,
            nmf_pred=nmf_pred,
            latent=latent_vec,
            normals=mf_normals)
        loss_total.backward()
        optimizer.step()

    return latent_vec

def sdf_create_mesh(latent_vec,
                    net,
                    N=256,
                    max_batch=32 ** 3,
                    threshold_value = 0.0,
                    filename=None, offset=None, scale=None):
    net.eval()
    latent_vec = latent_vec.squeeze(0).cuda()

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
        latent_repeat = latent_vec.repeat(local_samples, 1)
        inputs = torch.cat([latent_repeat, sample_subset], -1)
        sdf_pred = net(inputs)

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
                latent_size,
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


    for data, idx in tqdm(data_loader):
        if (not idx_select is None) and (idx not in idx_select):
            continue
        mesh_file = mesh_dir / '{}.ply'.format(rec_files[idx][1:])
        if mesh_file.is_file():
            continue
        else:
            mesh_file.parent.mkdir(parents=True, exist_ok=True)

        latent_vec = get_latent_code(decoder=net,
                                     data=data,
                                     latent_size=latent_size,
                                     num_iterations=800,
                                     base_lr=1.0e-2)

        scale_max = data['norm']['scale_max']
        scale_min = data['norm']['scale_min']

        with torch.no_grad():
            sdf_create_mesh(latent_vec=latent_vec,
                            net=net,
                            filename=mesh_file,
                            N=rec_res,
                            max_batch=int(2 ** 18),
                            scale=2.0/(scale_max-scale_min),
                            offset=-(0.5*(scale_max-scale_min)+scale_min),
                            threshold_value=0.0
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