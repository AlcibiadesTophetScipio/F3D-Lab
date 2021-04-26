from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
import trimesh
from pathlib import Path

from utils.metrics import *
from workspace import get_spec_with_default
from mesh_tools.mesh_operator import convert_sdf_samples_to_ply

__package__ = 'experiments.SDF_FAMILY'
from .data_process import unpack_sdf_samples_from_ram

def train(data_loader, net, optimizer, lat_vecs, exper_specs, epoch):
    net.train()
    loss = {'total': 0.0,
            'chunk':0.0,
            'reg':0.0,}
    metric = defaultdict(float)


    enforce_minmax = True
    clamp_dist = get_spec_with_default(exper_specs, "ClampingDistance", 0.1)
    do_code_regularization = get_spec_with_default(exper_specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(exper_specs, "CodeRegularizationLambda", 1e-4)
    grad_clip = get_spec_with_default(exper_specs, "GradientClipNorm", None)

    minT = -clamp_dist
    maxT = clamp_dist
    for data, indices in data_loader:
        optimizer.zero_grad()

        # Process the input data
        num_samp_per_scene = data['sdf'].shape[1]
        sdf_data = data['sdf'].reshape(-1, 4)
        num_sdf_samples = sdf_data.shape[0]
        sdf_data.requires_grad = False

        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)
        if enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, minT, maxT)

        # compose input
        idx_select = indices.unsqueeze(-1).repeat(1,num_samp_per_scene).view(-1)
        batch_vecs = lat_vecs(idx_select)
        input = torch.cat([batch_vecs, xyz], dim=1).cuda()

        pred_sdf = net(input)
        if enforce_minmax:
            pred_sdf = torch.clamp(pred_sdf, minT, maxT)
        chunk_loss = F.l1_loss(pred_sdf,
                               sdf_gt.cuda(),
                               reduction='sum') / num_sdf_samples
        loss['chunk'] += chunk_loss.item()
        if do_code_regularization:
            l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
            reg_loss = (
                               code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                       ) / num_sdf_samples
            loss['reg'] += reg_loss.item()
            total_loss = chunk_loss + reg_loss.cuda()
        else:
            total_loss = chunk_loss

        total_loss.backward()
        loss['total'] += total_loss

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

    return loss, metric, len(data_loader)

def calculate_trimesh_metric(gt_mesh, pred_mesh, num_mesh_samples=30000):
    pred_points_sampled = trimesh.sample.sample_surface(pred_mesh, num_mesh_samples)[0]
    gt_points_np = gt_mesh.vertices

    pred_points = torch.tensor(pred_points_sampled).unsqueeze(0).cuda()
    gt_points = torch.tensor(gt_points_np).unsqueeze(0).cuda()

    # Since there is no one-to-one point mapping relation
    return {'chamfer': chamferDis_bpm(gt_points, pred_points),}

def eval(files, load_epoch=0, t_writer=None):
    rec_files = files['rec']
    gt_files = files['gt']
    gt_sample_files = files['gt_sample']

    metric = defaultdict(float)

    tensor_show_num = len(rec_files) * 0.1 if len(rec_files) * 0.1 < 20 else 20
    tensor_show_list = torch.round(torch.rand(tensor_show_num) * len(rec_files))
    iter_num = 0
    for idx in tqdm(range(len(rec_files))):
        if not Path(rec_files[idx]).is_file():
            continue

        iter_num+=1
        rec_mesh = trimesh.load(rec_files[idx])
        gt_mesh = trimesh.load(gt_files[idx])
        gt_sample_mesh = trimesh.load(gt_sample_files[idx])

        gt_gts = calculate_trimesh_metric(gt_mesh=gt_mesh, pred_mesh=gt_sample_mesh)
        gt_rec = calculate_trimesh_metric(gt_mesh=gt_mesh, pred_mesh=rec_mesh)
        gts_rec = calculate_trimesh_metric(gt_mesh=gt_sample_mesh, pred_mesh=rec_mesh)

        for k,v in gt_gts.items():
            metric[f'{k}_gt2gts'] = v
        for k,v in gt_rec.items():
            metric[f'{k}_gt2rec'] = v
        for k,v in gts_rec.items():
            metric[f'{k}_gts2rec'] = v

        if t_writer and idx in tensor_show_list:
            name = '_'.join(rec_files[idx].parts[-3:])
            tag = '{}/rec'.format(name)
            vertices = torch.tensor(rec_mesh.vertices).unsqueeze(0)
            faces = torch.tensor(rec_mesh.faces).unsqueeze(0)
            t_writer.add_mesh(tag=tag,
                              global_step=load_epoch,
                              vertices=vertices,
                              faces=faces)

            tag = '{}/gt'.format(name)
            vertices = torch.tensor(gt_mesh.vertices).unsqueeze(0)
            faces = torch.tensor(gt_mesh.faces).unsqueeze(0)
            t_writer.add_mesh(tag=tag,
                              global_step=load_epoch,
                              vertices=vertices,
                              faces=faces)

            vertices = torch.tensor(gt_sample_mesh.vertices).unsqueeze(0)
            faces = torch.tensor(gt_sample_mesh.faces).unsqueeze(0)
            tag = '{}/gts'.format(name)
            t_writer.add_mesh(tag=tag,
                              global_step=load_epoch,
                              vertices=vertices,
                              faces=faces)

    return metric, {}, iter_num

def sdf_create_mesh(decoder, latent_vec, filename,
                    N=256, max_batch=32 ** 3, offset=None, scale=None):
    decoder.eval()

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
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].cuda()

        local_samples = sample_subset.shape[0]
        latent_repeat = latent_vec.expand(local_samples, -1)
        inputs = torch.cat([latent_repeat, sample_subset], 1)
        sdf_pred = decoder(inputs)

        samples[head: min(head + max_batch, num_samples), 3] = (
            sdf_pred.squeeze(1).detach().cpu()
        )
        head += max_batch

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

def get_latent_code(decoder,
                    data_sdf,
                    latent_size,
                    num_iterations,
                    clamp_dist=0.1,
                    base_lr=5e-4,
                    num_samples=30000):
    def local_adjust_lr(init_lr, optimizer, num_iter, dfactor, iter_interval):
        lr = init_lr * ((1.0/dfactor) ** (num_iter // iter_interval))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    dfactor = 10
    iter_interval = int(num_iterations / 2)
    # latent_vec = torch.normal(mean=0.0, std=0.01, size=[1, latent_size]).cuda()
    latent_vec = torch.ones(1, latent_size).normal_(mean=0, std=0.01).cuda()
    latent_vec.requires_grad = True
    optimizer = torch.optim.Adam([latent_vec], lr=base_lr)

    for i in range(num_iterations):
        optimizer.zero_grad()
        local_adjust_lr(base_lr, optimizer, i, dfactor, iter_interval)
        decoder.eval()

        sdf_data = unpack_sdf_samples_from_ram(data_sdf, num_samples).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        latent_inputs = latent_vec.expand(num_samples, -1)
        inputs = torch.cat([latent_inputs, xyz], 1).cuda()
        pred_sdf = decoder(inputs)
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = F.l1_loss(pred_sdf, sdf_gt)
        loss += 1e-4 * torch.mean(latent_vec.pow(2))
        loss.backward()
        optimizer.step()

        loss_num = loss.item()
        # if i % 50 == 0:
        #     print(loss_num)

    return latent_vec, loss_num

def reconstruct(data_loader, rec_files, net, ws, latent_size):
    mesh_dir = ws.get_dir('reconstruction_meshes_dir')
    latent_dir = ws.get_dir('rec_latent_code_dir')

    for data, idx in tqdm(data_loader):
        # mesh_file = mesh_dir/'{}.ply'.format(rec_files[idx][1:].replace('/','-'))
        # latent_vec_file = latent_dir/'{}.pth'.format(rec_files[idx][1:].replace('/','-'))
        mesh_file = mesh_dir / '{}.ply'.format(rec_files[idx][1:])
        latent_vec_file = latent_dir / '{}.pth'.format(rec_files[idx][1:])
        if mesh_file.is_file() and latent_vec_file.is_file():
            continue
        else:
            mesh_file.parent.mkdir(parents=True, exist_ok=True)
            latent_vec_file.parent.mkdir(parents=True, exist_ok=True)

        data_sdf = data['sdf']
        data_sdf[0] = data_sdf[0].reshape(-1,4)
        data_sdf[1] = data_sdf[1].reshape(-1,4)
        data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
        data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

        latent_vec, error = get_latent_code(decoder=net,
                                            data_sdf=data_sdf,
                                            latent_size=latent_size,
                                            num_iterations=800,
                                            num_samples=8000,
                                            base_lr=5e-3)
        torch.save(latent_vec.unsqueeze(0), latent_vec_file)

        latent_vec = torch.load(latent_vec_file).squeeze(0)
        with torch.no_grad():
            sdf_create_mesh(decoder=net,
                            latent_vec=latent_vec,
                            filename=mesh_file,
                            N=256,
                            max_batch=int(2 ** 18),
                            offset=data['norm']['offset'],
                            scale=data['norm']['scale'])

        #print('Stop!')

