import torch
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial import cKDTree

from mesh_tools.mesh_operator import convert_sdf_samples_to_ply
__package__ = 'experiments.AE_IF'
from .loss_function import mcif_loss_v1, mcif_loss_v2

def train(data_loader, net, optimizer):
    net.train()
    loss = {'total': 0.0}
    metric = defaultdict(float)

    loss_net = mcif_loss_v1()
    for data, indices in tqdm(data_loader):
        optimizer.zero_grad()

        data['pnts'] = data['pnts'].cuda()
        data['manifold_samples'] = data['manifold_samples'].cuda()
        data['nomanifold_samples'] = data['nomanifold_samples'].cuda()

        all_latent = net({'pnts': data['pnts'],})
        sdf_latent = all_latent['sdf_latent']
        mc_latent = all_latent['mc_latent']

        mf_pnts = data['manifold_samples'][:,:,:3]
        mf_pnts.requires_grad_()
        mf_idx_gt = data['manifold_samples'][:, :, -1].long()
        mf_data = {
            'sample_pnts': mf_pnts,
            # 'idx': mf_idx_gt,
        }
        mf_pred = net(mf_data, sdf_latent=sdf_latent)
        mf_sdf_pred = mf_pred['pred_sdf']


        # idx pred module
        # mf_idx_pred = mf_pred['pred_idx']
        # mf_idx_target = torch.zeros_like(mf_idx_pred).scatter_(1,mf_idx_gt.view(-1,1),1.0)
        # loss_mf_idx = F.cross_entropy(mf_idx_pred, mf_idx_gt.view(-1))
        # loss_mf_idx = F.binary_cross_entropy(mf_idx_pred, mf_idx_target)
        # loss_mf_idx = F.binary_cross_entropy_with_logits(mf_idx_pred, mf_idx_target, reduction='sum')


        nmf_pnts = data['nomanifold_samples'][:, :, :3]
        nmf_pnts.requires_grad_()
        nmf_idx_gt = data['nomanifold_samples'][:, :, -1].long()
        nmf_data = {
            'sample_pnts': nmf_pnts,
            # 'idx': nmf_idx_gt,
        }
        nmf_pred = net(nmf_data, sdf_latent=sdf_latent)
        nmf_sdf_pred = nmf_pred['pred_sdf']


        # nmf_idx_pred = nmf_pred['pred_idx']
        # nmf_idx_target = torch.zeros_like(nmf_idx_pred).scatter_(1, nmf_idx_gt.view(-1, 1), 1.0)
        # loss_nmf_idx = F.cross_entropy(nmf_idx_pred, nmf_idx_gt.view(-1))
        # loss_nmf_idx = F.binary_cross_entropy_with_logits(nmf_idx_pred, nmf_idx_target)
        # loss_total = loss_mf_idx + loss_nmf_idx
        # print(loss_mf_idx.item(), loss_nmf_idx.item(), (mf_idx_pred.max(-1)[1] == mf_idx_target.max(-1)[1]).sum())

        loss_total, loss_dict = loss_net(mf_pnts=mf_pnts,
                                         pred_mf_sdf=mf_sdf_pred,
                                         mf_norm=data['manifold_samples'][:, :, 3:-1],
                                         nmf_pnts=nmf_pnts,
                                         pred_nmf_sdf=nmf_sdf_pred,
                                         nmf_dist=data['nomanifold_samples'][:, :, 3:-1],
                                         sdf_latent=mc_latent)

        loss_total.backward()
        optimizer.step()

        loss['total'] += loss_total.item()
        # print(loss_total.item())
        for k,v in loss_dict.items():
            metric[k] += v
        #     print(f'loss {k} is {v}')

    return loss, metric, len(data_loader)

def train_2(data_loader, net, optimizer):
    net.train()
    loss = {'total': 0.0}
    metric = defaultdict(float)

    loss_net = mcif_loss_v2()
    for data, indices in tqdm(data_loader):
        optimizer.zero_grad()

        data['pnts'] = data['pnts'].cuda()
        data['manifold_samples'] = data['manifold_samples'].cuda()
        data['nomanifold_samples'] = data['nomanifold_samples'].cuda()

        sdf_latent = net({'pnts': data['pnts'], })['sdf_latent']

        mf_pnts = data['manifold_samples'][:, :, :3]
        mf_pnts.requires_grad_()
        mf_idx_gt = data['manifold_samples'][:, :, -1].long()
        mf_data = {
            'sample_pnts': mf_pnts,
            'idx': mf_idx_gt,
        }
        mf_pred = net(mf_data, {'sdf_latent':sdf_latent})
        mf_sdf_pred = mf_pred['pred_sdf']

        nmf_pnts = data['nomanifold_samples'][:, :, :3]
        nmf_pnts.requires_grad_()
        nmf_idx_gt = data['nomanifold_samples'][:, :, -1].long()
        nmf_data = {
            'sample_pnts': nmf_pnts,
            'idx': nmf_idx_gt,
        }
        nmf_pred = net(nmf_data, latent_code={'sdf_latent':sdf_latent})
        nmf_sdf_pred = nmf_pred['pred_sdf']

        loss_total, loss_dict = loss_net(
                                         mf_pnts=mf_pnts,
                                         pred_mf_sdf=mf_sdf_pred,
                                         mf_norm=data['manifold_samples'][:, :, 3:-1],
                                         nmf_pnts=nmf_pnts,
                                         pred_nmf_sdf=nmf_sdf_pred,
                                         nmf_dist=data['nomanifold_samples'][:, :, 3:-1],
                                         sdf_latent=sdf_latent)

        loss_total.backward()
        optimizer.step()

        loss['total'] += loss_total.item()
        for k, v in loss_dict.items():
            metric[k] += v

    return loss, metric, len(data_loader)

def eval():

    return

def sdf_create_mesh(decoder, sdf_latent, pnts,
                    N=256, max_batch=32 ** 3,
                    filename=None, offset=None, scale=None, voxel_idx=None):
    decoder.eval()
    pnts_tree = cKDTree(pnts.squeeze(0).detach().cpu().numpy())

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

        if voxel_idx is None:
            # find idx
            # correlated_idx = []
            # for idx_samples in range(len(sample_subset)):
            #     sample_query = sample_subset[idx_samples].detach().cpu().numpy()
            #     p_result = pnts_tree.query(sample_query)
            #     correlated_idx.append(p_result[1])
            # correlated_idx = torch.tensor(correlated_idx, dtype=torch.long, device=sdf_latent.device)

            sample_input = {
                'sample_pnts': sample_subset.unsqueeze(0),
            }
        else:
            # import pdb; pdb.set_trace()
            correlated_idx = voxel_idx[:, head: min(head + max_batch, num_samples)].cuda()
            sample_input = {
                'sample_pnts': sample_subset.unsqueeze(0),
                'idx': correlated_idx,
            }

        sdf_pred = decoder(sample_input, latent_code=sdf_latent)['pred_sdf']
        samples[head: min(head + max_batch, num_samples), 3] = (
            sdf_pred.squeeze().detach().cpu()
        )
        head += max_batch
    # else:
    #     sample_input = {
    #         'sample_pnts': samples[:,0:3].unsqueeze(0).cuda(),
    #         'idx': voxel_idx.cuda(),
    #     }
    #     sdf_pred = decoder(sample_input, sdf_latent=sdf_latent)['pred_sdf']
    #     samples[:, 3] = (sdf_pred.squeeze().detach().cpu())

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
                load_epoch=None):
    if load_epoch:
        mesh_dir = ws.get_dir('reconstruction_meshes_dir') / '{:04}'.format(load_epoch)
    else:
        mesh_dir = ws.get_dir('reconstruction_meshes_dir')

    net.eval()
    with torch.no_grad():
        for data, idx in tqdm(data_loader):
            mesh_file = mesh_dir / '{}.ply'.format(rec_files[idx][1:])
            if mesh_file.is_file():
                continue
            else:
                mesh_file.parent.mkdir(parents=True, exist_ok=True)

            if rec_num>1 and idx>rec_num-1:
                return

            pnts = data['pnts'].cuda()
            sdf_latent = net({'pnts': pnts,})

            scale_max = data['norm']['scale_max']
            scale_min = data['norm']['scale_min']
            sdf_create_mesh(decoder=net,
                            sdf_latent=sdf_latent,
                            pnts=pnts,
                            filename=mesh_file,
                            N=rec_res,
                            max_batch=int(2 ** 18),
                            scale=2.0/(scale_max-scale_min),
                            offset=-(0.5*(scale_max-scale_min)+scale_min),
                            voxel_idx=data['voxel_idx'])

def show_samples(data_loader, t_writer):

    for data, idx in tqdm(data_loader):
        if idx > 5:
            return

        mf_pnts = data['manifold_samples'][:,:,:3]
        tag = '{}/norm'.format(idx.item())
        t_writer.add_mesh(tag=tag,
                          vertices=mf_pnts,
                          global_step=0)

        nmf_pnts = data['nomanifold_samples'][:, :, :3]
        tag = '{}/nmf'.format(idx.item())
        t_writer.add_mesh(tag=tag,
                          vertices=nmf_pnts,
                          global_step=0)

        scale_max = data['norm']['scale_max']
        scale_min = data['norm']['scale_min']
        scale = 2.0 / (scale_max - scale_min)
        offset= -(0.5*(scale_max-scale_min)+scale_min)
        # import pdb; pdb.set_trace()
        mf_rec_pnts = mf_pnts/scale - offset
        tag = '{}/gt'.format(idx.item())
        t_writer.add_mesh(tag=tag,
                          vertices=mf_rec_pnts,
                          global_step=0)
