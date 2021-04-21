from collections import defaultdict
import torch.nn.functional as F

from utils.metrics import *

def calculate_metric(gt,pred):
    return {'euclidean': euclideanDis_bpm(gt, pred),
            'chamfer': chamferDis_bpm(gt, pred),
            'l1': l1Dis_bpm(gt, pred)}

def train(data_loader, net, optimizer, data_norm):
    net.train()
    loss = {'total': 0.0}
    metric = defaultdict(float)

    mean = data_norm['mean'].cuda()
    std = data_norm['std'].cuda()
    for data, indices in data_loader:
        optimizer.zero_grad()

        point_cloud = data['points'].cuda()
        x = (point_cloud-mean)/std
        gt = {'x': x}
        pred = net(gt)

        train_loss = F.l1_loss(pred['re_x'], gt['x'])
        train_loss.backward()
        optimizer.step()

        loss['total'] += train_loss.item()

        recover = pred['re_x']*std + mean
        instance_metric = calculate_metric(gt=point_cloud, pred=recover)
        for k, v in instance_metric.items():
            metric[k] += v

    return loss, metric, len(data_loader)

def eval(data_loader, net, data_norm, load_epoch=0, t_writer=None, faces=None):
    net.eval()
    metric = defaultdict(float)
    metric_euclidean_statistic = []

    mean = data_norm['mean'].cuda()
    std = data_norm['std'].cuda()

    tensor_show_num = len(data_loader)*0.1 if len(data_loader)*0.1<20 else 20
    tensor_show_list = torch.round(torch.rand(tensor_show_num)*len(data_loader))
    with torch.no_grad():
        for data, indices in data_loader:
            point_cloud = data['points'].cuda()
            x = (point_cloud - mean) / std
            gt = {'x': x}
            pred = net(gt)

            recover = pred['re_x'] * std + mean
            instance_metric = calculate_metric(gt=point_cloud, pred=recover)
            for k, v in instance_metric.items():
                metric[k] += v

            # construct statistic like coma
            metric_euclidean_statistic.append(euclideanDis_keep(point_cloud, recover))

            # show mesh in tensorboard
            if t_writer and indices in tensor_show_list:
                tag = '{}/gt'.format(indices.item())
                t_writer.add_mesh(tag=tag,
                                  global_step=load_epoch,
                                  vertices=point_cloud,
                                  faces=faces)
                tag = '{}/pred'.format(indices.item())
                t_writer.add_mesh(tag=tag,
                                  global_step=load_epoch,
                                  vertices=recover,
                                  faces=faces)

    statistic_error = torch.cat(metric_euclidean_statistic, dim=0)
    statistic = {"statistic_mean": statistic_error.mean(),
                 "statistic_std": statistic_error.std(),
                 "statistic_median": statistic_error.median()}

    return metric, statistic, len(data_loader)

def reconstruct(data_loader, net, data_norm, mesh_template, save_dir):
    save_dir = str(save_dir)
    net.eval()
    mean = data_norm['mean'].cuda()
    std = data_norm['std'].cuda()

    with torch.no_grad():
        for data, indices in data_loader:
            point_cloud = data['points'].cuda()
            x = (point_cloud - mean) / std
            gt = {'x': x}
            pred = net(gt)
            recover = pred['re_x'] * std + mean

            # save instance
            mesh_template.v = point_cloud.squeeze().detach().cpu().numpy()
            mesh_template.write_ply(save_dir+'/{:d}_gt.ply'.format(indices.item()))
            mesh_template.v = recover.squeeze().detach().cpu().numpy()
            mesh_template.write_ply(save_dir+'/{:d}_pred.ply'.format(indices.item()))