import torch
from pytorch3d.loss import chamfer_distance

def chamferDis_bpm(pc_source, pc_target, co=1000.0):
    '''

    :param pc_source: [B N C]
    :param pc_target: [B N C]
    :param co: multiply coefficient
    :return: average chamfer distance
    '''
    # average chamfer dis of points, not the object
    chamfer_dis, _ = chamfer_distance(pc_source, pc_target,
                                    point_reduction='mean',
                                    batch_reduction='mean')
    return chamfer_dis * co

def l1Dis_bpm(pc_source, pc_target):
    '''

    :param pc_source: [B N C]
    :param pc_target: [B N C]
    :return: average l1 distance
    '''
    # equal to l1 loss
    batch = pc_source.shape[0]
    point_num = pc_source.shape[1]
    channel_num = pc_source.shape[2]
    return (pc_source - pc_target).abs().sum()/(batch * channel_num * point_num)

def euclideanDis_bpm(pc_source, pc_target):
    '''

    :param pc_source: [B N C]
    :param pc_target: [B N C]
    :return: average euclidean distance
    '''
    # not same as mse loss
    batch = pc_source.shape[0]
    point_num = pc_source.shape[1]
    dist = (pc_source - pc_target).pow(2).sum(-1).pow(0.5).sum()

    return  dist / (batch * point_num)

def euclideanDis_keep(pc_source, pc_target):
    '''

    :param pc_source: [B N C]
    :param pc_target: [B N C]
    :return: average euclidean distance
    '''
    # not same as mse loss
    batch = pc_source.shape[0]
    point_num = pc_source.shape[1]
    dist = (pc_source - pc_target).pow(2).sum(-1).pow(0.5)

    return  dist

if __name__ == '__main__':

    torch.cuda.set_device(0)

    pd_x = torch.randn([32,6890,3]).cuda()
    pd_y = torch.randn_like(pd_x)

    torch3d_chamferDis, _ = chamfer_distance(pd_x, pd_y)
    torch3d_chamferDis_bpmean = chamferDis_bpm(pd_x, pd_y)

    torch_l1Dist = torch.nn.functional.l1_loss(pd_x, pd_y,
                                       reduction='mean').item()
    torch_l1Dist_bpmean = l1Dis_bpm(pd_x,pd_y)

    torch_l2Dist = torch.nn.functional.mse_loss(pd_x, pd_y, reduction='mean').item()
    torch_l2Dist_bpmean = euclideanDis_bpm(pd_x, pd_y)
    torch_l2Dist_keep = euclideanDis_keep(pd_x, pd_y)

    print("Done!")
