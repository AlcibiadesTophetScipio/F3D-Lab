import os
import sys
rootpath=str("/home/syao/Program/Source/New3D")
sys.path.append(rootpath)
sys.path.extend([rootpath+i for i in os.listdir(rootpath) if i[0]!="."])

import torch
import json
import time
import numpy as np
from psbody.mesh import Mesh
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

from networks.SpiralNet_Plus import SpiralNet_Plus
from utils.log_config import log_config
from utils.lr_config import adjust_learning_rate
from datasets import dfaust_dataset
from datasets.base import dataset_normalize
from mesh_tools.mesh_operator import scipy_to_torch_sparse, preprocess_spiral
from mesh_tools import coma_sampling

__package__ = 'experiments.AE_SOLO'
from .run_set import *
from .train_program import train, eval, reconstruct


if __name__ == '__main__':
    # torch.cuda.set_device(0)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    t = time.time()

    import argparse
    arg_parser = argparse.ArgumentParser(description="Train a SM network")
    arg_parser.add_argument(
        '-e',
        dest='experiment_config',
        default='/home/syao/Program/Source/New3D/experiments/AE_SOLO/experiments_config/dfaust_spiralnet_plus.json',
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default="16"
    )
    arg_parser.add_argument(
        "--load_epoch",
        type=int,
        default=1
    )
    arg_parser.add_argument(
        '--run_type',
        default='train'
    )
    args = arg_parser.parse_args()

    with open(args.experiment_config, 'r') as f:
        exper_specs = json.load(f)

    # workspace config
    base_dir = exper_specs['ExperimentsDir']
    experiment_dir = base_dir + '/' + exper_specs['ExperimentName']
    processed_data_dir = base_dir + '/' + exper_specs['ProcessedData']
    ws = get_workspace(experiment_dir=experiment_dir,
                       processed_data_dir=processed_data_dir)
    params_io = ws['ws_params']
    pdata_io = ws['ws_pdata']

    # log config
    log_dir = params_io.ws.get_dir('log_dir')
    logger = log_config(filename=log_dir / '{}_{}_info.log'.format(args.run_type, current_time))
    logger.info(args)
    tensor_board_dir = params_io.ws.get_dir('tensor_board_dir') / f'{args.run_type}_{current_time}'
    # tensor_board_dir = params_io.ws.get_dir('tensor_board_dir')
    tensor_board_dir.mkdir(parents=True, exist_ok=True)
    tensor_board_writer = SummaryWriter(tensor_board_dir)


    # dataset config
    dataset_dir = exper_specs['DatasetsDir']
    train_split_file = exper_specs['TrainSplit']
    test_split_file = exper_specs['TestSplit']
    train_dataset, test_dataset = dfaust_dataset.get_dataset(dataset_dir=dataset_dir,
                                                             train_split_file=train_split_file,
                                                             test_split_file=test_split_file)
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    # dataset normalization
    train_norm_file = pdata_io.get_dir('dataset_norm_dir') / '{}_train_norm.pth'.format(exper_specs['ExperimentName'])
    test_norm_file = pdata_io.get_dir('dataset_norm_dir') / '{}_test_norm.pth'.format(exper_specs['ExperimentName'])
    if train_norm_file.is_file():
        logger.info('Load dataset norm.')
        train_norm = torch.load(train_norm_file)
        test_norm = torch.load(test_norm_file)
    else:
        logger.info('Dataset Normalizing...')
        train_norm = dataset_normalize(train_dataset)
        test_norm = dataset_normalize(test_dataset)
        torch.save(train_norm, train_norm_file)
        torch.save(test_norm, test_norm_file)

    # mesh template
    mesh_template = Mesh(filename=exper_specs['TemplateFile'])
    mesh_sample_pth = pdata_io.get_dir('template_dir') / '{}.pth'.format(exper_specs['ExperimentName'])
    if mesh_sample_pth.is_file():
        logger.info('Load mesh template.')
    mesh_sample = coma_sampling(template_mesh=mesh_template,
                                mesh_sample_pth=mesh_sample_pth)
    A = mesh_sample['A']
    D = mesh_sample['D']
    U = mesh_sample['U']
    M = mesh_sample['M']

    D_t = [scipy_to_torch_sparse(d).cuda() for d in D]
    U_t = [scipy_to_torch_sparse(u).cuda() for u in U]
    A_t = [scipy_to_torch_sparse(a).cuda() for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    seq_length = [9, 9, 9, 9]
    dilation = [1, 1, 1, 1]
    spiral_indices_list = [
        preprocess_spiral(face=mesh_sample['F'][idx],
                          seq_length=seq_length[idx],
                          vertices=None,
                          dilation=dilation[idx]).cuda()
        for idx in range(len(mesh_sample['F']) - 1)
    ]

    # network setting
    net = SpiralNet_Plus(in_channels=3,
                         out_channels=[16, 16, 16, 32],
                         latent_channels=8,
                         spiral_indices=spiral_indices_list,
                         down_transform=D_t,
                         up_transform=U_t).cuda()

    # optimizer setting
    lr_schedules = get_lr(lr_config=exper_specs['LearningRateSchedule'])
    optimizer = torch.optim.Adam(params=net.parameters(),
                                 lr=lr_schedules[0].get_learning_rate(0),
                                 weight_decay=exper_specs['WeightDecay'])

    # parameters load setting
    save_info = {
        'model': (net, 'model_params_dir'),
        'optimizer': (optimizer, 'optimizer_params_dir'),
    }
    start_epoch = args.load_epoch
    if start_epoch > 1:
        start_epoch = params_io.load_checkpoint(info=save_info, epoch=start_epoch)
        logger.info(f"Load from {start_epoch}.")
        start_epoch += 1
    elif start_epoch < 0:
        start_epoch = params_io.load_checkpoint(info=save_info, epoch=None)
        logger.info(f"Load from {start_epoch}.")
        start_epoch += 1
    else:
        start_epoch = 1


    # msg show
    logger.info(net)
    logger.info(optimizer)


    if args.run_type == 'train':
        # train
        num_epochs = exper_specs['NumEpochs']
        log_frequency = exper_specs['SnapshotFrequency']
        for epoch in range(start_epoch, num_epochs+1):
            logger.info('\n')
            logger.info("Training at epoch:{}".format(epoch))
            time_mark = time.time()

            # training setup
            adjust_learning_rate(lr_schedules, optimizer, epoch)

            # model fit
            loss, metric, iter_num = train(data_loader=train_loader,
                                           net=net,
                                           optimizer=optimizer,
                                           data_norm=train_norm)

            seconds_elapsed = time.time() - time_mark
            logger.info("Epoch {} cost {:.2f}s.".format(epoch, seconds_elapsed))
            for k,v in loss.items():
                logger.info("The {} loss is {}".format(k, v / iter_num))
                tensor_board_writer.add_scalar('Train_Loss/{}'.format(k), v/iter_num, epoch)
            for k,v in metric.items():
                logger.info("The {} distance is {}".format(k, v / iter_num))
                tensor_board_writer.add_scalar('Train_Metric/{}'.format(k), v / iter_num, epoch)

            # record
            if epoch % log_frequency == 0:
                params_io.save_checkpoints(epoch=epoch, info=save_info)
            else:
                params_io.save_latest(epoch=epoch, info=save_info)
    elif args.run_type == 'eval':
        # eval
        # the data type of faces must be represented by a long data type, not use the pytorch tensorboard docs
        faces = torch.from_numpy(mesh_template.f.astype(np.int64)).unsqueeze(0)
        test_metric, test_statistic, iter_num = eval(data_loader=test_loader,
                                                     net=net,
                                                     data_norm=test_norm,
                                                     load_epoch=start_epoch-1,
                                                     t_writer=tensor_board_writer,
                                                     faces=faces)
        for k,v in test_metric.items():
            logger.info("The {} distance is {}".format(k, v / iter_num))
        for k,v in test_statistic.items():
            logger.info("The {} is {}".format(k, v))
    elif args.run_type == 'rec':
        # reconstruct
        reconstruct(data_loader=test_loader,
                    net=net,
                    data_norm=test_norm,
                    mesh_template=mesh_template,
                    save_dir=params_io.ws.get_dir('reconstruction_meshes_dir'))


    tensor_board_writer.close()
    t_duration = time.time() - t
    print("Done with {}s!".format(t_duration))
