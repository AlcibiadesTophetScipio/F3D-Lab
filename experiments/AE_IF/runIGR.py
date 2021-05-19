import os
import sys
rootpath=str("/home/syao/Program/Source/New3D")
sys.path.append(rootpath)
sys.path.extend([rootpath+i for i in os.listdir(rootpath) if i[0]!="."])

import torch
import json
import time
import numpy as np
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

from workspace import get_spec_with_default
from utils.log_config import log_config
from utils.lr_config import adjust_learning_rate
from networks.IgrNet import IGRNet
from experiments.IGR_FAMILY.train_program import train, reconstruct

__package__ = 'experiments.AE_IF'
from .run_set import *
from .data_process import MCIFSamples

def run(args, exper_specs):
    # workspace config
    base_dir = exper_specs['ExperimentsDir']
    experiment_dir = base_dir + '/' + exper_specs['ExperimentName']
    processed_data_dir = base_dir + '/' + exper_specs['ProcessedData']
    train_split_file = exper_specs['TrainSplit']
    test_split_file = exper_specs['TestSplit']
    ws = get_workspace(experiment_dir=experiment_dir,
                       processed_data_dir=processed_data_dir,
                       train_split_file=exper_specs['TrainSplit'],
                       test_split_file=exper_specs['TestSplit'])
    params_io = ws['ws_params']
    pdata_train_io = ws['ws_train_pdata']
    pdata_test_io = ws['ws_test_pdata']

    # log config
    if args.log:
        logger_file = params_io.ws.get_dir('log_dir') / \
                      '{}_{}_info.log'.format(args.run_type, current_time)
        tensor_board_dir = params_io.ws.get_dir('tensor_board_dir') / \
                           f'{args.run_type}_{current_time}'
    else:
        logger_file = params_io.ws.get_dir('log_dir') / 'temp.log'
        tensor_board_dir = params_io.ws.get_dir('tensor_board_dir') / 'temp'

    logger = log_config(filename=logger_file)
    logger.info(args)
    tensor_board_dir.mkdir(parents=True, exist_ok=True)
    tensor_board_writer = SummaryWriter(tensor_board_dir)

    # dataset config
    train_data_files = {
        'len': len(pdata_train_io.filenames),
        'pnts_files': [f.with_suffix(f.suffix + '.npy') for f in
                       pdata_train_io.get_instance_filenames('mcif_pnts_dir')],
        'norm_params_files': [f.with_suffix(f.suffix + '.npy') for f in
                              pdata_train_io.get_instance_filenames('mcif_norm_params_dir')],
        'manifold_files': [f.with_suffix(f.suffix + '.npy') for f in
                           pdata_train_io.get_instance_filenames('mcif_mf_samples_dir')],
        'nomanifold_files': [f.with_suffix(f.suffix + '.npy') for f in
                             pdata_train_io.get_instance_filenames('mcif_nomf_samples_dir')],
    }
    test_data_files = {
        'len': len(pdata_test_io.filenames),
        'pnts_files': [f.with_suffix(f.suffix + '.npy') for f in
                       pdata_test_io.get_instance_filenames('mcif_pnts_dir')],
        'norm_params_files': [f.with_suffix(f.suffix + '.npy') for f in
                              pdata_test_io.get_instance_filenames('mcif_norm_params_dir')],
        'manifold_files': [f.with_suffix(f.suffix + '.npy') for f in
                           pdata_test_io.get_instance_filenames('mcif_mf_samples_dir')],
        'nomanifold_files': [f.with_suffix(f.suffix + '.npy') for f in
                             pdata_test_io.get_instance_filenames('mcif_nomf_samples_dir')],
        'voxel_idx_files': [f.with_suffix(f.suffix + '.npy') for f in
                            pdata_test_io.get_instance_filenames('mcif_voxel_idx_dir')],
    }

    num_manifold_subsamp = exper_specs["ManifoldSubSamples"]
    num_nomanifold_subsamp = exper_specs['NomanifoldSubSamples']
    train_dataset = MCIFSamples(files=train_data_files,
                                manifold_subsample=num_manifold_subsamp,
                                nomanifold_subsample=num_nomanifold_subsamp)
    test_dataset = MCIFSamples(files=test_data_files,
                               manifold_subsample=num_manifold_subsamp,
                               nomanifold_subsample=num_nomanifold_subsamp,
                               voxel_idx=False)

    num_data_loader_threads = get_spec_with_default(exper_specs, "DataLoaderThreads", 8)
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_data_loader_threads
    )
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_data_loader_threads
    )

    # network setting
    latent_size = exper_specs["CodeLength"]
    net = IGRNet(d_in=(3+latent_size),
                 **exper_specs['NetworkSpecs']).cuda()

    # latent code setting
    train_lat_vecs_embed = torch.nn.Embedding(len(train_dataset), latent_size).cuda()
    torch.nn.init.constant_(
        train_lat_vecs_embed.weight.data,
        0.0
    )
    train_lat_vecs = torch.zeros(len(train_dataset), latent_size).cuda()
    train_lat_vecs.requires_grad_()

    # optimizer setting
    lr_schedules = get_lr(lr_config=exper_specs['LearningRateSchedule'])
    optimizer = torch.optim.Adam(
        [
            {
                "params": net.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": train_lat_vecs,
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    # parameters load setting
    save_info = {
        'model': (net, 'model_params_dir'),
        'latentcode': (train_lat_vecs_embed, 'latent_codes_dir'),
        'optimizer': (optimizer, 'optimizer_params_dir'),
    }
    start_epoch = args.load_epoch
    if start_epoch > 1:
        start_epoch = params_io.load_checkpoint(info=save_info, epoch=start_epoch)
        logger.info(f"Load from {start_epoch}.")
        start_epoch += 1
        train_lat_vecs = train_lat_vecs_embed.weight.data
    elif start_epoch < 0:
        start_epoch = params_io.load_checkpoint(info=save_info, epoch=None)
        logger.info(f"Load from {start_epoch}.")
        start_epoch += 1
        train_lat_vecs = train_lat_vecs_embed.weight.data
    else:
        start_epoch = 1

    # msg show
    logger.info(net)
    logger.info(optimizer)

    if args.run_type == 'train':
        # train
        num_epochs = exper_specs['NumEpochs']
        log_frequency = exper_specs['SnapshotFrequency']
        for epoch in range(start_epoch, num_epochs + 1):
            logger.info('\n')
            logger.info("Training at epoch:{}".format(epoch))
            time_mark = time.time()

            # training setup
            adjust_learning_rate(lr_schedules, optimizer, epoch)

            # model fit
            loss, metric, iter_num = train(data_loader=train_loader,
                                           net=net,
                                           optimizer=optimizer,
                                           lat_vecs=train_lat_vecs)

            seconds_elapsed = time.time() - time_mark
            logger.info("Epoch {} cost {:.2f}s.".format(epoch, seconds_elapsed))
            for k, v in loss.items():
                logger.info("The {} loss is {}".format(k, v / iter_num))
                tensor_board_writer.add_scalar('Train_Loss/{}'.format(k), v / iter_num, epoch)
            for k, v in metric.items():
                logger.info("The {} distance is {}".format(k, v / iter_num))
                tensor_board_writer.add_scalar('Train_Metric/{}'.format(k), v / iter_num, epoch)

            # record
            if epoch % log_frequency == 0:
                train_lat_vecs_embed.weight.data = train_lat_vecs
                params_io.save_checkpoints(epoch=epoch, info=save_info)
            else:
                train_lat_vecs_embed.weight.data = train_lat_vecs
                params_io.save_latest(epoch=epoch, info=save_info)
    elif args.run_type == 'eval':
        # eval
        # the data type of faces must be represented by a long data type, not use the pytorch tensorboard docs
        faces = torch.from_numpy(mesh_template.f.astype(np.int64)).unsqueeze(0)
        test_metric, test_statistic, iter_num = eval(data_loader=test_loader,
                                                     net=net,
                                                     data_norm=test_norm,
                                                     load_epoch=start_epoch - 1,
                                                     t_writer=tensor_board_writer,
                                                     faces=faces)
        for k, v in test_metric.items():
            logger.info("The {} distance is {}".format(k, v / iter_num))
        for k, v in test_statistic.items():
            logger.info("The {} is {}".format(k, v))
    elif args.run_type == 'rec':
        # reconstruct
        reconstruct(data_loader=test_loader,
                    net=net,
                    rec_res=args.rec_res,
                    rec_files=pdata_test_io.filenames,
                    ws=params_io.ws,
                    latent_size=latent_size,
                    rec_num=get_spec_with_default(exper_specs, "ReconstructionNums", 0),
                    load_epoch=start_epoch-1,
                    t_writer=tensor_board_writer)

    tensor_board_writer.close()
    t_duration = time.time() - t
    logger.info("Done with {}s!".format(t_duration))

if __name__ == '__main__':
    # torch.cuda.set_device(0)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    t = time.time()

    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default="16"
    )
    arg_parser.add_argument(
        "--load_epoch",
        type=int,
        default=-1,
        help='''Select epoch for loading parameters,
            "-1" means that load from the latest record and is invalid for original "train" of run_type.'''
    )
    arg_parser.add_argument(
        '--run_type',
        default='rec',
        help='Select from [train, eval, rec].'
    )
    arg_parser.add_argument(
        "--rec_res",
        type=int,
        default="256",
        help='Grid resolution.'
    )
    arg_parser.add_argument(
        '--log',
        action="store_true"
    )
    args = arg_parser.parse_args()

    exp_etc = '/home/syao/Program/Source/New3D/experiments/AE_IF/experiments_config/dfaust_igr.json'
    with open(exp_etc, 'r') as f:
        exper_specs = json.load(f)

    run(args=args, exper_specs=exper_specs)
