import os
import sys
rootpath=str("/home/syao/Program/Source/New3D")
sys.path.append(rootpath)
sys.path.extend([rootpath+i for i in os.listdir(rootpath) if i[0]!="."])

import torch
import json
import time
import math
import numpy as np
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

from workspace import get_spec_with_default
from networks.DeepSDF_Decoder import DeepSDF_Decoder
from utils.log_config import log_config
from utils.lr_config import adjust_learning_rate

__package__ = 'experiments.SDF_FAMILY'
from .run_set import *
from .train_program import train, eval, reconstruct
from .data_process import SDFSamples

if __name__ == '__main__':
    # torch.cuda.set_device(0)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    t = time.time()

    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-e',
        dest='experiment_config',
        default='/home/syao/Program/Source/New3D/experiments/SDF_FAMILY/experiments_config/dfaust_deepsdf.json'
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default="16"
    )
    arg_parser.add_argument(
        "--load_epoch",
        type=int,
        default=-1,
        help='Select epoch for loading parameters, \
        "-1" means that load from the latest record and is invalid for original "train" of run_type.'
    )
    arg_parser.add_argument(
        '--run_type',
        default='eval',
        help='Select from [train, eval, rec].'
    )
    arg_parser.add_argument(
        "--rec_res",
        type=int,
        default="256",
        help='Grid resolution.'
    )
    args = arg_parser.parse_args()

    with open(args.experiment_config, 'r') as f:
        exper_specs = json.load(f)

    # workspace config
    base_dir = exper_specs['ExperimentsDir']
    experiment_dir = base_dir + '/' + exper_specs['ExperimentName']
    # processed_data_dir = exper_specs['DatasetsDir']
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
    log_dir = params_io.ws.get_dir('log_dir')
    logger = log_config(filename=log_dir / '{}_{}_info.log'.format(args.run_type, current_time))
    logger.info(args)
    tensor_board_dir = params_io.ws.get_dir('tensor_board_dir') / f'{args.run_type}_{current_time}'
    # tensor_board_dir = params_io.ws.get_dir('tensor_board_dir')
    tensor_board_dir.mkdir(parents=True, exist_ok=True)
    tensor_board_writer = SummaryWriter(tensor_board_dir)


    # dataset config
    train_data_files = {
        'len': len(pdata_train_io.filenames),
        'sdf_files': [f.with_suffix('.npz') for f in
                      pdata_train_io.get_instance_filenames('sdf_samples_dir')],
        'norm_params_file': [f.with_suffix('.npz') for f
                             in pdata_train_io.get_instance_filenames('sdf_norm_params_dir')],
    }
    test_data_files = {
        'len': len(pdata_test_io.filenames),
        'sdf_files': [f.with_suffix('.npz') for f in
                      pdata_test_io.get_instance_filenames('sdf_samples_dir')],
        'norm_params_file': [f.with_suffix('.npz') for f in
                             pdata_test_io.get_instance_filenames('sdf_norm_params_dir')],
    }

    # dataset_dir = exper_specs['DatasetsDir']
    num_samp_per_scene = exper_specs["SamplesPerScene"]
    train_dataset = SDFSamples(files=train_data_files,
                               subsample=num_samp_per_scene)
    test_dataset = SDFSamples(files=test_data_files,
                              subsample=num_samp_per_scene,
                              load_ram=True)

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
    logger.debug("loading data with {} threads".format(num_data_loader_threads))

    # network setting
    latent_size = exper_specs["CodeLength"]
    net = DeepSDF_Decoder(latent_size, **exper_specs['NetworkSpecs']).cuda()

    # latent code setting
    code_bound = get_spec_with_default(exper_specs, "CodeBound", None)
    train_lat_vecs = torch.nn.Embedding(len(train_dataset), latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        train_lat_vecs.weight.data,
        0.0,
        get_spec_with_default(exper_specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    # optimizer setting
    lr_schedules = get_lr(lr_config=exper_specs['LearningRateSchedule'])
    optimizer = torch.optim.Adam(
        [
            {
                "params": net.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": train_lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    # parameters load setting
    save_info = {
        'model': (net, 'sdf_model_params_dir'),
        'latentcode': (train_lat_vecs, 'sdf_latent_codes_dir'),
        'optimizer': (optimizer, 'sdf_optimizer_params_dir'),
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
        # checkpoints config
        num_epochs = exper_specs['NumEpochs']
        log_frequency = exper_specs['SnapshotFrequency']
        checkpoints = list(
            range(
                exper_specs["SnapshotFrequency"],
                exper_specs["NumEpochs"] + 1,
                exper_specs["SnapshotFrequency"],
            )
        )
        for checkpoint in exper_specs["AdditionalSnapshots"]:
            checkpoints.append(checkpoint)
        checkpoints.sort()

        # train
        logger.info("training with {} GPU(s)".format(torch.cuda.device_count()))
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
                                           lat_vecs=train_lat_vecs,
                                           exper_specs=exper_specs,
                                           epoch=epoch)
            seconds_elapsed = time.time() - time_mark
            logger.info("Epoch {} cost {:.2f}s.".format(epoch, seconds_elapsed))
            for k,v in loss.items():
                logger.info("The {} loss is {}".format(k, v / iter_num))
                tensor_board_writer.add_scalar('Train_Loss/{}'.format(k), v/iter_num, epoch)
            for k,v in metric.items():
                logger.info("The {} distance is {}".format(k, v / iter_num))
                tensor_board_writer.add_scalar('Train_Metric/{}'.format(k), v / iter_num, epoch)

            # record
            if epoch in checkpoints:
                params_io.save_checkpoints(epoch=epoch, info=save_info)
            else:
                params_io.save_latest(epoch=epoch, info=save_info)
    elif args.run_type == 'eval':
        # eval
        eval_files = {'gt_sample':[f.with_suffix('.ply') for f in
                                   pdata_test_io.get_instance_filenames('sdf_surface_dir')],
                      'rec': [params_io.ws.get_dir('reconstruction_meshes_dir') / (f[1:]+'.ply')
                              for f in pdata_test_io.filenames],
                      'gt': [Path(exper_specs['DatasetsDir']) / (f[1:]+'.obj')
                             for f in pdata_test_io.filenames]
                      }
        # the data type of faces must be represented by a long data type, not use the pytorch tensorboard docs
        test_metric, test_statistic, iter_num = eval(files=eval_files,
                                                     load_epoch=start_epoch-1,
                                                     t_writer=tensor_board_writer)
        for k,v in test_metric.items():
            logger.info("The {} distance is {}".format(k, v / iter_num))
        for k,v in test_statistic.items():
            logger.info("The {} is {}".format(k, v))
    elif args.run_type == 'rec':
        # reconstruct
        rec_files = pdata_test_io.filenames
        reconstruct(data_loader=test_loader,
                    rec_files=rec_files,
                    net=net,
                    latent_size=latent_size,
                    ws=params_io.ws,
                    rec_res=args.rec_res)

    tensor_board_writer.close()
    t_duration = time.time() - t
    print("Done with {}s!".format(t_duration))
