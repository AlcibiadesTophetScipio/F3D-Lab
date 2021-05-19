
from pathlib import Path

from workspace import Handled_Data_IO, Parametrers_IO, WorkSpaceManager
from utils.lr_config import StepLearningRateSchedule, \
    WarmupLearningRateSchedule, \
    ConstantLearningRateSchedule

def get_workspace(experiment_dir,
                  processed_data_dir,
                  train_split_file,
                  test_split_file
                  ):
    base_dir = Path(experiment_dir)
    sub_dir = {
        'latent_codes_dir': 'LatentCodes',
        'model_params_dir': 'ModelParameters',
        'optimizer_params_dir': 'OptimizerParameters',
        'reconstruction_meshes_dir': 'Reconstruction',
        'log_dir': 'Logs',
        'tensor_board_dir': "TensorBoard",
    }
    net_log_dir_config = {
        'base_dir': base_dir,
        'sub_dir': sub_dir,
    }
    params_io = Parametrers_IO(net_log_dir_config)

    base_dir = Path(processed_data_dir)
    sub_dir = {
        'template_file_dir': 'TemplateSample',
        'mcif_pnts_dir': 'McifPnts',
        'mcif_mf_samples_dir': 'McifManifoldSamples',
        'mcif_nomf_samples_dir': 'McifNomanifoldSamples',
        'mcif_norm_params_dir': 'NormalizationParameters',
        'mcif_voxel_idx_dir': 'VoxelIndex',
    }
    pdata_io_dir_config = {
        'base_dir': base_dir,
        'sub_dir': sub_dir,
    }
    train_data_io = Handled_Data_IO(dir_config=pdata_io_dir_config,
                                    split_file=train_split_file)
    test_data_io = Handled_Data_IO(dir_config=pdata_io_dir_config,
                                   split_file=test_split_file)

    return {
        'ws_params': params_io,
        'ws_train_pdata': train_data_io,
        'ws_test_pdata': test_data_io,
    }

def get_lr(lr_config):
    schedules = []

    for schedule_specs in lr_config:
        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules