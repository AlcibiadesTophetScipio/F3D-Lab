
from pathlib import Path

from workspace import Handled_Data_IO, Parametrers_IO, WorkSpaceManager
from utils.lr_config import StepLearningRateSchedule

def get_workspace(experiment_dir,
                  processed_data_dir):
    base_dir = Path(experiment_dir)
    sub_dir = {
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
        'dataset_norm_dir': 'DatasetNormalization',
        'template_dir': 'TemplateSample',
    }
    pdata_io_dir_config = {
        'base_dir': base_dir,
        'sub_dir': sub_dir,
    }
    pdata_io = WorkSpaceManager(config=pdata_io_dir_config)

    return {
        'ws_params': params_io,
        'ws_pdata': pdata_io,
    }

def get_lr(lr_config):
    schedules = []
    schedules.append(
        StepLearningRateSchedule(
            lr_config['Initial'],
            lr_config["Interval"],
            lr_config["Factor"],
    ))
    return schedules