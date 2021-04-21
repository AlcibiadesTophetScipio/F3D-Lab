import torch
import json
from pathlib import Path

def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default

class WorkSpaceManager(object):
    def __init__(self, config):
        self.base_dir = config['base_dir']
        self.sub_dir = {**config['sub_dir']}

    def get_dir(self, dir_name=None):
        if dir_name is None:
            return Path(self.base_dir)
        try:
            r_dir = Path(self.base_dir) / self.sub_dir[dir_name]
            r_dir.mkdir(parents=True, exist_ok=True)
        except:
            raise Exception('Not specify directory by name: {}'.format(dir_name))
        return r_dir

class Parametrers_IO(object):
    def __init__(self, dir_config):
        self.ws = WorkSpaceManager(dir_config)

    def _save(self, epoch, parameters, dir_name, file_name):
        torch.save(
            {"epoch": epoch,
             "paramaters": parameters.state_dict(),
             },
            self.ws.get_dir(dir_name) / file_name,
        )

    def _load(self, obj, dir_name, file_name, params):
        file_path = self.ws.get_dir(dir_name) / file_name
        if not file_path.is_file():
            raise Exception('Load fail with {}.'.format(file_path))

        data = torch.load(file_path)
        obj.load_state_dict(data[params])

        return data['epoch']

    def save_latest(self, epoch, info):
        for k, v in info.items():
            self._save(epoch=epoch, parameters=v[0], dir_name=v[1], file_name='latest.pth')

    def save_checkpoints(self, epoch, info):
        for k, v in info.items():
            self._save(epoch=epoch, parameters=v[0], dir_name=v[1], file_name='{}.pth'.format(epoch))

    def load_checkpoint(self, info, epoch=None, params_name='paramaters'):
        file_name = '{}.pth'.format(epoch) if epoch else 'latest.pth'
        for k, v in info.items():
            load_eopoch = self._load(obj=v[0],
                                     dir_name=v[1],
                                     file_name=file_name,
                                     params=params_name)

        return load_eopoch

def get_object_name(f, par_name=''):
    if isinstance(f, dict):
        for k,v in f.items():
            for n in get_object_name(v, par_name=par_name+'/'+k):
                yield n

            # yield k, get_dict_name(v)
    if isinstance(f, list):
        for n in f:
            yield par_name+'/'+n

    pass

class Handled_Data_IO(object):
    def __init__(self, dir_config, split_file=None):
        self.ws = WorkSpaceManager(dir_config)
        if split_file:
            self.filenames = self._get_files(split_file)
        else:
            self.filenames = None

    def _get_files(self, split_file):
        with open(split_file, 'r') as f:
            split = json.load(f)
        iter_filenames = get_object_name(split)
        return list(iter_filenames)

    def get_instance_filenames(self, dir_name, split_file=None):
        if split_file:
            self.filenames = self._get_files(split_file)

        if not self.filenames:
            raise Exception('Please provide datasets split config file.')

        for n in self.filenames:
            yield self.ws.get_dir(dir_name)/n[1:]

