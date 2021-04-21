from pathlib import Path
import torch
from collections import OrderedDict

if __name__ == '__main__':
    data = Path('/home/syao/Program/Experiments/SM/dfaust_50002/SdfModelParameters')
    file_name = 'old_2000.pth'
    new_file = data/'new.pth'
    params = torch.load(data/file_name)

    epoch = params['epoch']
    state_dict = params['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    torch.save({'epoch':epoch,
                'model_state_dict':new_state_dict}, new_file)