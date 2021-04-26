import h5py
import json
from collections import OrderedDict, defaultdict

if __name__ == "__main__":
    f_dataset_path = '/home/syao/Program/Datasets/DFAUST/raw/registrations_f.hdf5'
    m_dataset_path = '/home/syao/Program/Datasets/DFAUST/raw/registrations_m.hdf5'

    sid = {}
    seq = {}
    with h5py.File(f_dataset_path, 'r') as f:
        for filename in f:
            pos = filename.find('_')
            num = "{}_female".format(filename[:pos])
            act = filename[pos+1:]
            sid[num] = num
            seq[act] = act

    with h5py.File(m_dataset_path, 'r') as f:
        for filename in f:
            pos = filename.find('_')
            num = "{}_male".format(filename[:pos])
            act = filename[pos+1:]
            sid[num] = num
            seq[act] = act

    print(sid.keys(),seq.keys())

    dataset_statistic = defaultdict(list)
    dataset_statistic.fromkeys(sid.keys(), [])
    for dataset_name in sid:
        for act_name in seq:
            name_without_sexy = dataset_name.split('_')[0]
            sexy = dataset_name.split('_')[1]
            sidseq = name_without_sexy+'_'+act_name
            if sexy == 'female':
                with h5py.File(f_dataset_path, 'r') as f:
                    if sidseq in f:
                        dataset_statistic[dataset_name].append(act_name)
                        continue
            else:
                with h5py.File(m_dataset_path, 'r') as f:
                    if sidseq in f:
                        dataset_statistic[dataset_name].append(act_name)
                        continue

    with open("./dfaust_config.json", "w") as f:
        json_str = json.dump({"DFAUST":dataset_statistic}, f)

