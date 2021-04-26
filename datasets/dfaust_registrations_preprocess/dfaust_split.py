from pathlib import Path
import glob
import json
import random
from collections import defaultdict

if __name__ == '__main__':
    base_dir = Path('/home/syao/Program/Datasets/DFAUST')
    split_dir = Path('./splits')
    split_dir.mkdir(parents=True,exist_ok=True)

    with open('./dfaust_config.json', 'r') as f:
        config = json.load(f)

    dfaust_split_train = defaultdict()
    dfaust_split_test = defaultdict()

    for dataset in config['DFAUST']:
        sid = dataset.split('_')[0]
        sexy = dataset.split('_')[-1]

        train_dataset = defaultdict(list)
        test_dataset = defaultdict(list)
        for seq in config['DFAUST'][dataset]:
            tdir = base_dir/sid/seq
            # print(tdir)
            ifile = glob.glob(str(tdir/'*.obj'))
            files = [n.split('/')[-1].split('.')[0] for n in ifile]
            random.shuffle(files)

            file_nums = len(files)
            pos = int(file_nums * 0.8)
            train_dataset[seq] = files[:pos]
            test_dataset[seq] = files[pos:]

        # dfaust_split_train[dataset] = train_dataset
        # dfaust_split_test[dataset] = test_dataset

        filename_train = split_dir/"dfaust_{}_train.json".format(sid)
        filename_test = split_dir/"dfaust_{}_test.json".format(sid)
    
        with open(filename_train, 'w') as f:
            json.dump({"DFAUST_REGIS": {sid: train_dataset}}, f)

        with open(filename_test, 'w') as f:
            json.dump({"DFAUST_REGIS": {sid: test_dataset}}, f)
