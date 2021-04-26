import concurrent.futures
import subprocess
import logging
import json

from pathlib import Path

def process_data(executable, sid, seq, tdir, path):
    command = [executable, '--sid', sid, '--seq', seq, '--tdir', tdir, '--path', path]
    logging.debug(command)

    logging.info('Processing {}_{}'.format(sid,seq))
    subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    subproc.wait()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data_dir = Path('/home/syao/Program/Datasets/DFAUST_original/raw')
    target_dir = Path('/home/syao/Program/Datasets/DFAUST')

    with open('./dfaust_config.json', 'r') as f:
        config = json.load(f)
    logging.debug(config)

    raw_data_args = []

    for dataset in config['DFAUST']:
        sid = dataset.split('_')[0]
        sexy = dataset.split('_')[-1]
        if sexy == 'male':
            sexy = 'm'
        else:
            sexy = 'f'

        for seq in config['DFAUST'][dataset]:
            tdir = target_dir/sid/seq
            if not tdir.exists():
                raw_data_args.append(
                    (
                        sid,
                        sexy,
                        seq,
                    )
                )
    logging.debug(raw_data_args)

    executable = 'write_sequence_to_obj.py'
    with concurrent.futures.ThreadPoolExecutor(
                max_workers=int(8)
        ) as executor:
        for (sid,sexy,seq) in raw_data_args:
            logging.debug(sid, seq)

            tdir = target_dir/sid
            source_path = data_dir/'registrations_{}.hdf5'.format(sexy)
            executor.submit(
                process_data,
                executable=str(executable),
                sid=sid,
                seq=seq,
                tdir=str(tdir),
                path=str(source_path)
            )
        executor.shutdown()
