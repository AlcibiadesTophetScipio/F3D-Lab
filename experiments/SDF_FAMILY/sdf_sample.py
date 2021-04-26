import torch
from torch_geometric.utils import to_undirected
import torch_geometric
from pathlib import Path
import concurrent.futures
import subprocess
import logging

from workspace import Handled_Data_IO

# import smio
# def handle_obj(source_dir, target_dir, files, num_nodes=6890):
#     for f in files:
#         obj_file = (source_dir / f[1:]).with_suffix('.obj')
#         pt_file = (target_dir / f[1:]).with_suffix('.pt')
#         if pt_file.is_file():
#             continue
#         pt_file.parent.mkdir(parents=True, exist_ok=True)
#
#         if not obj_file.is_file():
#             raise Exception(
#                 "Requested non-existent file '{}'".format(obj_file)
#             )
#         obj_data = smio.read_obj(obj_file)
#         pos = obj_data['pos']
#         face = obj_data['face']
#         edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
#         edge_index = to_undirected(edge_index, num_nodes=num_nodes)
#
#         pt_data = torch_geometric.data.Data(pos=pos, edge_index=edge_index)
#         torch.save(pt_data, pt_file)

def process_mesh(mesh_filepath, target_filepath, executable, additional_args):
    logging.info(mesh_filepath + " --> " + target_filepath)
    command = [executable, "-m", mesh_filepath, "-o", target_filepath] + additional_args

    subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    subproc.wait()


def sdf_prepreocess(source_dir, data_io, files,
                    surface_sampling=False,
                    test_sampling=False):
    additional_general_args = []
    if surface_sampling:
        executable = ("bin/SampleVisibleMeshSurface")
        subdir = data_io.ws.get_dir('sdf_surface_dir')
        extension = ".ply"
    else:
        executable = ("bin/PreprocessMesh")
        subdir = data_io.ws.get_dir('sdf_samples_dir')
        extension = ".npz"
        if test_sampling:
            additional_general_args += ["-t"]

    meshes_targets_and_specific_args = []
    for f in files:
        obj_file = (source_dir / f[1:]).with_suffix('.obj')
        sdf_file = (subdir / f[1:]).with_suffix(extension)
        if sdf_file.is_file():
            continue

        sdf_file.parent.mkdir(parents=True, exist_ok=True)
        if not obj_file.is_file():
            raise Exception(
                "Requested non-existent file '{}'".format(obj_file)
            )

        specific_args = []
        if surface_sampling:
            norm_params_file = (data_io.ws.get_dir(
                'sdf_norm_params_dir') / f[1:]).with_suffix('.npz')
            specific_args = ["-n", str(norm_params_file)]

        meshes_targets_and_specific_args.append(
            (
                str(obj_file),
                str(sdf_file),
                specific_args,
            )
        )

    if len(meshes_targets_and_specific_args) == 0:
        return

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(16)
    ) as executor:
        for (
                mesh_filepath,
                target_filepath,
                specific_args,
        ) in meshes_targets_and_specific_args:
            print(mesh_filepath, target_filepath, specific_args)
            executor.submit(
                process_mesh,
                mesh_filepath,
                target_filepath,
                executable,
                specific_args + additional_general_args,
            )

    return


if __name__ == '__main__':
    print(Path.cwd())
    logging.basicConfig(level=logging.INFO)

    train_data = True
    if train_data:
        data_split_file = '/home/syao/Program/Source/New3D/data_split_config/dfaust/dfaust_50002_train.json'
    else:
        data_split_file = '/home/syao/Program/Source/New3D/data_split_config/dfaust/dfaust_50002_test.json'

    data_dir = '/home/syao/Program/Experiments/N3D/SDF_FAMILY/Data'

    base_dir = Path(data_dir)
    sub_dir = {
        # 'geo_instance_dir': 'GeoInstances',
        'sdf_samples_dir': 'SdfSamples',
        'sdf_surface_dir': 'SurfaceSamples',
        'sdf_norm_params_dir': 'NormalizationParameters',
    }
    data_io_dir_config = {
        'base_dir': base_dir,
        'sub_dir': sub_dir,
    }
    data_io = Handled_Data_IO(dir_config=data_io_dir_config,
                              split_file=data_split_file)

    # print('Transform the obj data into pos with edge_index...')
    # handle_obj(source_dir=Path('/home/syao/Program/Datasets/'),
    #            target_dir=data_io.ws.get_dir('geo_instance_dir'),
    #            files=data_io.filenames)

    dataset_dir = Path('/DATA/')
    if train_data:
        print('Processing SDF sampling for train...')
        sdf_prepreocess(source_dir=dataset_dir,
                        data_io=data_io,
                        files=data_io.filenames)
    else:
        print('Processing SDF sampling for test...')
        sdf_prepreocess(source_dir=dataset_dir,
                        data_io=data_io,
                        files=data_io.filenames,
                        test_sampling=True)

        print('Processing SDF surface sampling...')
        sdf_prepreocess(source_dir=dataset_dir,
                        data_io=data_io,
                        files=data_io.filenames,
                        surface_sampling=True)

    print('Done!')