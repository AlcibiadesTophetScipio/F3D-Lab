
import torch
import torch_geometric


if __name__ == '__main__':
    print('PyCharm')
    print(torch.__name__, torch.__version__)
    print(torch_geometric.__name__, torch_geometric.__version__)

    print(f'cuda num: {torch.cuda.device_count()}')
    print(f'cuda version: {torch.version.cuda}')
