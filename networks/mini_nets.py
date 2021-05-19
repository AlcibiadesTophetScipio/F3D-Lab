import torch
import torch.nn as nn
import torch.functional as F

class NMF_InstanceNorm(nn.Module):
    '''
    Instance Normalization. Refer Section 3 in paper (Instance Normalization) for more details and Fig 5 for visual intuition.
    This is crucial for cross-category training.
    Paper: Neural Mesh Flow: 3D Manifold Mesh Generation via Diffeomorphic Flows
    '''

    def __init__(self, zdim):
        super(NMF_InstanceNorm, self).__init__()
        '''
        Initialization.
        '''
        self.norm = nn.Sequential(nn.Linear(zdim, 256), nn.ReLU(), nn.Linear(256, 3), nn.Sigmoid()).float()

    def forward(self, input, code):
        '''
        input: point cloud of shape BxNx3
        code: shape embedding of shape Bx1xzdim
        '''
        centered_input = input - torch.mean(input, axis=1, keepdim=True)  # Center the point cloud
        centered_input = centered_input * (
                    1 - self.norm(code))  # anisotropic scaling the point cloud w.r.t. the shape embedding

        return centered_input

class PointNet_Encoder(nn.Module):
    '''
    PointNet Encoder by Qi. et.al
    '''

    def __init__(self, zdim, input_dim=3):
        super(PointNet_Encoder, self).__init__()

        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, zdim, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(zdim)

        self.fc1 = nn.Linear(zdim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, zdim)

    def forward(self, x):
        '''
        Input: Nx#ptsx3
        Output: Nxzdim
        '''
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.zdim)

        ms = F.relu(self.fc_bn1(self.fc1(x)))
        ms = F.relu(self.fc_bn2(self.fc2(ms)))
        ms = self.fc3(ms)

        return ms