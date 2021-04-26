import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing


class vcoconv(MessagePassing):
    def __init__(self, in_channels, out_channels, weight_num=6):
        super().__init__(aggr='add', flow="target_to_source", node_dim=-2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_num = weight_num
        weights = nn.Parameter(torch.randn(weight_num, out_channels*in_channels))
        self.register_parameter("weights", weights)

        bias = nn.Parameter(torch.zeros(out_channels))
        self.register_parameter("bias", bias)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weights, 0, 0.1)
        # torch.nn.init.orthogonal_(self.weights)
        return

    def forward(self, x, edge_index, A_coes):
        # x: [BxNxC], edge_index: [2xE], A_coes: [ExW]
        e_size = A_coes.size(0)
        co = F.normalize(A_coes, dim=-1, p=1)
        # co = F.softmax(A_coes, dim=-1)
        w_compos = torch.matmul(co, self.weights)
        w_compos = w_compos.reshape(e_size, self.out_channels, self.in_channels) #[ExOxI]
        result = self.propagate(edge_index=edge_index, x=x, w_compos=w_compos)
        return result

    def message(self, x_j, w_compos):
        agg_wx = torch.einsum('eij, bej->bei', [w_compos, x_j])
        return agg_wx

    def update(self, aggr_out):
        # aggr_out: [NxBxC]
        return aggr_out+self.bias

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__,
                                       self.in_channels,
                                       self.out_channels,
                                       self.weight_num)
