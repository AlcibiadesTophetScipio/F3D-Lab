import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing

class co_norm(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', flow="target_to_source", node_dim=-2)

    def forward(self, edge_index, co):
        return self.propagate(edge_index=edge_index, co=co)

    def message(self, co):
        return co.abs()

class pool_vdco(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edges):
        super().__init__(aggr='add', flow="target_to_source", node_dim=-2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edges = num_edges

        co_density = nn.Parameter(torch.randn(num_edges))
        self.register_parameter("co_density", co_density)
        self.co_norm = co_norm()

        if in_channels == out_channels:
            self.linear = None
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        co_norm = self.co_norm(edge_index=edge_index, co=self.co_density)
        x = self.propagate(edge_index=edge_index, x=x, conorm=co_norm, co=self.co_density)
        return x

    def message(self, x_j, co, conorm_j):
        if self.linear:
            agg = self.linear(x_j)
        else:
            agg = x_j
        return agg * (co.abs()/conorm_j).view(1,-1,1)

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.num_edges)
