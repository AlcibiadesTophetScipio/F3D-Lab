
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian

import math

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class chebconv_coma(MessagePassing):
    def __init__(self, in_channels, out_channels, K,
                 bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='target_to_source', node_dim=-2, **kwargs)

        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    @staticmethod
    def norm(edge_index,
                 num_nodes,
                 edge_weight=None,
                 normalization='sym',
                 dtype=None
                 ):
        lambda_max = 2.0
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def forward(self, x, edge_index, norm):
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = torch.matmul(Tx_0, self.weight[0])

        # propagate_type: (x: Tensor, norm: Tensor)
        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(1, -1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0))