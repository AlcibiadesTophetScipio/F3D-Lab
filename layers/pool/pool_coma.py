
from torch_geometric.nn.conv import MessagePassing


class pool_coma(MessagePassing):
    def __init__(self):
        # Notice this change in the new version of Geometric
        '''
        Since the trans matrix is represented as (i,j), the flow must set as 'target_to_source'
        '''
        super(pool_coma, self).__init__(flow='target_to_source', node_dim=-2)

    def forward(self, x, pool_mat):
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(),
                             size=pool_mat.transpose(0,1).size())
        return out

    def message(self, x_j, norm):
        return norm.view(1, -1, 1) * x_j