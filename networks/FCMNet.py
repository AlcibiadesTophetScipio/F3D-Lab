import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.conv import vcoconv
from layers.pool import pool_vdco, pool_coma

class FCMNet(torch.nn.Module):
    def __init__(self,
                 downsample_matrices,
                 upsample_matrices,
                 adjacency_matrices,
                 filters,
                 num_nodes,
                 z_dim=8,
                 weight_num=16):
        super().__init__()
        self.d_mat = downsample_matrices
        self.u_mat = upsample_matrices
        self.filters = filters
        self.adj_edge_index = [adjacency_matrices[i]._indices() for i in range(len(num_nodes))]
        self.vco_convs = nn.ModuleList([vcoconv(self.filters[i], self.filters[i+1], weight_num=weight_num)
                                        for i in range(len(self.filters)-1)])
        self.vd_downres = nn.ModuleList([pool_vdco(self.filters[i], self.filters[i + 1], self.adj_edge_index[i].size(1))
                                         for i in range(len(self.filters) - 1)])
        self.vco_deconvs = nn.ModuleList([vcoconv(self.filters[-i-1], self.filters[-i-2], weight_num=weight_num)
                                          for i in range(len(self.filters) - 1)])
        self.vd_upwnres = nn.ModuleList([pool_vdco(self.filters[-i - 1], self.filters[-i - 2], self.adj_edge_index[-i - 2].size(1))
                                         for i in range(len(self.filters) - 1)])
        assert len(downsample_matrices) == len(self.vco_convs)
        assert len(upsample_matrices) == len(self.vco_deconvs)

        self.edge_cos = nn.ParameterList([])
        for i in range(len(num_nodes)):
            edge_cos = torch.randn(self.adj_edge_index[i].size(1), weight_num)
            edge_cos = nn.Parameter(edge_cos)
            self.edge_cos.append(edge_cos)

        self.pool = pool_coma()

        self.enc = torch.nn.Linear(self.d_mat[-1].shape[0] * self.filters[-1], z_dim)
        self.dec = torch.nn.Linear(z_dim, self.filters[-1]*self.u_mat[-1].shape[1])
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec.weight, 0, 0.1)
        torch.nn.init.constant_(self.enc.bias, 0)
        torch.nn.init.constant_(self.dec.bias, 0)

    def encoder(self, x):
        for i in range(len(self.vco_convs)):
            # x_res = self.vd_downres[i](x, self.adj_edge_index[i])
            x = F.relu(self.vco_convs[i](x, self.adj_edge_index[i], self.edge_cos[i]))
            x = self.pool(x, self.d_mat[i])
            # x += x_res

        x = x.reshape(x.shape[0], self.enc.in_features)
        out = self.enc(x)
        return out

    def decoder(self, x):
        x = self.dec(x)
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(len(self.vco_deconvs)-1):
            x = self.pool(x, self.u_mat[-i-1])
            # x_res = self.gbc_upres[i](x, self.adj_edge_index[-i-1])
            x = F.relu(self.vco_deconvs[i](x, self.adj_edge_index[-i-2], self.edge_cos[-i-2]))
            # x += x_res

        x = self.pool(x, self.u_mat[0])
        out = self.vco_deconvs[-1](x, self.adj_edge_index[0], self.edge_cos[0])
        return out

    def forward(self, data):
        x = data['x']
        x = self.encoder(x)
        x = self.decoder(x)
        re_x = x

        return {"re_x": re_x}