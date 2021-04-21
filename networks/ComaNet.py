import torch
import torch.nn.functional as F

from layers.conv import chebconv_coma
from layers.pool import pool_coma

class ComaNet(torch.nn.Module):
    def __init__(self,
                 downsample_matrices,
                 upsample_matrices,
                 adjacency_matrices,
                 filters,
                 K,
                 num_nodes,
                 z_dim=8):
        super().__init__()
        self.d_mat = downsample_matrices
        self.u_mat = upsample_matrices
        self.filters = filters
        self.edge_index, self.A_norm = zip(*[chebconv_coma.norm(adjacency_matrices[i]._indices(),
                                                                num_nodes[i]) for i in range(len(num_nodes))])

        self.cheb_convs = torch.nn.ModuleList([chebconv_coma(self.filters[i], self.filters[i+1], K[i])
                                               for i in range(len(self.filters)-1)])
        assert len(downsample_matrices) == len(self.cheb_convs)

        self.cheb_deconvs = torch.nn.ModuleList([chebconv_coma(self.filters[-i-1], self.filters[-i-2], K[-i])
                                                for i in range(len(self.filters) - 1)])
        assert len(upsample_matrices) == len(self.cheb_deconvs)
        self.cheb_deconvs[-1].bias = None
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
        for i in range(len(self.cheb_convs)):
            x = F.relu(self.cheb_convs[i](x, self.edge_index[i], self.A_norm[i]))
            x = self.pool(x, self.d_mat[i])

        x = x.reshape(x.shape[0], self.enc.in_features)
        out = self.enc(x)
        return out

    def decoder(self, x):
        x = self.dec(x)
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(len(self.cheb_deconvs)-1):
            x = self.pool(x, self.u_mat[-i-1])
            x = F.relu(self.cheb_deconvs[i](x, self.edge_index[-i-2], self.A_norm[-i-2]))
        x = self.pool(x, self.u_mat[0])
        out = self.cheb_deconvs[-1](x, self.edge_index[0], self.A_norm[0])
        return out

    def forward(self, data):
        x = data['x']
        x = self.encoder(x)
        x = self.decoder(x)
        re_x = x

        return {"re_x": re_x}