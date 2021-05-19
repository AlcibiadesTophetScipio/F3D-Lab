import torch
import torch.nn.functional as F

# num_sdf_latent = 6890
# num_mf_pnts = 10000
# batch_size = 8
# mc_latent = torch.randn([batch_size, 256])
# sdf_latent = torch.randn([batch_size, num_sdf_latent, 128])
#
# mf_pnts = torch.randn([batch_size, num_mf_pnts, 3])
# mf_idx_gt = torch.empty([batch_size, num_mf_pnts], dtype=torch.long).random_(num_sdf_latent)
#
# idx_mf_input = torch.cat([mf_pnts, mc_latent.unsqueeze(1).repeat(1,num_mf_pnts,1)],dim=-1)
# mf_idx_pred = torch.randn([batch_size, num_mf_pnts, num_sdf_latent])
# mf_idx_pred = mf_idx_pred.view(-1, num_sdf_latent)
# loss_mf_idx = F.cross_entropy(mf_idx_pred, mf_idx_gt.view(-1))
#
# mf_idx_select = mf_idx_pred.max(dim=-1)[1].view(batch_size,-1)
# mf_sdf_latent = torch.cat([sdf_latent[i].index_select(dim=0, index=mf_idx_select[i]).unsqueeze(0)
#                                    for i in range(batch_size)], dim=0)
# mf_input = torch.cat([mf_pnts, mf_sdf_latent], dim=-1)

def tuple_test():
    return 1,2

re = tuple_test()

print('Done!')

