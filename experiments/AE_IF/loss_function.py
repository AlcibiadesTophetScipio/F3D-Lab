import torch
import torch.nn.functional as F

def gradient(inputs, outputs):
    grad_outputs = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True)[0]
    return grad

class mcif_loss_v1(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 2021-05-02
        # self.sdf_lambda = 1e1
        # self.dist_lambda = 1
        # self.latent_lambda = 1e-4
        # self.normal_lambda = 1e-3
        # self.invs_lambda = 1e-3
        # self.grad_lambda = 5e-4

        # 2021-05-04
        # self.sdf_lambda = 1.0
        # self.dist_lambda = 1.0
        # self.latent_lambda = 0.0
        # self.normal_lambda = 0.0
        # self.invs_lambda = 0.0
        # self.grad_lambda = 0.0

        # v2 05-04
        # self.sdf_lambda = 1.0
        # self.dist_lambda = 1.0
        # self.latent_lambda = 0.0
        # self.normal_lambda = 1e-3
        # self.invs_lambda = 0.0
        # self.grad_lambda = 5e-4

        # v2 05-05
        self.sdf_lambda = 1.0
        self.dist_lambda = 1.0
        self.latent_lambda = 0.0
        self.normal_lambda = 1e-3
        self.invs_lambda = 1e-3
        self.grad_lambda = 5e-4

    def forward(self,
                mf_pnts,
                pred_mf_sdf,
                mf_norm,
                nmf_pnts,
                pred_nmf_sdf,
                nmf_dist,
                sdf_latent,
                dist_eps=1e-6):
        # manifold loss (base)
        mf_loss = (pred_mf_sdf.abs()).mean()

        # normal loss (IGR)
        mf_grad = gradient(inputs=mf_pnts, outputs=pred_mf_sdf)
        normals_loss_dist = ((mf_grad - mf_norm).abs()).norm(2, dim=-1).mean()
        normals_loss = (1 - F.cosine_similarity(mf_grad, mf_norm, dim=-1)).mean()

        # eikonal loss (IGR)
        nmf_grad = gradient(inputs=nmf_pnts, outputs=pred_nmf_sdf)
        grad_loss = ((nmf_grad.norm(2, dim=-1) - 1) ** 2).mean()

        # distance loss (SAL)
        nmf_dist_clamp = torch.clamp_max(nmf_dist, 0.1)
        dist_loss = torch.abs(pred_nmf_sdf.abs() - nmf_dist_clamp).mean()

        # (Siren) put the pred distance of nomanifold points away from the surface
        invs_loss = torch.where(pred_nmf_sdf.abs()>dist_eps, torch.zeros_like(pred_nmf_sdf),
                                torch.exp(-1e2 * torch.abs(pred_nmf_sdf)))
        invs_loss = invs_loss.mean()

        # latent reg
        latent_loss = sdf_latent.norm(dim=-1, p=2).sum()

        loss_total = mf_loss * self.sdf_lambda \
                     + dist_loss * self.dist_lambda \
                     + normals_loss * self.normal_lambda \
                     + invs_loss * self.invs_lambda \
                     + grad_loss * self.grad_lambda \
                     + latent_loss * self.latent_lambda

        return loss_total, {
                            'sdf_term': mf_loss.item(),
                            'dist_term': dist_loss.item(),
                            'latent_term': latent_loss.item(),
                            'normal_term': normals_loss.item(),
                            'normal_dist_term': normals_loss_dist.item(),
                            'invs_term': invs_loss.item(),
                            'grad_term': grad_loss.item(),
                            }

class mcif_loss_v2(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # t3 05-07
        self.sdf_lambda = 1.0
        self.dist_lambda = 1.0
        self.latent_lambda = 0.0
        self.normal_lambda = 1e-3
        self.grad_lambda = 5e-4

    def forward(self,
                mf_pnts,
                pred_mf_sdf,
                mf_norm,
                nmf_pnts,
                pred_nmf_sdf,
                nmf_dist,
                sdf_latent,
                dist_eps=1e-6):
        # manifold loss (base)
        mf_loss = (pred_mf_sdf.abs()).mean()

        # normal loss (IGR)
        mf_grad = gradient(inputs=mf_pnts, outputs=pred_mf_sdf)
        normals_loss_dist = ((mf_grad - mf_norm).abs()).norm(2, dim=-1).mean()
        normals_loss = (1 - F.cosine_similarity(mf_grad, mf_norm, dim=-1)).mean()

        # eikonal loss (IGR)
        nmf_grad = gradient(inputs=nmf_pnts, outputs=pred_nmf_sdf)
        grad_loss = ((nmf_grad.norm(2, dim=-1) - 1) ** 2).mean()

        # distance loss (SAL)
        # nmf_dist_clamp = torch.clamp_max(nmf_dist, 0.1)
        dist_loss = torch.abs(pred_nmf_sdf.abs() - nmf_dist).mean()

        # (Siren) put the pred distance of nomanifold points away from the surface
        # invs_loss = torch.where(pred_nmf_sdf.abs()>dist_eps, torch.zeros_like(pred_nmf_sdf),
        #                         torch.exp(-1e2 * torch.abs(pred_nmf_sdf)))
        # invs_loss = invs_loss.mean()

        # latent reg
        latent_loss = sdf_latent.norm(dim=-1, p=2).mean()

        loss_total = mf_loss * self.sdf_lambda \
                     + dist_loss * self.dist_lambda \
                     + normals_loss * self.normal_lambda \
                     + grad_loss * self.grad_lambda \
                     + latent_loss * self.latent_lambda

        return loss_total, {
                            'sdf_term': mf_loss.item(),
                            'dist_term': dist_loss.item(),
                            'latent_term': latent_loss.item(),
                            'normal_term': normals_loss.item(),
                            'normal_dist_term': normals_loss_dist.item(),
                            'grad_term': grad_loss.item(),
                            }