import torch
import torch.nn as nn

class loss_sal(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self,
                nonmanifold_pnts_pred,
                nonmanifold_gt,
                latent_reg=None
                ):
        recon_term = torch.abs(nonmanifold_pnts_pred.squeeze().abs() - nonmanifold_gt)
        loss = recon_term.mean()
        if latent_reg is not None:
            loss = loss + latent_reg.mean()
            reg_term = latent_reg.mean().detach()
        else:
            reg_term = torch.tensor([0.0])
        return {"loss": loss, 'recon_term': recon_term.mean(), 'reg_term': reg_term.mean()}
