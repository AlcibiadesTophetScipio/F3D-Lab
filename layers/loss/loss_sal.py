import torch
import torch.nn as nn

class loss_sal(nn.Module):
    def __init__(self, latent_lambda=1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_lambda = latent_lambda

    def forward(self,
                nonmanifold_pnts_pred,
                nonmanifold_gt,
                latent_reg=None
                ):
        recon_term = torch.abs(nonmanifold_pnts_pred.squeeze().abs() - nonmanifold_gt).mean()
        loss = recon_term
        if latent_reg is not None:
            reg_term = latent_reg.mean()
            loss = loss + self.latent_lambda * reg_term
        else:
            reg_term = torch.tensor([0.0])

        return {"loss": loss,
                'dist_term': recon_term.item(),
                'latent_term': reg_term.item()
                }
