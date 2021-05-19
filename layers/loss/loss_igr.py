import torch
import torch.nn as nn
from torch.autograd import grad

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, :, -3:]
    return points_grad


class loss_igr(nn.Module):
    def __init__(self,
                 grad_lambda=1e-1,
                 normals_lambda=1.0,
                 latent_lambda=1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.grad_lambda = grad_lambda
        self.normals_lambda = normals_lambda
        self.latent_lambda = latent_lambda

    def forward(self,
                mnfld_pnts,
                mnfld_pred,
                nonmnfld_pnts,
                nonmnfld_pred,
                latent=None,
                normals=None
                ):

        # compute grad
        mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
        nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

        # manifold loss
        mnfld_loss = (mnfld_pred.abs()).mean()

        # eikonal loss
        grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
        loss = mnfld_loss + self.grad_lambda * grad_loss

        # normals loss
        if normals is None:
            normals_loss = torch.tensor([1.0])
        else:
            normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=-1).mean()
            loss = loss + self.normals_lambda * normals_loss

        # latent reg loss
        if latent is None:
            latent_loss = torch.tensor([0.0])
        else:
            latent_loss = latent.norm(dim=-1).mean()
            loss = loss + self.latent_lambda * latent_loss

        return {'loss':loss,
                'sdf_term': mnfld_loss.item(),
                'grad_term':grad_loss.item(),
                'normals_term':normals_loss.item(),
                'latent_term': latent_loss.item()
                }
