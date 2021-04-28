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
        only_inputs=True)[0][:, -3:]
    return points_grad


class loss_igr(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grad_lambda = 0.1
        self.normals_lambda = 1.0

    def forward(self,
                mnfld_pnts,
                mnfld_pred,
                nonmnfld_pnts,
                nonmnfld_pred,
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
        if normals:
            normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
            loss = loss + self.normals_lambda * normals_loss
        else:
            normals_loss = torch.zeros(1)

        return {'loss':loss,
                'grad_term':grad_loss.item(),
                'normals_term':normals_loss.item()}
