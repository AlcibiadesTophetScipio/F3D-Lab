import torch
import torch.nn as nn
import torch.nn.functional as F

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

class loss_siren(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sdf_lambda = 3e3       # 1e4  # 3e3
        self.inter_lambda = 1e2     # 1e2  # 1e3
        self.normal_lambda = 1e2    # 1e2
        self.grad_lambda = 5e1      # 1e1

    def forward(self,
                gt_sdf,
                gt_normals,
                coords,
                pred_sdf,
                ):
        grad = gradient(pred_sdf, coords)

        # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
        sdf_constraint = torch.where(gt_sdf != -1, pred_sdf,
                                     torch.zeros_like(pred_sdf))
        inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf),
                                       torch.exp(-1e2 * torch.abs(pred_sdf)))
        normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
                                        torch.zeros_like(grad[..., :1]))
        grad_constraint = torch.abs(grad.norm(dim=-1) - 1)

        loss_sdf = torch.abs(sdf_constraint).mean()
        loss_inter = inter_constraint.mean()
        loss_normal = normal_constraint.mean()
        loss_grad = grad_constraint.mean()

        loss = loss_sdf * self.sdf_lambda \
               + loss_inter * self.inter_lambda \
               + loss_normal * self.normal_lambda \
               + loss_grad * self.grad_lambda

        return {'loss': loss,
                'sdf_term':  loss_sdf.item(),
                'inter_term':  loss_inter.item(),
                'normal_term':  loss_normal.item(),
                'grad_term':  loss_grad.item(),
                }