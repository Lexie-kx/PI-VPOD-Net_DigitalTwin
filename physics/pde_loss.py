import torch
import torch.nn as nn
from .diff_ops import FastPhysicsEvaluator


class PIVPODLoss(nn.Module):
    def __init__(self, pde_weight=1.0, kl_weight=1e-5, device='cpu'):
        super().__init__()
        self.pde_weight = pde_weight
        self.kl_weight = kl_weight
        self.mse_loss = nn.MSELoss()

        self.physics_evaluator = FastPhysicsEvaluator(dx=1.0, dy=1.0, device=device)

    def forward(self, pred_field, target_field, mu, logvar):
        # 1. 数据重构损失 (Data Loss)
        loss_data = self.mse_loss(pred_field, target_field)

        # 2. VIB KL 散度损失 (KL Loss)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        loss_kl = torch.mean(loss_kl)

        # 3. 物理残差损失 (PDE Loss): 引入强非线性流体约束
        diff_residual = self.physics_evaluator.compute_surrogate_fluid_residual(pred_field, nu=1e-3)
        loss_pde = torch.mean(diff_residual ** 2)

        # 4. 总损失加权融合
        total_loss = loss_data + self.kl_weight * loss_kl + self.pde_weight * loss_pde

        return total_loss, loss_data, loss_kl, loss_pde