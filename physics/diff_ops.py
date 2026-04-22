import torch
import torch.nn as nn
import torch.nn.functional as F


class FastPhysicsEvaluator(nn.Module):
    """
    基于固定权重 F.conv2d 的极速物理偏导数计算器。
    它作为替代求解器的“判官”，不参与网络自身的梯度构建，专门用于极速计算物理残差 (PDE Loss)。
    """

    def __init__(self, dx=1.0, dy=1.0, device='cpu'):
        super().__init__()
        self.dx = dx
        self.dy = dy

        # 1. 一阶中心差分卷积核 (求速度梯度 ∂u/∂x, ∂u/∂y)
        kernel_x = torch.tensor([
            [0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3) / self.dx

        kernel_y = torch.tensor([
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3) / self.dy

        # 2. 二阶拉普拉斯卷积核 (求粘性扩散项 ∂²u/∂x² + ∂²u/∂y²)
        kernel_lap = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3) / (self.dx * self.dy)

        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)
        self.register_buffer('weight_lap', kernel_lap)

    def partial_x(self, field):
        padded = F.pad(field, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(padded, self.weight_x)

    def partial_y(self, field):
        padded = F.pad(field, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(padded, self.weight_y)

    def laplacian(self, field):
        padded = F.pad(field, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(padded, self.weight_lap)

    def compute_ns_momentum_residual(self, u, v, p_x=0, p_y=0, nu=1e-3):
        if u.dim() == 3:
            u = u.unsqueeze(1)
            v = v.unsqueeze(1)

        u_x = self.partial_x(u)
        u_y = self.partial_y(u)
        v_x = self.partial_x(v)
        v_y = self.partial_y(v)

        conv_u = u * u_x + v * u_y
        conv_v = u * v_x + v * v_y

        diff_u = nu * self.laplacian(u)
        diff_v = nu * self.laplacian(v)

        res_x = conv_u - diff_u + p_x
        res_y = conv_v - diff_v + p_y

        return torch.mean(res_x ** 2) + torch.mean(res_y ** 2)

    def compute_surrogate_fluid_residual(self, field, nu=1e-3):
        """
        计算 2D 广义非线性流体残差 (包含对流项和扩散项)
        field: 重建出的 2D 物理场 (Batch, 1, X, Y)
        nu: 运动粘度系数
        """
        # 1. 计算一阶空间导数和拉普拉斯
        w_x = self.partial_x(field)
        w_y = self.partial_y(field)
        lap = self.laplacian(field)

        # 2. 构造非线性对流项 (Non-linear Convection)
        convection = field * w_x + field * w_y

        # 3. 构造扩散项 (Diffusion)
        diffusion = nu * lap

        # 4. 物理残差: Convection - Diffusion
        residual = convection - diffusion
        return residual