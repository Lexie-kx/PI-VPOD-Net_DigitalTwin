import torch
import torch.nn as nn


class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, hidden_dim)
        )

    def forward(self, g):
        return self.net(g)


class HeadA_Reconstruct(nn.Module):
    # 【修改点 1】：把输入的特征维度从之前的 128，变成了 belief_dim (8)
    def __init__(self, pod_basis_path, belief_dim=8, device='cpu'):
        super().__init__()
        pod_data = torch.load(pod_basis_path, map_location=device)
        self.register_buffer('V', pod_data['V'])
        self.register_buffer('mean_field', pod_data['mean_field'])
        self.grid_shape = pod_data['grid_shape']
        self.num_modes = self.V.shape[1]

        # 从 8 维的 belief 还原出 POD 系数
        self.fc_coef = nn.Linear(belief_dim, self.num_modes)

    def forward(self, b_nom):
        c = self.fc_coef(b_nom)
        field_flat = torch.matmul(c, self.V.T) + self.mean_field.T

        batch_size = b_nom.shape[0]
        field_2d = field_flat.view(batch_size, 1, *self.grid_shape)

        return field_2d


class HeadB_Belief(nn.Module):
    def __init__(self, hidden_dim, belief_dim):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, belief_dim)
        self.fc_logvar = nn.Linear(hidden_dim, belief_dim)

    def forward(self, features):
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        b_nom = mu + eps * std

        return b_nom, mu, logvar


class PI_VPOD_Net(nn.Module):
    def __init__(self, input_dim, pod_basis_path):
        super().__init__()
        hidden_dim = 128
        belief_dim = 8

        self.branch = BranchNet(input_dim, hidden_dim)
        # 【修改点 2】：把 belief_dim 传给 HeadA
        self.head_b = HeadB_Belief(hidden_dim, belief_dim)
        self.head_a = HeadA_Reconstruct(pod_basis_path, belief_dim=belief_dim)

    def forward(self, g):
        # 1. 提取基础特征
        features = self.branch(g)

        # 2. 【核心】：先通过 Head B 强制压缩出 8 维的 Nominal Belief
        b_nom, mu, logvar = self.head_b(features)

        # 3. 【核心】：再让 Head A 只能拿着这 8 维的 b_nom 去硬着头皮画流场！
        field_2d = self.head_a(b_nom)

        return field_2d, b_nom, mu, logvar