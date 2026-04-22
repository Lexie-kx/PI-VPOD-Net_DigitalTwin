import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import scipy.io as sio
import numpy as np


class NavierStokesDataset(Dataset):
    """
    针对 Navier-Stokes MVP 数据集的 PyTorch Dataset。
    一次性加载到内存 (3.58GB 现代电脑完全无压力)，实现极速的 Batch 切片读取。
    """

    def __init__(self, mat_path: str, is_train: bool = True, n_train: int = 4000, target_time_step: int = -1):
        super().__init__()
        self.is_train = is_train
        self.n_train = n_train

        print(f"正在将数据集加载至内存 ({'训练集' if is_train else '测试集'}) ...")

        try:
            # 读取 h5py 格式 (MAT v7.3)
            with h5py.File(mat_path, 'r') as f:
                # 提取输入参数 g (初始条件 a) -> h5py 读出形状通常相反，需转置
                a_data = np.array(f['a']).transpose(2, 1, 0)  # 调整为 (5000, 64, 64)
                # 提取目标流场 u
                u_data = np.array(f['u']).transpose(3, 2, 1, 0)  # 调整为 (5000, 64, 64, 50)
        except OSError:
            # 降级读取老版本 scipy.io
            mat = sio.loadmat(mat_path)
            a_data = mat['a']
            u_data = mat['u']

        # 转换为 Float32 张量
        a_tensor = torch.tensor(a_data, dtype=torch.float32)
        u_tensor = torch.tensor(u_data, dtype=torch.float32)

        # 划分训练集与测试集
        if self.is_train:
            self.inputs = a_tensor[:self.n_train]  # Shape: (4000, 64, 64)
            self.targets = u_tensor[:self.n_train]  # Shape: (4000, 64, 64, 50)
        else:
            self.inputs = a_tensor[self.n_train:]  # Shape: (1000, 64, 64)
            self.targets = u_tensor[self.n_train:]  # Shape: (1000, 64, 64, 50)

        # 为连续铸造预留：通常工业只关注最终的稳态流场。
        # 这里我们取 N-S 方程演化到最后一步 (t=50) 的状态作为目标拟合场。
        # 取切片后 Target Shape 变为 (Batch, 64, 64)
        self.targets = self.targets[:, :, :, target_time_step]

        # 增加通道维度 (Channel)，适配卷积网络: (Batch, 1, X, Y)
        self.inputs = self.inputs.unsqueeze(1)
        self.targets = self.targets.unsqueeze(1)

        print(f"✅ 数据集就绪! 输入特征 $g$ 形状: {self.inputs.shape}, 目标场形状: {self.targets.shape}")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # 返回: 工艺参数 g, 真实高维流场
        return self.inputs[idx], self.targets[idx]


def get_dataloaders(mat_path, batch_size=32, n_train=4000):
    """
    一键获取 训练集 和 测试集 的 DataLoader
    """
    train_dataset = NavierStokesDataset(mat_path, is_train=True, n_train=n_train)
    test_dataset = NavierStokesDataset(mat_path, is_train=False, n_train=n_train)

    # 训练集打乱 (shuffle=True)，测试集不打乱
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader


# ----------------- 简单测试逻辑 -----------------
if __name__ == "__main__":
    mat_file = "../data/raw/navier_stokes/NavierStokes_V1e-3_N5000_T50/ns_V1e-3_N5000_T50.mat"

    train_loader, test_loader = get_dataloaders(mat_file, batch_size=16)

    # 抽一个 Batch 看看
    for batch_g, batch_u in train_loader:
        print(f"\n成功抽取一个 Batch!")
        print(f"Batch Input  (工艺参数 g)   Shape: {batch_g.shape}")
        print(f"Batch Target (高保真流场 u) Shape: {batch_u.shape}")
        break