import os
import h5py
import scipy.io as sio
import numpy as np
import torch


def load_ns_data(file_path: str):
    """
    解析 N-S .mat 数据集，兼容 MAT v7.3 (h5py) 和老版本 (scipy.io)。
    """
    print(f"正在读取数据集: {file_path} ...")
    try:
        # 3.58GB 极大概率是 HDF5 (MAT v7.3) 格式
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())

            # 【核心修改点】：优先寻找并锁定表示全时序流场的 'u'
            if 'u' in keys:
                data_key = 'u'
            else:
                data_key = [k for k in keys if not k.startswith('#')][0]

            print(f"✅ 发现并锁定目标数据键: {data_key}")

            # h5py 读取的形状通常与 Matlab 相反，需要转置
            # 真实 'u' 在 h5py 中读出通常为 (T, Y, X, N) 即 (50, 64, 64, 5000)
            data = np.array(f[data_key])
            if data.ndim == 4:
                # 调整为 (N, X, Y, T)
                data = data.transpose(3, 2, 1, 0)
    except OSError:
        # 降级方案：使用 scipy.io
        print("降级使用 scipy.io 读取...")
        mat = sio.loadmat(file_path)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        data_key = 'u' if 'u' in keys else keys[0]
        data = mat[data_key]

    print(f"原始数据读取完成，当前目标流场形状 (N, X, Y, T): {data.shape}")
    return torch.tensor(data, dtype=torch.float32)

def extract_pod_basis(data_tensor: torch.Tensor, n_train: int = 4000, keep_modes: int = 64):
    """
    基于空间协方差矩阵提取 POD 基函数 (V)
    输入 data_tensor 形状: (N, X, Y, T)
    输出 V 形状: (X*Y, keep_modes)
    """
    # 1. 划分训练集
    train_data = data_tensor[:n_train]  # Shape: (4000, 64, 64, 50)
    N, X, Y, T = train_data.shape
    M = X * Y  # 空间节点数，如 4096
    S = N * T  # 快照总数，如 200000

    print(f"训练集划分完毕。准备展平...")
    print(f"空间节点数 M: {M}, 总快照数 S: {S}")

    # 2. 空间与时间/样本展平 (M, S)
    # 将 X, Y 展平为一维，将 N, T 展平为另一维
    snapshots = train_data.permute(1, 2, 0, 3).reshape(M, S)  # Shape: (4096, 200000)

    # 3. 计算均值并中心化 (为了严格的 POD，减去时间平均或总体平均)
    # 对于流场拟合，通常保留均值或减去整体均值场
    mean_field = torch.mean(snapshots, dim=1, keepdim=True)  # Shape: (4096, 1)
    snapshots_centered = snapshots - mean_field

    # 4. 计算空间协方差矩阵 C = A * A^T / S
    # 这里直接在 M 维度上做点乘，避免 (S, S) 的巨大矩阵运算
    print("正在计算空间协方差矩阵 (可利用 GPU 加速)...")
    if torch.cuda.is_available():
        snapshots_centered = snapshots_centered.cuda()

    C = torch.matmul(snapshots_centered, snapshots_centered.T) / S  # Shape: (4096, 4096)

    # 5. 特征值分解 (使用 eigh，因为协方差矩阵是实对称阵，比 SVD 更快更稳)
    print("正在进行特征值分解 (Eigen Decomposition)...")
    eigenvalues, eigenvectors = torch.linalg.eigh(C)

    # eigh 返回的特征值是升序的，需要反转为降序
    eigenvalues = eigenvalues.flip(dims=(0,))
    eigenvectors = eigenvectors.flip(dims=(1,))

    # 6. 截断保留前 keep_modes 个模态
    V = eigenvectors[:, :keep_modes]  # Shape: (4096, keep_modes)

    # 计算能量占比
    total_energy = torch.sum(eigenvalues)
    kept_energy = torch.sum(eigenvalues[:keep_modes])
    print(f"成功提取前 {keep_modes} 个 POD 基函数。")
    print(f"截断能量占比 (Cumulative Energy): {(kept_energy / total_energy * 100):.4f}%")

    if torch.cuda.is_available():
        V = V.cpu()
        mean_field = mean_field.cpu()

    return V, mean_field


if __name__ == "__main__":
    # 【已完全匹配你的截图路径】
    mat_path = "../data/raw/navier_stokes/NavierStokes_V1e-3_N5000_T50/ns_V1e-3_N5000_T50.mat"

    # 保存路径保持不变
    save_path = "../data/processed/offline_pod_basis.pt"

    if os.path.exists(mat_path):
        # 阶段一：解析 Tensor 形状
        full_tensor = load_ns_data(mat_path)

        # 阶段二：提取并保存固定 POD 基函数
        # 假设保留前 128 个模态
        V_basis, mean_field = extract_pod_basis(full_tensor, n_train=4000, keep_modes=128)

        # 保存基函数矩阵，供 PI-VPOD-Net 的 Head A 直接加载作为固定权重
        save_dict = {
            'V': V_basis,  # Shape: (4096, 128)
            'mean_field': mean_field,  # Shape: (4096, 1)
            'grid_shape': (full_tensor.shape[1], full_tensor.shape[2])  # (64, 64)
        }

        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(save_dict, save_path)
        print(f"\n✅ 离线基函数已成功保存至: {save_path}")
        print("🎉 第一步通关完毕！")
    else:
        print(f"❌ 错误：未找到数据文件 {mat_path}")
        print("请再检查一下资源管理器，确保文件名和路径严丝合缝！")