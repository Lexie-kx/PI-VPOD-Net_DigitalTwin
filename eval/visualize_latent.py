import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 把项目根目录加到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import get_dataloaders
from models.pi_vpod_net import PI_VPOD_Net


def visualize_latent_belief():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("正在潜入 AI 的隐空间 (Latent Space)...")

    # 1. 加载数据和模型 (取 Batch Size = 3，我们拿 3 个不同的流体样本做对比)
    mat_data_path = "../data/raw/navier_stokes/NavierStokes_V1e-3_N5000_T50/ns_V1e-3_N5000_T50.mat"
    _, test_loader = get_dataloaders(mat_data_path, batch_size=3, n_train=4000)

    pod_basis_path = "../data/processed/offline_pod_basis.pt"
    model = PI_VPOD_Net(input_dim=4096, pod_basis_path=pod_basis_path).to(device)

    model_weight_path = "../data/processed/pi_vpod_net_ns.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 2. 提取特征
    batch_g, _ = next(iter(test_loader))
    batch_g_flat = batch_g.to(device).view(3, -1)

    with torch.no_grad():
        # 我们这次不要重建的流场(第一个返回值)，只抓取它脑子里的 8 维信念特征
        _, b_nom, mu, logvar = model(batch_g_flat)

    # 转换成 numpy 数组方便画图
    # mu 是这 8 个维度的均值，std 是标准差 (代表 AI 的“不确定度”或者说“信息过滤程度”)
    mu_np = mu.cpu().numpy()
    std_np = torch.exp(0.5 * logvar).cpu().numpy()

    # 3. 绘制带有不确定性的特征柱状图
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("Latent Belief Extracted by $R_{T \to B}$ (The 8D Control Vector)", fontsize=16, fontweight='bold')

    x = np.arange(8)  # 8 个维度
    width = 0.5

    for i in range(3):
        ax = axes[i]
        # 画出均值柱子，并加上代表标准差（不确定度）的误差棒
        ax.bar(x, mu_np[i], width, yerr=std_np[i], capsize=5, color='coral', edgecolor='black', alpha=0.8)

        ax.set_ylabel(f"Sample {i + 1}\nActivation", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Dim {j + 1}" for j in range(8)])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linewidth=1)  # 画一条 0 基准线

    axes[-1].set_xlabel("Latent Dimensions (The 8 Nominal Belief Features)", fontsize=12)
    plt.tight_layout()

    save_path = "../data/processed/latent_belief_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n🧠 隐空间信念向量分析图已保存至: {save_path}")

    plt.show()


if __name__ == "__main__":
    visualize_latent_belief()