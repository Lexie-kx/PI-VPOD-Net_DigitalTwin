import torch
import matplotlib.pyplot as plt
import sys
import os

# 把项目根目录加到系统路径，防止找不到 utils 和 models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import get_dataloaders
from models.pi_vpod_net import PI_VPOD_Net


def visualize_simulation_result():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("正在加载测试集数据和模型...")
    # 1. 加载测试集 (取 Batch Size = 1 方便单张可视化)
    mat_data_path = "../data/raw/navier_stokes/NavierStokes_V1e-3_N5000_T50/ns_V1e-3_N5000_T50.mat"
    _, test_loader = get_dataloaders(mat_data_path, batch_size=1, n_train=4000)

    # 2. 实例化模型并加载刚才保存的权重
    pod_basis_path = "../data/processed/offline_pod_basis.pt"
    model = PI_VPOD_Net(input_dim=4096, pod_basis_path=pod_basis_path).to(device)

    model_weight_path = "../data/processed/pi_vpod_net_ns.pth"
    if not os.path.exists(model_weight_path):
        print(f"❌ 找不到模型权重文件 {model_weight_path}，请确保 train_mvp_ns.py 已经跑完并保存了模型！")
        return

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()  # 开启测试模式

    # 3. 从测试集中抽取一个盲盒样本
    batch_g, batch_u = next(iter(test_loader))
    batch_g_flat = batch_g.to(device).view(1, -1)

    # 4. 让 AI 秒算流场
    print("AI 正在进行极速流场推演...")
    with torch.no_grad():
        pred_field, _, _, _ = model(batch_g_flat)

    # 转换回 numpy 以便画图
    true_field_np = batch_u[0, 0].cpu().numpy()
    ai_field_np = pred_field[0, 0].cpu().numpy()
    # 计算绝对误差场
    error_field = abs(true_field_np - ai_field_np)

    # 5. 绘制顶级的物理场对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 真值场
    im0 = axes[0].imshow(true_field_np, cmap='jet', origin='lower')
    axes[0].set_title("Ground Truth (CFD Simulation)", fontsize=14)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # AI 预测场
    im1 = axes[1].imshow(ai_field_np, cmap='jet', origin='lower')
    axes[1].set_title("AI Surrogate Prediction (PI-VPOD-Net)", fontsize=14)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 误差场
    im2 = axes[2].imshow(error_field, cmap='magma', origin='lower')
    axes[2].set_title("Absolute Error", fontsize=14)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # 保存结果图
    save_path = "../data/processed/simulation_result.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n🎉 仿真对比图已生成并保存至: {save_path}")

    plt.show()


if __name__ == "__main__":
    visualize_simulation_result()