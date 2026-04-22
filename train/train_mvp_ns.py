import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import matplotlib.pyplot as plt

# 确保可以找到项目根目录下的包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import get_dataloaders
from models.pi_vpod_net import PI_VPOD_Net
from physics.pde_loss import PIVPODLoss


# 如果你在前一步新建了 eval/visualize_field.py，可以取消注释下面这行，每5轮静默画一次流场图
# from eval.visualize_field import plot_and_save_field

def plot_loss_curve(history, current_epoch=None):
    """静默绘制并保存 Loss 曲线图，绝不弹窗阻塞训练"""
    if not history['total']:
        return

    fig = plt.figure(figsize=(10, 6))
    plt.plot(history['total'], label='Total Loss')
    plt.plot(history['data'], label='Data Loss (MSE)')
    plt.plot(history['pde'], label='PDE Constraint Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    title = 'PI-VPOD-Net Training Convergence'
    if current_epoch is not None:
        title += f' (Up to Epoch {current_epoch})'
    plt.title(title)

    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    save_path = "../data/processed/training_loss_curve.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)  # 核心：画完立刻在后台销毁画布，绝不弹窗！


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 50
    learning_rate = 1e-3
    pod_basis_path = "../data/processed/offline_pod_basis.pt"
    mat_data_path = "../data/raw/navier_stokes/NavierStokes_V1e-3_N5000_T50/ns_V1e-3_N5000_T50.mat"

    history = {'total': [], 'data': [], 'pde': [], 'kl': []}

    train_loader, test_loader = get_dataloaders(mat_data_path, batch_size=batch_size)

    model = PI_VPOD_Net(input_dim=4096, pod_basis_path=pod_basis_path).to(device)
    criterion = PIVPODLoss(pde_weight=1.0, kl_weight=1e-5, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n🚀 开始训练 PI-VPOD-Net (深度增强版) | 设备: {device}")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0, 'data': 0, 'pde': 0, 'kl': 0}

        for i, (batch_g, batch_u) in enumerate(train_loader):
            batch_g = batch_g.to(device).view(batch_g.size(0), -1)
            batch_u = batch_u.to(device)

            optimizer.zero_grad()
            pred_field, b_nom, mu, logvar = model(batch_g)
            loss, l_data, l_kl, l_pde = criterion(pred_field, batch_u, mu, logvar)

            loss.backward()
            optimizer.step()

            epoch_losses['total'] += loss.item()
            epoch_losses['data'] += l_data.item()
            epoch_losses['pde'] += l_pde.item()
            epoch_losses['kl'] += l_kl.item()

        num_batches = len(train_loader)
        history['total'].append(epoch_losses['total'] / num_batches)
        history['data'].append(epoch_losses['data'] / num_batches)
        history['pde'].append(epoch_losses['pde'] / num_batches)
        history['kl'].append(epoch_losses['kl'] / num_batches)

        print(f"Epoch [{epoch + 1}/{epochs}] Avg Loss: {history['total'][-1]:.6f} "
              f"(Data: {history['data'][-1]:.6f}, PDE: {history['pde'][-1]:.6f})")

        # 👇 每次训练完一轮，静悄悄地更新硬盘里的 Loss 图，绝不弹窗
        plot_loss_curve(history, current_epoch=epoch + 1)

        if (epoch + 1) % 5 == 0:
            model.eval()
            test_mse = 0
            with torch.no_grad():
                for batch_g, batch_u in test_loader:
                    batch_g = batch_g.to(device).view(batch_g.size(0), -1)
                    batch_u = batch_u.to(device)
                    pred_field, _, _, _ = model(batch_g)
                    test_mse += torch.mean((pred_field - batch_u) ** 2).item()
            print(f"      >>>> 测试集 MSE: {test_mse / len(test_loader):.8f}")

            # 如果你导入了 plot_and_save_field，取消注释下面这行来顺便存流场图
            # try:
            #     plot_and_save_field(model, test_loader, device, epoch=epoch+1)
            # except NameError:
            #     pass

    # 训练结束，保存最终模型权重！
    os.makedirs("../data/processed", exist_ok=True)
    torch.save(model.state_dict(), "../data/processed/pi_vpod_net_ns.pth")
    print("\n💾 训练完成！模型权重已保存至: ../data/processed/pi_vpod_net_ns.pth")


if __name__ == "__main__":
    train()