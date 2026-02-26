import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入自定义的数据集和模型
from dataset import CustomDataset
from ear_SH_model2 import MyModel
import scipy.io as sio
import numpy as np
import random

from config import (
    batch_size,
    default_seed,
    learning_rate,
    log_file,
    num_epochs,
    num_individuals,
    shvec_path,
)

from utils import calLSD

# ------------------------------
# 设置随机种子以确保可重复性
# ------------------------------
def set_seed(seed=default_seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ------------------------------
# 设置设备（GPU 或 CPU）
# ------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def build_train_val_test_split(num_individuals, val_ratio=0.1, test_ratio=0.1, seed=default_seed):
    indices = np.arange(num_individuals)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    test_size = max(1, int(round(num_individuals * test_ratio)))
    val_size = max(1, int(round(num_individuals * val_ratio)))

    if test_size + val_size >= num_individuals:
        test_size = 1
        val_size = 1

    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def train_once(num_individuals, log_file=log_file):
    """
    执行单次训练/验证/测试划分

    参数:
    - num_individuals: 数据集中个体的数量
    - log_file: 记录训练过程的日志文件

    返回:
    - final_train_loss: 最后一轮训练损失
    - final_val_loss: 最后一轮验证损失
    - test_loss: 测试损失
    - test_lsd_recon_smooth: 测试集 LSD 重建平滑
    - test_lsd_recon_raw: 测试集 LSD 重建原始
    """
    set_seed()

    effective_num_individuals = num_individuals - 2
    if effective_num_individuals <= 2:
        raise ValueError("Not enough subjects after excluding first and last individuals")

    train_idx, val_idx, test_idx = build_train_val_test_split(effective_num_individuals)
    split_message = (
        f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)} "
        f"(excluded first/last artificial-head subjects)"
    )
    print(split_message)

    all_lsd_recon_raw_f = []

    with open(log_file, "w") as f:
        f.write(split_message + "\n")

        # 加载数据集
        train_dataset = CustomDataset(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, split="train")
        val_dataset = CustomDataset(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, split="val")
        test_dataset = CustomDataset(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, split="test")

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 获取样本数据以确定输出维度
        sample_z_ear, sample_hrtf_sh, _, _ = train_dataset[0]
        num_freqs = sample_hrtf_sh.shape[0]        # 频率数
        num_sh_coeffs = sample_hrtf_sh.shape[1]    # 球谐系数数

        # 实例化模型
        model = MyModel(num_freqs=num_freqs, num_sh_coeffs=num_sh_coeffs).to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()                                # 均方误差损失函数
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 优化器

        final_train_loss = np.nan
        final_val_loss = np.nan
        final_val_lsd_recon_smooth = np.nan
        final_val_lsd_recon_raw = np.nan
        best_val_loss = float("inf")
        best_epoch = -1
        best_state_dict = None

        # 训练循环
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for z_ear, hrtf_sh, hrtf_amp, subject in train_loader:
                z_ear = z_ear.float().to(device)
                hrtf_sh = hrtf_sh.float().to(device)

                hrtf_sh_pred = model(z_ear)
                loss = criterion(hrtf_sh_pred, hrtf_sh)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            final_train_loss = running_loss / len(train_loader)

            # 在验证集上评估模型
            model.eval()
            val_loss = 0.0
            val_lsd_recon_smooth = 0.0
            val_lsd_recon_raw = 0.0

            with torch.no_grad():
                for z_ear, hrtf_sh, hrtf_amp, subject in val_loader:
                    z_ear = z_ear.float().to(device)
                    hrtf_sh = hrtf_sh.float().to(device)
                    hrtf_amp = hrtf_amp.float().to(device)

                    hrtf_sh_pred = model(z_ear)
                    loss = criterion(hrtf_sh_pred, hrtf_sh)

                    lsd_smooth, lsd_raw, _ = calLSD(
                        hrtf_sh_pred,
                        hrtf_sh,
                        hrtf_amp,
                        shvec_path=shvec_path,
                    )

                    val_loss += loss.item()
                    val_lsd_recon_smooth += lsd_smooth.mean().item()
                    val_lsd_recon_raw += lsd_raw.mean().item()

            final_val_loss = val_loss / len(val_loader)
            final_val_lsd_recon_smooth = val_lsd_recon_smooth / len(val_loader)
            final_val_lsd_recon_raw = val_lsd_recon_raw / len(val_loader)

            log_message = (
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f}, "
                f"Val LSD Recon Smooth: {final_val_lsd_recon_smooth:.4f}, "
                f"Val LSD Recon Raw: {final_val_lsd_recon_raw:.4f}"
            )
            print(log_message)
            f.write(log_message + "\n")

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_epoch = epoch + 1
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        if best_state_dict is None:
            raise RuntimeError("No valid best model found from validation")

        model.load_state_dict(best_state_dict)
        best_model_message = f"Using best model from epoch {best_epoch} with Val Loss {best_val_loss:.4f} for testing"
        print(best_model_message)
        f.write(best_model_message + "\n")

        # 在测试集上评估最终模型
        model.eval()
        test_loss_total = 0.0
        test_lsd_recon_smooth_total = 0.0
        test_lsd_recon_raw_total = 0.0

        with torch.no_grad():
            for z_ear, hrtf_sh, hrtf_amp, subject in test_loader:
                z_ear = z_ear.float().to(device)
                hrtf_sh = hrtf_sh.float().to(device)
                hrtf_amp = hrtf_amp.float().to(device)

                hrtf_sh_pred = model(z_ear)
                loss = criterion(hrtf_sh_pred, hrtf_sh)

                lsd_smooth, lsd_raw, lsd_recon_raw_f_single = calLSD(
                    hrtf_sh_pred,
                    hrtf_sh,
                    hrtf_amp,
                    shvec_path=shvec_path,
                )

                test_loss_total += loss.item()
                test_lsd_recon_smooth_total += lsd_smooth.mean().item()
                test_lsd_recon_raw_total += lsd_raw.mean().item()
                all_lsd_recon_raw_f.append(lsd_recon_raw_f_single)

        test_loss = test_loss_total / len(test_loader)
        test_lsd_recon_smooth = test_lsd_recon_smooth_total / len(test_loader)
        test_lsd_recon_raw = test_lsd_recon_raw_total / len(test_loader)

        all_lsd_recon_raw_f_tensor = torch.cat(all_lsd_recon_raw_f, dim=0)

        final_log_message = (
            f"\nFinal Train Loss: {final_train_loss:.4f}, "
            f"Final Val Loss: {final_val_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Mean LSD Recon Smooth: {test_lsd_recon_smooth:.4f}, "
            f"Test Mean LSD Recon Raw: {test_lsd_recon_raw:.4f}"
        )
        print(final_log_message)
        f.write(final_log_message + "\n")

    return (
        final_train_loss,
        final_val_loss,
        test_loss,
        test_lsd_recon_smooth,
        test_lsd_recon_raw,
        all_lsd_recon_raw_f_tensor,
    )

if __name__ == "__main__":
    final_train_loss, final_val_loss, test_loss, test_lsd_recon_smooth, test_lsd_recon_raw, all_lsd_recon_raw_f_tensor = train_once(num_individuals)
    # # 保存为 .mat 文件
    # sio.savemat('tet.mat', {'all_lsd_recon_raw_f_tensor_8order': all_lsd_recon_raw_f_tensor.cpu().numpy()})
    # print("All LSD Recon Raw F Tensor Shape:", all_lsd_recon_raw_f_tensor.shape)
