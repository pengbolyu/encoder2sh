import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
import shutil

# 导入自定义的数据集和模型
from dataset import CustomDataset
from ear_SH_model2 import MyModel
import scipy.io as sio
import numpy as np
import random

from config import (
    BASE_DIR,
    batch_size,
    default_seed,
    learning_rate,
    log_file,
    lr_scheduler_factor,
    lr_scheduler_patience,
    lr_warmup_epochs,
    min_learning_rate,
    num_epochs,
    num_individuals,
    run_dir,
    shvec_path,
    weight_decay,
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


def prepare_run_artifacts(log_file_name):
    code_dir = os.path.join(run_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    log_path = os.path.join(run_dir, log_file_name)

    # Snapshot current training code for reproducibility
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(src_dir):
        if filename.endswith(".py"):
            shutil.copy2(os.path.join(src_dir, filename), os.path.join(code_dir, filename))

    return run_dir, log_path


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


def build_kfold_splits(num_individuals, n_splits=10, val_ratio=0.1, seed=default_seed):
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if num_individuals < n_splits:
        raise ValueError("num_individuals must be >= n_splits")

    indices = np.arange(num_individuals)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    fold_test_indices = np.array_split(indices, n_splits)
    splits = []

    for fold_idx in range(n_splits):
        test_idx = fold_test_indices[fold_idx]
        remaining_idx = np.concatenate(
            [fold_test_indices[i] for i in range(n_splits) if i != fold_idx]
        )

        fold_rng = np.random.default_rng(seed + fold_idx + 1)
        fold_rng.shuffle(remaining_idx)

        val_size = max(1, int(round(num_individuals * val_ratio)))
        if val_size >= remaining_idx.shape[0]:
            val_size = max(1, remaining_idx.shape[0] - 1)

        val_idx = remaining_idx[:val_size]
        train_idx = remaining_idx[val_size:]

        if train_idx.shape[0] == 0:
            raise ValueError("Empty train split in fold construction")

        splits.append((train_idx.tolist(), val_idx.tolist(), test_idx.tolist()))

    return splits


def cross_validate_train(num_individuals, log_file=log_file, n_splits=10, val_ratio=0.1):
    """
    执行 K 折交叉验证训练/验证/测试

    参数:
    - num_individuals: 数据集中个体的数量
    - log_file: 记录训练过程的日志文件

    返回:
    - cv_train_loss_mean: 各折最终训练损失均值
    - cv_val_loss_mean: 各折最终验证损失均值
    - cv_test_loss_mean: 各折测试损失均值
    - cv_test_lsd_recon_smooth_mean: 各折测试 LSD 重建平滑均值
    - cv_test_lsd_recon_raw_mean: 各折测试 LSD 重建原始均值
    """
    set_seed()

    effective_num_individuals = num_individuals - 2
    if effective_num_individuals <= 2:
        raise ValueError("Not enough subjects after excluding first and last individuals")

    fold_splits = build_kfold_splits(
        effective_num_individuals,
        n_splits=n_splits,
        val_ratio=val_ratio,
        seed=default_seed,
    )

    cv_message = (
        f"Cross validation -> folds: {n_splits}, val_ratio: {val_ratio}, "
        f"subjects: {effective_num_individuals} (excluded first/last artificial-head subjects)"
    )
    print(cv_message)

    run_dir, log_path = prepare_run_artifacts(log_file)
    all_lsd_recon_raw_f = []
    fold_train_losses = []
    fold_val_losses = []
    fold_test_losses = []
    fold_test_lsd_smooth = []
    fold_test_lsd_raw = []
    fold_val_lsd_smooth = []
    fold_val_lsd_raw = []

    with open(log_path, "w") as f:
        f.write(cv_message + "\n")
        f.write(f"Run directory: {run_dir}\n")

        for fold_id, (train_idx, val_idx, test_idx) in enumerate(fold_splits, start=1):
            fold_header = (
                f"\n[Fold {fold_id}/{n_splits}] "
                f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
            )
            print(fold_header)
            f.write(fold_header + "\n")

            train_dataset = CustomDataset(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, split="train")
            val_dataset = CustomDataset(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, split="val")
            test_dataset = CustomDataset(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, split="test")

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            sample_z_ear, sample_hrtf_sh, _, _ = train_dataset[0]
            num_freqs = sample_hrtf_sh.shape[0]
            num_sh_coeffs = sample_hrtf_sh.shape[1]

            model = MyModel(num_freqs=num_freqs, num_sh_coeffs=num_sh_coeffs).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                min_lr=min_learning_rate,
            )

            final_train_loss = np.nan
            final_val_loss = np.nan
            final_val_lsd_recon_smooth = np.nan
            final_val_lsd_recon_raw = np.nan
            best_val_loss = float("inf")
            best_val_lsd_raw_metric = float("inf")
            best_epoch = -1
            best_state_dict = None
            best_val_lsd_smooth = None
            best_val_lsd_raw = None

            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0

                if lr_warmup_epochs > 0 and epoch < lr_warmup_epochs:
                    warmup_lr = learning_rate * float(epoch + 1) / float(lr_warmup_epochs)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = warmup_lr

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

                # Monitor validation LSD Raw because the final target metric is LSD on test.
                if lr_warmup_epochs <= 0 or epoch >= lr_warmup_epochs:
                    scheduler.step(final_val_lsd_recon_raw)

                current_lr = optimizer.param_groups[0]["lr"]

                log_message = (
                    f"Fold {fold_id}, Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f}, "
                    f"Val LSD Recon Smooth: {final_val_lsd_recon_smooth:.4f}, "
                    f"Val LSD Recon Raw: {final_val_lsd_recon_raw:.4f}, "
                    f"LR: {current_lr:.8f}"
                )
                print(log_message)
                f.write(log_message + "\n")

                if final_val_lsd_recon_raw < best_val_lsd_raw_metric:
                    best_val_loss = final_val_loss
                    best_val_lsd_raw_metric = final_val_lsd_recon_raw
                    best_epoch = epoch + 1
                    best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                    best_val_lsd_smooth = final_val_lsd_recon_smooth
                    best_val_lsd_raw = final_val_lsd_recon_raw

            if best_state_dict is None:
                raise RuntimeError(f"No valid best model found from validation in fold {fold_id}")

            if best_val_lsd_smooth is None or best_val_lsd_raw is None:
                raise RuntimeError(f"No valid best val LSD found in fold {fold_id}")

            model.load_state_dict(best_state_dict)
            best_model_message = (
                f"Fold {fold_id}: using best model from epoch {best_epoch} "
                f"with Val LSD Raw {best_val_lsd_raw:.4f} for testing"
            )
            print(best_model_message)
            f.write(best_model_message + "\n")

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

            fold_train_losses.append(final_train_loss)
            fold_val_losses.append(best_val_loss)
            fold_val_lsd_smooth.append(best_val_lsd_smooth)
            fold_val_lsd_raw.append(best_val_lsd_raw)
            fold_test_losses.append(test_loss)
            fold_test_lsd_smooth.append(test_lsd_recon_smooth)
            fold_test_lsd_raw.append(test_lsd_recon_raw)

            fold_summary = (
                f"Fold {fold_id} Summary -> Train Loss: {final_train_loss:.4f}, "
                f"Val Loss(best): {best_val_loss:.4f}, Val LSD Raw(best): {best_val_lsd_raw:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Test LSD Smooth: {test_lsd_recon_smooth:.4f}, Test LSD Raw: {test_lsd_recon_raw:.4f}"
            )
            print(fold_summary)
            f.write(fold_summary + "\n")

        all_lsd_recon_raw_f_tensor = torch.cat(all_lsd_recon_raw_f, dim=0)

        cv_train_loss_mean = float(np.mean(fold_train_losses))
        cv_val_loss_mean = float(np.mean(fold_val_losses))
        cv_test_loss_mean = float(np.mean(fold_test_losses))
        cv_test_lsd_recon_smooth_mean = float(np.mean(fold_test_lsd_smooth))
        cv_test_lsd_recon_raw_mean = float(np.mean(fold_test_lsd_raw))
        cv_val_lsd_recon_smooth_mean = float(np.mean(fold_val_lsd_smooth))
        cv_val_lsd_recon_raw_mean = float(np.mean(fold_val_lsd_raw))

        cv_train_loss_std = float(np.std(fold_train_losses))
        cv_val_loss_std = float(np.std(fold_val_losses))
        cv_test_loss_std = float(np.std(fold_test_losses))
        cv_test_lsd_recon_smooth_std = float(np.std(fold_test_lsd_smooth))
        cv_test_lsd_recon_raw_std = float(np.std(fold_test_lsd_raw))
        cv_val_lsd_recon_smooth_std = float(np.std(fold_val_lsd_smooth))
        cv_val_lsd_recon_raw_std = float(np.std(fold_val_lsd_raw))

        final_log_message = (
            f"\nCV Final ({n_splits}-fold):\n"
            f"Train Loss: {cv_train_loss_mean:.4f} ± {cv_train_loss_std:.4f}\n"
            f"Val Loss: {cv_val_loss_mean:.4f} ± {cv_val_loss_std:.4f}\n"
            f"Val LSD Smooth: {cv_val_lsd_recon_smooth_mean:.4f} ± {cv_val_lsd_recon_smooth_std:.4f}\n"
            f"Val LSD Raw: {cv_val_lsd_recon_raw_mean:.4f} ± {cv_val_lsd_recon_raw_std:.4f}\n"
            f"Test Loss: {cv_test_loss_mean:.4f} ± {cv_test_loss_std:.4f}\n"
            f"Test LSD Smooth: {cv_test_lsd_recon_smooth_mean:.4f} ± {cv_test_lsd_recon_smooth_std:.4f}\n"
            f"Test LSD Raw: {cv_test_lsd_recon_raw_mean:.4f} ± {cv_test_lsd_recon_raw_std:.4f}"
        )
        print(final_log_message)
        f.write(final_log_message + "\n")

    return (
        cv_train_loss_mean,
        cv_val_loss_mean,
        cv_test_loss_mean,
        cv_test_lsd_recon_smooth_mean,
        cv_test_lsd_recon_raw_mean,
        all_lsd_recon_raw_f_tensor,
    )

if __name__ == "__main__":
    cv_train_loss, cv_val_loss, cv_test_loss, cv_test_lsd_recon_smooth, cv_test_lsd_recon_raw, all_lsd_recon_raw_f_tensor = cross_validate_train(
        num_individuals,
        n_splits=10,
        val_ratio=0.1,
    )
    # # 保存为 .mat 文件
    # sio.savemat('tet.mat', {'all_lsd_recon_raw_f_tensor_8order': all_lsd_recon_raw_f_tensor.cpu().numpy()})
    # print("All LSD Recon Raw F Tensor Shape:", all_lsd_recon_raw_f_tensor.shape)
    lsd_out_path = os.path.join(run_dir, "all_lsd_recon_raw_f_tensor.mat")
    sio.savemat(lsd_out_path, {
        "all_lsd_recon_raw_f_tensor": all_lsd_recon_raw_f_tensor.cpu().numpy()
    })