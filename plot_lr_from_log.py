import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, w=5):
    if w <= 1 or len(x) < w:
        return np.array(x, dtype=float)
    x = np.array(x, dtype=float)
    kernel = np.ones(w) / w
    y = np.convolve(x, kernel, mode="valid")
    pad = [np.nan] * (w - 1)
    return np.array(pad + y.tolist(), dtype=float)


def parse_log(log_path, fold_id=1):
    pattern = re.compile(
        r"Fold\s+(\d+),\s+Epoch\s+\[(\d+)/(\d+)\],.*?Val LSD Recon Raw:\s*([0-9.]+),\s*LR:\s*([0-9.eE+-]+)"
    )

    epochs = []
    val_lsd_raw = []
    lrs = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            this_fold = int(m.group(1))
            if this_fold != fold_id:
                continue

            epoch = int(m.group(2))
            lsd_raw = float(m.group(4))
            lr = float(m.group(5))

            epochs.append(epoch)
            val_lsd_raw.append(lsd_raw)
            lrs.append(lr)

    if len(epochs) == 0:
        raise ValueError(f"没有解析到 Fold {fold_id} 的数据，请检查日志格式或 fold_id")

    return np.array(epochs), np.array(val_lsd_raw), np.array(lrs)


def find_lr_drop_epochs(epochs, lrs, tol=1e-15):
    drop_epochs = []
    for i in range(1, len(lrs)):
        if lrs[i] < lrs[i - 1] - tol:
            drop_epochs.append(int(epochs[i]))
    return drop_epochs


def plot_curve(log_path, out_path, fold_id=1, smooth_window=5):
    epochs, val_lsd_raw, lrs = parse_log(log_path, fold_id=fold_id)
    val_lsd_raw_smooth = moving_average(val_lsd_raw, w=smooth_window)
    drop_epochs = find_lr_drop_epochs(epochs, lrs)

    best_idx = int(np.argmin(val_lsd_raw))
    best_epoch = int(epochs[best_idx])
    best_val = float(val_lsd_raw[best_idx])

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(epochs, val_lsd_raw, color="#1f77b4", alpha=0.35, linewidth=1.2, label="Val LSD Raw")
    ax1.plot(epochs, val_lsd_raw_smooth, color="#1f77b4", linewidth=2.0, label=f"Val LSD Raw MA({smooth_window})")
    ax1.scatter([best_epoch], [best_val], color="#1f77b4", s=40, zorder=5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val LSD Raw", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, color="#d62728", linewidth=1.8, label="Learning Rate")
    ax2.set_yscale("log")
    ax2.set_ylabel("Learning Rate (log scale)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    for ep in drop_epochs:
        ax1.axvline(ep, color="gray", linestyle="--", alpha=0.25, linewidth=1)

    title = (
        f"Fold {fold_id} | Val LSD Raw & LR\n"
        f"Best Val LSD Raw = {best_val:.4f} at Epoch {best_epoch}"
    )
    plt.title(title)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    print(f"已保存图像: {out_path}")
    print(f"最优 Val LSD Raw: {best_val:.6f}, Epoch: {best_epoch}")
    print(f"LR 下降发生在 Epoch: {drop_epochs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="训练日志路径")
    parser.add_argument("--out", default="lr_val_lsd_curve.png", help="输出图片路径")
    parser.add_argument("--fold", type=int, default=1, help="绘制哪个 fold")
    parser.add_argument("--smooth", type=int, default=5, help="LSD 平滑窗口")
    args = parser.parse_args()

    plot_curve(
        log_path=args.log,
        out_path=args.out,
        fold_id=args.fold,
        smooth_window=args.smooth
    )