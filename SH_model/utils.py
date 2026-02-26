# Import libraries
import torch
import torch.nn as nn
import scipy.io as sio

def mse(first, second):
    """
    计算均方误差（Mean Squared Error, MSE）

    参数:
    - first: 第一个输入张量
    - second: 第二个输入张量

    返回:
    - mse: 均方误差
    """
    return (first.float() - second.float()) ** 2

def restore_hrtf(pre_sh_hrtf, shvec_path):
    """
    将预测的球谐系数张量还原为 HRTF

    参数:
    - pre_sh_hrtf: 预测的球谐系数张量，形状为 (batch_size, num_freqs, num_sh_coeffs)
    - shvec_path: 球谐矩阵文件路径

    返回:
    - hrtf: 还原后的 HRTF，形状为 (batch_size, num_positions, num_freqs)
    """
    # 加载球谐矩阵
    shvec = torch.from_numpy(sio.loadmat(shvec_path)["SHbase_P"])  # 形状为 (num_positions, num_sh_coeffs)
    shvec = shvec.float()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 将张量移到设备上
    shvec = shvec.to(device)
    pre_sh_hrtf = pre_sh_hrtf.to(device)
    
    # 还原 HRTF
    batch_size, num_freqs, num_sh_coeffs = pre_sh_hrtf.shape
    num_positions = shvec.shape[0]
    
    # 变换球谐系数张量的形状以便进行矩阵乘法
    pre_sh_hrtf = pre_sh_hrtf.permute(0, 2, 1)  # 形状变为 (batch_size, num_sh_coeffs, num_freqs)
    
    # 进行矩阵乘法还原 HRTF
    hrtf = torch.bmm(shvec.unsqueeze(0).repeat(batch_size, 1, 1), pre_sh_hrtf)  # 形状为 (batch_size, num_positions, num_freqs)
    
    return hrtf

def calLSD(pre_sh_hrtf, hrtf_sh, hrtf_amp, shvec_path):
    """
    计算 LSD（Log-Spectral Distortion）

    参数:
    - pre_sh_hrtf: 预测的 HRTF 球谐系数，形状为 (batch_size, num_freqs, num_sh_coeffs)
    - hrtf_sh: 真实的 HRTF 球谐系数，形状为 (batch_size, num_freqs, num_sh_coeffs)
    - hrtf_amp: 真实的 HRTF 幅度，形状为 (batch_size, num_freqs, num_positions)
    - shvec_path: 球谐矩阵文件路径

    返回:
    - lsd_recon_smooth: 平滑后的 LSD
    - lsd_recon_raw: 原始 HRTF 的 LSD
    """
    # 还原预测的 HRTF 和真实的 HRTF
    predicted_hrtf = restore_hrtf(pre_sh_hrtf, shvec_path)  # 形状为 (batch_size, num_positions, num_freqs)
    smoothed_hrtf = restore_hrtf(hrtf_sh, shvec_path)       # 形状为 (batch_size, num_positions, num_freqs)

    # 确保 hrtf_amp 在设备上
    device = predicted_hrtf.device
    hrtf_amp = hrtf_amp.to(device)

    # 调整 hrtf_amp 的形状
    hrtf_amp = hrtf_amp.permute(0, 2, 1)  # 形状变为 (batch_size, num_positions, num_freqs)

    # 计算 LSD
    lsd_recon_smooth = torch.sqrt(mse(predicted_hrtf, smoothed_hrtf).mean(dim=(1, 2)))
    lsd_recon_raw = torch.sqrt(mse(predicted_hrtf, hrtf_amp).mean(dim=(1, 2)))
    lsd_recon_raw_f = torch.sqrt(mse(predicted_hrtf, hrtf_amp).mean(dim=(1)))
    return lsd_recon_smooth, lsd_recon_raw, lsd_recon_raw_f

# 示例用法
if __name__ == "__main__":
    # 假设输入张量的形状
    batch_size = 5
    num_freqs = 41
    num_sh_coeffs = 64
    num_positions = 440  # 假设空间位置数量为 440

    # 生成示例数据
    pre_sh_hrtf = torch.rand(batch_size, num_freqs, num_sh_coeffs)
    hrtf_sh = torch.rand(batch_size, num_freqs, num_sh_coeffs)
    hrtf_amp = torch.rand(batch_size, num_freqs, num_positions)

    # # 计算 LSD
    # lsd_recon_smooth, lsd_recon_raw = calLSD(pre_sh_hrtf, hrtf_sh, hrtf_amp)

    # print("LSD 重建平滑:", lsd_recon_smooth)
    # print("LSD 重建原始:", lsd_recon_raw)
