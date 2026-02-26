import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import datetime, pickle, os, zipfile
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as sio
import gc

from config import BASE_DIR, file_numbers_path, hrtf_path, num_individuals

class CustomDataset(Dataset):
    def __init__(self, train_idx, val_idx, test_idx, split="train"):
        super(CustomDataset, self).__init__()
        self.train_idx = np.array(train_idx, dtype=int)
        self.val_idx = np.array(val_idx, dtype=int)
        self.test_idx = np.array(test_idx, dtype=int)
        self.split = split
        self.left_or_right = 0   # 选择左耳或右耳
        # 加载人体参数矩阵（SubjectID + 头部与左耳参数）
        anthropometric_path = os.path.join(BASE_DIR, "Data", "AntrhopometricMeasures.csv")
        if os.path.exists(anthropometric_path):
            anthropometric_data = np.genfromtxt(
                anthropometric_path,
                delimiter=",",
                skip_header=1
            )
        else:
            print("anthropometric_data not found")
            raise FileNotFoundError(f"{anthropometric_path} not found")

        if anthropometric_data.ndim == 1:
            anthropometric_data = anthropometric_data.reshape(1, -1)

        subject_ids = anthropometric_data[:, 0].astype(int)
        feature_matrix = anthropometric_data[:, 1:26]

        # 加载具有网格的个体索引，并对人体参数做同样的主体筛选
        fileNumbers = sio.loadmat(file_numbers_path)["fileNumbers"]  # 代表了加载顺序
        fileNumbers = fileNumbers.flatten()
        fileNumbers = np.sort(fileNumbers) - 1    # 顺序
        if fileNumbers.shape[0] <= 2:
            raise ValueError("Not enough subjects after fileNumbers loading to exclude first and last individuals")
        # 去除第一个和最后一个个体（人工头）
        fileNumbers = fileNumbers[1:-1]
        feature_matrix = feature_matrix[fileNumbers]

        # 基于当前划分的训练主体计算统计量，避免信息泄漏
        train_features = feature_matrix[self.train_idx]

        mu = np.mean(train_features, axis=0)
        sigma = np.std(train_features, axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)

        # 进行 sigmoid 归一化
        feature_matrix = 1.0 / (1.0 + np.exp(-((feature_matrix - mu) / sigma)))

        img_encode_matrix = torch.from_numpy(feature_matrix).float()

        self.img_encode_matrix_train = img_encode_matrix[self.train_idx]
        self.img_encode_matrix_val = img_encode_matrix[self.val_idx]
        self.img_encode_matrix_test = img_encode_matrix[self.test_idx]

        # 频率索引
        # freq_logind = np.arange(17, 17+2*38, 2)
        freq_logind = np.arange(1, 1+43*2, 2)

        # 加载sh系数
        sh_mat = sio.loadmat(hrtf_path)["hrtf_SHT_dBmat"]  # 96 128 25 2
        sh_mat = sh_mat[fileNumbers]
        sh_mat = sh_mat[:, freq_logind, :, :]

        # 划分训练集、验证集和测试集
        self.sht_mat_train = sh_mat[self.train_idx]
        self.sht_mat_val = sh_mat[self.val_idx]
        self.sht_mat_test = sh_mat[self.test_idx]

        # 加载测量的HRTF
        measured_hrtf = sio.loadmat(hrtf_path)["hrtf_freq_allDB"]  # 96 128 440 2
        measured_hrtf = measured_hrtf[fileNumbers]
        measured_hrtf = measured_hrtf[:, freq_logind, :, :]

        # 划分训练集、验证集和测试集
        self.measured_hrtf_train = measured_hrtf[self.train_idx]
        self.measured_hrtf_val = measured_hrtf[self.val_idx]
        self.measured_hrtf_test = measured_hrtf[self.test_idx]

    def __len__(self):
        if self.split == "train":
            return self.sht_mat_train.shape[0]
        if self.split == "val":
            return self.sht_mat_val.shape[0]
        if self.split == "test":
            return self.sht_mat_test.shape[0]
        raise ValueError(f"Unsupported split: {self.split}")

    def __getitem__(self, idx):
        if self.split == "train":
            subject = idx % self.sht_mat_train.shape[0]
            z_ear = self.img_encode_matrix_train[subject, :]
            hrtf_sh = self.sht_mat_train[subject, :, :, self.left_or_right]
            hrtf_amp = self.measured_hrtf_train[subject, :, :, self.left_or_right]
            subject_id = int(self.train_idx[subject])
        elif self.split == "val":
            subject = idx % self.sht_mat_val.shape[0]
            z_ear = self.img_encode_matrix_val[subject, :]
            hrtf_sh = self.sht_mat_val[subject, :, :, self.left_or_right]
            hrtf_amp = self.measured_hrtf_val[subject, :, :, self.left_or_right]
            subject_id = int(self.val_idx[subject])
        elif self.split == "test":
            subject = idx % self.sht_mat_test.shape[0]
            z_ear = self.img_encode_matrix_test[subject, :]
            hrtf_sh = self.sht_mat_test[subject, :, :, self.left_or_right]
            hrtf_amp = self.measured_hrtf_test[subject, :, :, self.left_or_right]
            subject_id = int(self.test_idx[subject])
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        # 数据转换为张量
        z_ear = z_ear.float()  # 确保 z_ear 为浮点数张量
        hrtf_sh = torch.from_numpy(hrtf_sh).float()    # 转换为浮点数张量
        hrtf_amp = torch.from_numpy(hrtf_amp).float()  # 转换为浮点数张量

        return z_ear, hrtf_sh, hrtf_amp, subject_id
