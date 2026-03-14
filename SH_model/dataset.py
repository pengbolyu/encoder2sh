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

from config import BASE_DIR, encode_path, file_numbers_path, hrtf_path, num_individuals

class CustomDataset(Dataset):
    def __init__(self, train_idx, val_idx, test_idx, split="train"):
        super(CustomDataset, self).__init__()
        self.train_idx = np.array(train_idx, dtype=int)
        self.val_idx = np.array(val_idx, dtype=int)
        self.test_idx = np.array(test_idx, dtype=int)
        self.split = split
        self.left_or_right = 0   # 选择左耳或右耳
        # 加载耳部图片编码矩阵
        if os.path.exists(encode_path):
            img_encode_matrix = torch.load(encode_path, map_location="cpu")
        else:
            print("img_encode_matrix not found")
            raise FileNotFoundError(f"{encode_path} not found")

        if isinstance(img_encode_matrix, np.ndarray):
            img_encode_matrix = torch.from_numpy(img_encode_matrix)

        # 加载具有网格的个体索引，并对耳部编码做同样的主体筛选
        fileNumbers = sio.loadmat(file_numbers_path)["fileNumbers"]  # 代表了加载顺序
        fileNumbers = fileNumbers.flatten()
        fileNumbers = np.sort(fileNumbers).astype(np.int64) - 1    # 顺序
        img_encode_matrix = img_encode_matrix.float()
        if img_encode_matrix.shape[0] != fileNumbers.shape[0]:
            raise ValueError(
                f"Before filtering: img_encode_matrix.shape[0]={img_encode_matrix.shape[0]} "
                f"!= fileNumbers.shape[0]={fileNumbers.shape[0]}"
            )

        # 去除第一个和最后一个个体（人工头）以及 fileNumbers 中编号 33 的个体
        # 注意：fileNumbers 已执行 -1，因此编号 33 对应值为 32。
        # 既支持单个 int，也支持列表/元组
        remove_subject_ids = [33]   # 或者 remove_subject_ids = 33
        remove_subject_idxs = np.atleast_1d(remove_subject_ids).astype(np.int64) - 1

        keep_mask = np.ones(fileNumbers.shape[0], dtype=np.bool_)
        keep_mask[0] = False
        keep_mask[-1] = False
        keep_mask[np.isin(fileNumbers, remove_subject_idxs)] = False

        fileNumbers = fileNumbers[keep_mask]
        img_encode_matrix = img_encode_matrix[keep_mask]
        assert img_encode_matrix.shape[0] == fileNumbers.shape[0], (
            f"img_encode_matrix.shape[0]={img_encode_matrix.shape[0]} != fileNumbers.shape[0]={fileNumbers.shape[0]}"
        )

        self.img_encode_matrix_train = img_encode_matrix[self.train_idx]
        self.img_encode_matrix_val = img_encode_matrix[self.val_idx]
        self.img_encode_matrix_test = img_encode_matrix[self.test_idx]

        # 频率索引
        # freq_logind = np.arange(17, 17+2*38, 2)
        freq_logind = np.arange(1, 1+43*2, 2)
        self.fileNumbers_filter = fileNumbers

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
            subject_id = int(self.fileNumbers_filter[self.train_idx[subject]] + 1)  # +1 返回实际ID
        elif self.split == "val":
            subject = idx % self.sht_mat_val.shape[0]
            z_ear = self.img_encode_matrix_val[subject, :]
            hrtf_sh = self.sht_mat_val[subject, :, :, self.left_or_right]
            hrtf_amp = self.measured_hrtf_val[subject, :, :, self.left_or_right]
            subject_id = int(self.fileNumbers_filter[self.val_idx[subject]] + 1)
        elif self.split == "test":
            subject = idx % self.sht_mat_test.shape[0]
            z_ear = self.img_encode_matrix_test[subject, :]
            hrtf_sh = self.sht_mat_test[subject, :, :, self.left_or_right]
            hrtf_amp = self.measured_hrtf_test[subject, :, :, self.left_or_right]
            subject_id = int(self.fileNumbers_filter[self.test_idx[subject]] + 1)
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        # 数据转换为张量
        z_ear = z_ear.float()  # 确保 z_ear 为浮点数张量
        hrtf_sh = torch.from_numpy(hrtf_sh).float()    # 转换为浮点数张量
        hrtf_amp = torch.from_numpy(hrtf_amp).float()  # 转换为浮点数张量

        return z_ear, hrtf_sh, hrtf_amp, subject_id
