import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """残差块，包含卷积层、批归一化层和跳跃连接"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample  # 如果输入和输出通道数不同，使用下采样调整

    def forward(self, x):
        identity = x  # 保存输入用于跳跃连接

        if self.downsample is not None:
            identity = self.downsample(x)  # 下采样调整输入大小

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 将卷积输出与输入相加（跳跃连接）
        out = self.relu(out)

        return out

class MyModel(nn.Module):
    """基于残差网络的模型，从耳朵编码直接预测频率数 * 球谐系数的矩阵"""

    def __init__(self, num_freqs, num_sh_coeffs):
        super(MyModel, self).__init__()

        self.num_freqs = num_freqs
        self.num_sh_coeffs = num_sh_coeffs

        # 初始卷积层，调整通道数
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        # 构建残差层
        self.layer1 = self._make_layer(
            block=ResidualBlock,
            in_channels=64,
            out_channels=128,
            blocks=2,
            stride=2
        )
        self.layer2 = self._make_layer(
            block=ResidualBlock,
            in_channels=128,
            out_channels=256,
            blocks=2,
            stride=2
        )
        self.layer3 = self._make_layer(
            block=ResidualBlock,
            in_channels=256,
            out_channels=512,
            blocks=2,
            stride=2
        )

        # 全局平均池化层，降低特征维度
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 全连接层，输出预测结果
        self.fc = nn.Linear(512, num_freqs * num_sh_coeffs)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """构建残差层"""

        downsample = None
        if stride != 1 or in_channels != out_channels:
            # 当需要改变通道数或步幅时，进行下采样
            downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, ear_encoding):
        """
        前向传播函数

        参数:
        - ear_encoding: 输入的耳朵编码，形状为 (batch_size, 1, length)

        返回:
        - x: 预测的 HRTF 球谐系数矩阵，形状为 (batch_size, num_freqs, num_sh_coeffs)
        """

        if ear_encoding.dim() == 2:
            ear_encoding = ear_encoding.unsqueeze(1)

        # 初始卷积和激活
        x = self.conv1(ear_encoding)
        x = self.bn1(x)
        x = self.relu(x)

        # 通过残差层提取特征
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 全局平均池化，输出形状为 (batch_size, 512, 1)
        x = self.avgpool(x)

        # 展平特征，形状为 (batch_size, 512)
        x = x.view(x.size(0), -1)

        # 全连接层，输出预测结果，形状为 (batch_size, num_freqs * num_sh_coeffs)
        x = self.fc(x)

        # 调整输出形状为 (batch_size, num_freqs, num_sh_coeffs)
        x = x.view(x.size(0), self.num_freqs, self.num_sh_coeffs)

        return x
