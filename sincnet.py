import torch, math
import torch.nn.functional as F
import numpy as np

import logging
from typing import List, Tuple, Union

import torch
import torch.optim as optim

def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

def to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

class SincLayer(torch.nn.Module):
    def __init__(self, device):
        super(SincLayer, self).__init__()
        self.out_channels = 80 # 첫번째 레이어의 컨볼루션 필터(즉, 싱크 함수) 개수
        self.kernel_size = 251 # 싱크 함수의 길이  L
        self.sample_rate = 16000 # SincNet이 주로 보는 주파수 영역대
        self.in_channels = 1 # 첫번째 레이어의 입력 채널 수, raw wave form이 입력되기 때문에 채널 수는 1이 됩니다.
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.bias = False
        self.groups = 1
        self.min_low_hz = 50
        self.min_band_hz = 50
        self.low_hz = 30  # 싱크 함수가 보는 주파수 구간의 lower bound.
        self.high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz) # 싱크 함수가 보는 주파수 구간의 upper bound.
        self.device = device

    def forward(self, x):
        mel = np.linspace(to_mel(self.low_hz),
                    to_mel(self.high_hz),
                    self.out_channels + 1)
        hz = to_hz(mel)

        low_hz_ = torch.nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        band_hz_ = torch.nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
        n = (self.kernel_size - 1) / 2.0
        n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

        low = self.min_low_hz + torch.abs(low_hz_)
        high = torch.clamp(input=low + self.min_band_hz + torch.abs(band_hz_),
                        min=self.min_low_hz,
                        max=self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, n_)
        f_times_t_high = torch.matmul(high, n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (n_ / 2)) * window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        filters = (band_pass).view(self.out_channels, 1, self.kernel_size).to(self.device)

        # 이후 이 80개 필터들이 각각 입력 raw wave에 대해 1d conv 실시
        sincnet_result = F.conv1d(x, filters, stride=self.stride,
                                padding=self.padding, dilation=self.dilation,
                                bias=None, groups=1)
        
        return sincnet_result

class SincNet(torch.nn.Module):
    def __init__(self, device, num_classes):
        super(SincNet, self).__init__()

        self.sinc = SincLayer(device)
        self.pool = torch.nn.MaxPool1d(2, ceil_mode=True)
        self.norm = torch.nn.LayerNorm((80, 9875))
        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(0.2)
        self.convs = torch.nn.Sequential(
            torch.nn.Conv1d(80, 64, kernel_size=80),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(64, 10, kernel_size=80)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(97170, 300),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(300, num_classes),
        )
        self.device=device
        
    def forward(self,x):
        out = self.sinc(x)
        out = self.pool(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.convs(out)
        out = self.classifier(out)
        return out
        
