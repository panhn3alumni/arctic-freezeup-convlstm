#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unet_model.py —— 纯空间 UNet (跨架构对照模型)

与 ConvLSTM (SeaIcePredictor) 的对照点:
  - 完全没有时间记忆 (无 LSTM/无自回归): 输入窗口 10 天 x C 通道拍平成 2D 卷积通道
  - 一次性输出全部 10 个预报时效 (one-shot multi-lead)
  - 其余协议 (数据/划分/损失/优化器/早停/指标) 与 step3_train.py 逐项一致

结构: 经典 4 级编解码 UNet, 80x720 -> 5x45 -> 80x720, 跳连保留空间细节。
输出 Sigmoid 与 ConvLSTM 输出口径一致 (SIC 归一化 0~1)。
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x2"""

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet10(nn.Module):
    def __init__(self, in_channels, n_lead=10, base_ch=64):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c4, c5)
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = DoubleConv(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)
        self.head = nn.Conv2d(c1, n_lead, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, T_in*C, H, W)  -- 时间 x 通道 拍平, 无时间维
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.sigmoid(self.head(d1))   # (B, n_lead, H, W)
