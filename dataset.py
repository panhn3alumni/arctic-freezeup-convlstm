import torch
import numpy as np
from torch.utils.data import Dataset
import os


# class SeaIceDataset(Dataset):
#     def __init__(self, seq_path, tgt_path, mask_path=None, met_path=None):
#         if not os.path.exists(seq_path):
#             raise FileNotFoundError(f"数据文件不存在: {seq_path}")
#
#         # --- 修改点 1 & 2: 加上 mmap_mode='r' ---
#         # 这样不会把 36GB 数据全读进内存，而是建立一个“映射”，用到哪读哪
#         self.sequences = np.load(seq_path, mmap_mode='r')
#         self.targets = np.load(tgt_path, mmap_mode='r')
#
#         self.mask = None
#
#         self.targets_met = None
#         if met_path and os.path.exists(met_path):
#             self.targets_met = np.load(met_path, mmap_mode='r')
#
#         if mask_path and os.path.exists(mask_path):
#             self.mask = np.load(mask_path)
#             # 确保掩码形状为 (1, H, W) 以便广播
#             if self.mask.ndim == 2:
#                 self.mask = self.mask[np.newaxis, :, :]
#
#         print(f"📚 数据集加载完成 (使用内存映射): {len(self.sequences)} 样本")
#         print(f"   输入形状: {self.sequences.shape}")
#         print(f"   目标形状: {self.targets.shape}")
#
#     def __len__(self):
#         return len(self.sequences)
#
#     def __getitem__(self, idx):
#         # 这里读取的是 numpy 数组，然后转 torch
#         seq = self.sequences[idx]
#         tgt = self.targets[idx]
#
#         # print(f"输入数据范围: {seq.min()} ~ {seq.max()}")
#
#         # 转换类型并转 Tensor
#         seq = torch.from_numpy(seq.copy()).float()
#         tgt = torch.from_numpy(tgt.copy()).float()
#
#         item = {'sequence': seq, 'target': tgt}
#
#         if self.mask is not None:
#             item['mask'] = torch.from_numpy(self.mask).float()
#
#         if self.targets_met is not None:
#             item['target_met'] = torch.from_numpy(self.targets_met[idx].copy()).float()
#
#         return item


class SeaIceDataset(Dataset):
    def __init__(self, seq_path, tgt_path, mask_path=None, met_path=None, crop_size=None, keep_channels=None, met_keep_channels=None):
        if not os.path.exists(seq_path):
            raise FileNotFoundError(f"数据文件不存在: {seq_path}")

        self.sequences = np.load(seq_path, mmap_mode='r')   # (N, T, C, H, W)
        self.targets = np.load(tgt_path, mmap_mode='r')     # (N, T_out, 1, H, W)
        self.crop_size = crop_size                          # 例如 (80, 720)
        self.keep_channels = keep_channels
        self.met_keep_channels = met_keep_channels

        self.mask = None
        if mask_path and os.path.exists(mask_path):
            mask = np.load(mask_path)                       # (H, W) 或 (1, H, W)
            if mask.ndim == 2:
                mask = mask[np.newaxis, :, :]               # → (1, H, W)
            if self.crop_size is not None:
                h_crop, w_crop = self.crop_size
                mask = mask[:, :h_crop, :w_crop]             # 静态裁剪
            self.mask = mask

        self.targets_met = None
        if met_path and os.path.exists(met_path):
            self.targets_met = np.load(met_path, mmap_mode='r')  # (N, T, n_met, H, W)

        print(f"📚 数据集加载完成 (使用内存映射): {len(self.sequences)} 样本")
        print(f"   输入形状: {self.sequences.shape}")
        print(f"   目标形状: {self.targets.shape}")
        if self.crop_size is not None:
            print(f"   已裁剪到: H={self.crop_size[0]}, W={self.crop_size[1]}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 根据 crop_size 进行选择性读取，减少内存拷贝
        if self.crop_size is not None:
            h_crop, w_crop = self.crop_size
            seq = self.sequences[idx, :, :, :h_crop, :w_crop].copy()   # (T, C, h_crop, w_crop)
            tgt = self.targets[idx, :, :, :h_crop, :w_crop].copy()     # (T_out, 1, h_crop, w_crop)
        else:
            seq = self.sequences[idx].copy()
            tgt = self.targets[idx].copy()

        if self.keep_channels is not None:
            seq = seq[:, self.keep_channels, :, :]  # (T, keep_channels, H, W)

        seq = torch.from_numpy(seq).float()
        tgt = torch.from_numpy(tgt).float()

        item = {'sequence': seq, 'target': tgt}

        if self.mask is not None:
            item['mask'] = torch.from_numpy(self.mask).float()  # 已经裁剪过，形状 (1, h_crop, w_crop)

        if self.targets_met is not None:
            if self.crop_size is not None:
                h_crop, w_crop = self.crop_size
                met = self.targets_met[idx, :, :, :h_crop, :w_crop].copy()
            else:
                met = self.targets_met[idx].copy()

            if self.met_keep_channels is not None:
                met = met[:, self.met_keep_channels, :, :]  # (T_out, met_keep_channels, H, W)

            item['target_met'] = torch.from_numpy(met).float()

        return item