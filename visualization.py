import os
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')   # 强制使用非交互式后端，避免线程冲突
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime


with open('config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)


# ===== 绘制训练/验证 loss 曲线 =====
def plot_loss_curve(train_losses, val_losses, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    fp_save = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(fp_save, dpi=200, bbox_inches='tight')
    plt.close()
    print("📈 已保存 loss 曲线图到", fp_save)


def save_prediction_images(epoch, pred, target, land_mask, save_dir="/home/haonanp/thinclient_drives/./results/valid"):  #
    """
    保存带陆地掩膜的预测结果图（Cartopy 北极极射投影）。

    Args:
        epoch (int): 当前 epoch
        pred (Tensor): 预测结果, shape (B, T, 1, H, W)
        target (Tensor): 真实值, shape (B, T, 1, H, W)
        land_mask (Tensor or np.ndarray): 陆地掩膜, 可以是 (H, W) 或 (B, H, W) 等
        save_dir (str): 保存目录
    """

    # ===== 1. 加载 config 以获取经纬度范围 =====
    data_cfg = cfg['data']
    lat_min = data_cfg.get('lat_min', 65.0)
    lat_max = data_cfg.get('lat_max', 85.0)
    lon_min = data_cfg.get('lon_min', 30.0)
    lon_max = data_cfg.get('lon_max', 210.0)

    save_dir = os.path.join(save_dir, cfg['infer'].get('output_path', datetime.now().strftime('%Y%m%d_%H%H%S')))
    os.makedirs(save_dir, exist_ok=True)

    # ===== 2. 数据预处理：只取第一个样本，并 squeeze 多余的维度 =====
    # pred/target: (B, T, 1, H, W) -> (T, H, W)
    pred = pred[0].squeeze(1)     # 去掉 batch 和通道维
    target = target[0].squeeze(1)

    # land_mask：统一成 (H, W) 布尔数组
    if isinstance(land_mask, torch.Tensor):
        land_mask_np = land_mask.cpu().numpy()
    else:
        land_mask_np = np.asarray(land_mask)
    while land_mask_np.ndim > 2:
        land_mask_np = land_mask_np[0]   # 多次取第一维
    land_mask_bool = land_mask_np > 0.5

    H, W = land_mask_bool.shape

    # 生成经纬度网格
    lats = np.linspace(lat_max, lat_min, H)  # 从高到低
    lons = np.linspace(lon_min, lon_max, W)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # ===== 3. 确定要画的时间步 =====
    time_steps = [0, 1, 3, 6, 10]
    max_t = pred.shape[0]
    valid_steps = [t for t in time_steps if t < max_t]
    n_rows = len(valid_steps)
    if n_rows == 0:
        print("Warning: No valid time steps to plot.")
        return

    # ===== 4. 绘图设置（北极极射投影） =====
    proj = ccrs.NorthPolarStereo(central_longitude=120)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(20, 4 * n_rows))
    gs_main = gridspec.GridSpec(n_rows, 4, wspace=0.05, hspace=0.3)

    for i, t_idx in enumerate(valid_steps):
        p = pred[t_idx].cpu().numpy()  # (H, W)
        t = target[t_idx].cpu().numpy()

        abs_error = np.abs(p - t)
        diff = p - t

        # 陆地设为 NaN，这样 pcolormesh 会将其留白（或显示为背景色）
        t_masked = np.where(land_mask_bool, np.nan, t)
        p_masked = np.where(land_mask_bool, np.nan, p)
        abs_error_masked = np.where(land_mask_bool, np.nan, abs_error)
        diff_masked = np.where(land_mask_bool, np.nan, diff)

        data_list = [
            (t_masked, 'viridis', 0.0, 1.0, f"Observation (T+{t_idx + 1})"),
            (p_masked, 'viridis', 0.0, 1.0, f"Prediction (T+{t_idx + 1})"),
            (abs_error_masked, 'hot', 0.0, 0.5, f"Absolute Error (T+{t_idx + 1})"),
            (diff_masked, 'coolwarm', -0.5, 0.5, f"Difference (Pred - Obs) (T+{t_idx + 1})")
        ]

        for j, (data, cmap_name, vmin, vmax, title) in enumerate(data_list):
            ax = fig.add_subplot(gs_main[i, j], projection=proj)
            ax.set_extent([lon_min, lon_max, 70, 85], crs=data_crs)  # 稍微放大一点以看清边缘
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.gridlines(draw_labels=True, dms=True,
                         x_inline=False, y_inline=False, crs=data_crs,
                         linewidth=0.3, color='gray', alpha=0.5)

            pcm = ax.pcolormesh(lon2d, lat2d, data,
                                transform=data_crs,
                                cmap=cmap_name, vmin=vmin, vmax=vmax,
                                shading='auto')
            ax.set_title(title, fontsize=9, pad=2)

            cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal',
                                pad=0.2, aspect=40, shrink=0.8)
            cbar.ax.tick_params(labelsize=7)

    plt.suptitle(f"Epoch {epoch + 1} Validation Results", fontsize=14)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05,
                        wspace=0.05, hspace=0.15)

    save_path = os.path.join(save_dir, f"epoch_{epoch + 1:03d}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"📈 已保存预测图到 {save_path}")


def plot_metric_histograms(metrics_dict, epoch, save_dir="/home/haonanp/thinclient_drives/./results/valid"):  #
    """
    绘制验证集逐样本的6大指标频数分布直方图
    metrics_dict: {'TS': [...], 'Prec': [...], 'Rec': [...], 'AEE': [...], 'ME': [...], 'IIEE': [...]}
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    keys = ['TS', 'Prec', 'Rec', 'AEE', 'ME', 'IIEE']
    xlabels = {
        'TS': 'Threat Score',
        'Prec': 'Precision',
        'Rec': 'Recall',
        'AEE': 'AEE',
        'ME': 'ME',
        'IIEE': 'IIEE'
    }

    for ax, key in zip(axes.flatten(), keys):
        data = np.array(metrics_dict[key])
        # 过滤掉全0样本(无冰且预测无冰)导致的NaN或无意义指标
        valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center')
            ax.set_xlabel(xlabels[key])
            ax.set_ylabel('Frequency')
            continue

        ax.hist(valid_data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        mean_val = np.mean(valid_data)
        median_val = np.median(valid_data)
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='-', label=f'Median: {median_val:.3f}')
        ax.set_xlabel(xlabels[key])
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # plt.suptitle(f'Epoch {epoch + 1} Binary Metrics Distribution', fontsize=16)
    plt.tight_layout()

    save_dir = os.path.join(save_dir, cfg['infer'].get('output_path', datetime.now().strftime('%Y%m%d_%H%H%S')))
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'metric_hist_epoch{epoch + 1}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 已保存指标分布直方图: {save_path}")