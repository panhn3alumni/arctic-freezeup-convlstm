#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step16_architecture_schematic.py —— 论文 Fig. 3: ConvLSTM 预报架构示意图 v3 (CPU, 秒级)

v3 (2026-09-29): 修正为与 convlstm.SeaIcePredictor / 论文 P50 一致的真实结构:
  输入 (10 d × 15 ch × 80 × 720) → ConvLSTM 层1 (64 hidden, 3×3)
  → ConvLSTM 层2 (64 hidden, 3×3) → 卷积头 (3×3 conv + ReLU → 1×1 conv + sigmoid)
  → 10 天 SIC 预报。
  训练注脚同步改为 ocean-masked MSE (删除 SSIM/dropout 等与代码不符的表述)。
v2: 等轴测 (isometric) 3D 长方体风格 —— 仿常见 ConvLSTM 结构图:
  每层画一个带网格线的斜视角长方体, 顶面贴真实海冰 SIC 图 (取自测试集一帧),
  层间粗箭头, 输入/输出画成薄片堆叠。
输出: results/figures/fig3_architecture.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager

_names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = 'results/figures/fig3_architecture.png'

COS30, SIN30 = np.cos(np.radians(30)), np.sin(np.radians(30))
ICE_CMAP = LinearSegmentedColormap.from_list(
    'ice', ['#f7fbff', '#cfe6f5', '#8ec5e0', '#4a90b8', '#1d4e6e'])


def iso(x, y, z):
    """斜二测投影 (经典架构图风格): 层沿屏幕水平方向一字排开 (x 只影响 screen_x),
    深度 y 向右上抬, z 向上 —— 旧版 (x+y)/2 项会让整体沿 45° 爬坡"""
    return (x - y * COS30 * 0.6, z + y * SIN30 * 0.6)


def poly(ax, pts3d, fc, ec='0.15', lw=0.9, zorder=1, alpha=1.0):
    xs, ys = zip(*[iso(*p) for p in pts3d])
    ax.add_patch(Polygon(list(zip(xs, ys)), closed=True, fc=fc, ec=ec,
                         lw=lw, zorder=zorder, alpha=alpha))


def grid_lines(ax, p0, e1, e2, n1, n2, color='0.45', lw=0.35, zorder=2):
    """在 p0 + u*e1 + v*e2 的平行四边形面上画 n1×n2 网格线"""
    for i in range(1, n1):
        t = i / n1
        a = (p0[0] + e1[0] * t, p0[1] + e1[1] * t)
        b = (a[0] + e2[0], a[1] + e2[1])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=zorder)
    for j in range(1, n2):
        t = j / n2
        a = (p0[0] + e2[0] * t, p0[1] + e2[1] * t)
        b = (a[0] + e1[0], a[1] + e1[1])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=zorder)


def sic_thumbnail(nu=52, nv=22):
    """真实 SIC 一帧降采样; 本地无 npy 时退化为合成场"""
    try:
        t = np.load('npy/targets_seasonal.npy', mmap_mode='r')
        i = int(0.9 * t.shape[0])
        sic = np.asarray(t[i, 9, 0], dtype=np.float32)[:80, :720]
    except Exception:
        yy, xx = np.mgrid[0:nv, 0:nu]
        sic = 0.45 + 0.35 * np.sin(xx / 6.0) * np.cos(yy / 4.0)
    H, W = sic.shape
    ri = np.linspace(0, H, nv + 1).astype(int)
    ci = np.linspace(0, W, nu + 1).astype(int)
    thumb = np.zeros((nv, nu), dtype=np.float32)
    for j in range(nv):
        for k in range(nu):
            blk = sic[ri[j]:ri[j + 1], ci[k]:ci[k + 1]]
            thumb[j, k] = np.nanmean(blk)
    return np.clip(np.nan_to_num(thumb), 0, 1)


def slab(ax, ox, oz, w, d, h, sic=None, label=None, fc='1.0',
         top_cmap=None, face_grid=(6, 3), zorder=10, fs=8.0):
    """以 (ox, oz) 为前下角锚点画一个长方体: x∈[0,w], y∈[0,d], z∈[oz,oz+h]"""
    top = [(ox, 0, oz + h), (ox + w, 0, oz + h), (ox + w, d, oz + h), (ox, d, oz + h)]
    front = [(ox, 0, oz), (ox + w, 0, oz), (ox + w, 0, oz + h), (ox, 0, oz + h)]
    side = [(ox + w, 0, oz), (ox + w, d, oz), (ox + w, d, oz + h), (ox + w, 0, oz + h)]
    # 顶面: 贴 SIC 图
    if sic is not None and top_cmap is not None:
        nu, nv = 52, 22
        Xg = np.zeros((nv + 1, nu + 1)); Yg = np.zeros((nv + 1, nu + 1))
        for j in range(nv + 1):
            for k in range(nu + 1):
                Xg[j, k], Yg[j, k] = iso(ox + w * k / nu, d * j / nv, oz + h)
        ax.pcolormesh(Xg, Yg, sic, cmap=top_cmap, shading='auto',
                      vmin=0, vmax=1, zorder=zorder + 1)
        # 边向量与新 iso() 严格一致: x 边 -> (w, 0), y 边 -> iso 的 y 方向增量
        grid_lines(ax, iso(ox, 0, oz + h), (w, 0),
                   (-d * COS30 * 0.6, d * SIN30 * 0.6), 8, 3,
                   color='0.25', lw=0.3, zorder=zorder + 2)
        poly(ax, top, 'none', ec='0.15', lw=1.0, zorder=zorder + 3)
    else:
        poly(ax, top, fc, ec='0.15', lw=1.0, zorder=zorder)
        grid_lines(ax, iso(ox, 0, oz + h), (w, 0),
                   (-d * COS30 * 0.6, d * SIN30 * 0.6), face_grid[0], face_grid[1],
                   zorder=zorder + 1)
    # 正面 + 右侧面
    poly(ax, front, fc, ec='0.15', lw=1.0, zorder=zorder)
    poly(ax, side, _shade(fc), ec='0.15', lw=1.0, zorder=zorder)
    grid_lines(ax, iso(ox, 0, oz), (w, 0), (0, h),
               face_grid[0], max(face_grid[1], 2), zorder=zorder + 1)
    grid_lines(ax, iso(ox + w, 0, oz), (-d * COS30 * 0.6, d * SIN30 * 0.6), (0, h),
               max(face_grid[1], 2), max(face_grid[1], 2), zorder=zorder + 1)
    if label:
        cx, cy = iso(ox + w / 2, -0.05, oz + h / 2)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=fs, zorder=zorder + 4)


def _shade(fc):
    """右侧面用暗一档的颜色"""
    if fc in ('1.0', 'white'):
        return '0.82'
    return fc


def thick_arrow(ax, p_from, p_to, color='#2b2b2b', lw=3.2, ms=26):
    ax.add_patch(FancyArrowPatch(p_from, p_to, arrowstyle='-|>',
                                 mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=1, shrinkB=1, zorder=20))


def main():
    os.makedirs('results/figures', exist_ok=True)
    sic = sic_thumbnail()

    fig, ax = plt.subplots(figsize=(16.5, 6.2))
    ax.set_aspect('equal')
    ax.axis('off')

    W, D = 2.6, 1.0          # 每层宽度/深度
    GAP = 0.9                # 层间距 (x 方向)
    d = 0.12                 # 薄片厚度

    def x_at(k):
        return k * (W + GAP)

    heights = {'in': 0.9, 'c1': 1.7, 'c2': 1.7, 'h1': 0.85, 'h2': 0.85, 'out': 0.9}

    z0 = 0.0
    k = 0
    # ---- 输入堆叠: 3 片薄片代表 10 d × 15 ch ----
    xin = x_at(k); k += 1
    for s in range(3):
        off = s * 0.34
        poly(ax, [(xin, off * 0.7, z0 + off * 0.5),
                  (xin + W, off * 0.7, z0 + off * 0.5),
                  (xin + W, off * 0.7 + d, z0 + off * 0.5),
                  (xin, off * 0.7 + d, z0 + off * 0.5)],
             '#dce9f7', ec='0.2', lw=0.8, zorder=5 + s)
        grid_lines(ax, iso(xin, off * 0.7, z0 + off * 0.5), (W, 0),
                   (-d * COS30 * 0.6, d * SIN30 * 0.6), 4, 1,
                   color='0.4', lw=0.3, zorder=6 + s)
    ax.text(*iso(xin + W / 2, -0.25, z0 + 0.45),
            'Input tensor\n10 d × 15 ch\n80 × 720 grid',
            ha='center', va='center', fontsize=8.5, zorder=30)

    # ---- ConvLSTM 2 层 (64 hidden 各一层) + 卷积头 2 段 ----
    # 与 convlstm.py / 论文 P50 一致: 无 BN/dropout; 头部 3×3 conv(ReLU) + 1×1 conv(sigmoid)
    blocks = [('ConvLSTM layer 1\n64 hidden, 3×3', heights['c1'], '#cfe3d6', True),
              ('ConvLSTM layer 2\n64 hidden, 3×3', heights['c2'], '#b7d8c4', True),
              ('3×3 conv\nReLU', heights['h1'], '#f4e6c8', False),
              ('1×1 conv\nsigmoid', heights['h2'], '#f6d5d0', False)]
    xs = []
    for lab, h, fc, with_sic in blocks:
        x = x_at(k); k += 1
        if with_sic:
            slab(ax, x, z0, W, D, h, sic=sic, top_cmap=ICE_CMAP, label=lab,
                 fc=fc, fs=8.5)
        else:
            slab(ax, x, z0, W, D, h, label=lab, fc=fc, fs=8.5, face_grid=(4, 2))
        xs.append((x, h))
    x = x_at(k); k += 1
    for s in range(3):
        off = s * 0.34
        poly(ax, [(x, off * 0.7, z0 + off * 0.5),
                  (x + W, off * 0.7, z0 + off * 0.5),
                  (x + W, off * 0.7 + d, z0 + off * 0.5),
                  (x, off * 0.7 + d, z0 + off * 0.5)],
             '#dce9f7', ec='0.2', lw=0.8, zorder=5 + s)
    ax.text(*iso(x + W / 2, -0.25, z0 + 0.45),
            'Output\n10-day SIC forecast\n80 × 720 grid',
            ha='center', va='center', fontsize=8.5, zorder=30)
    xout = x

    # ---- 层间粗箭头 ----
    def arrow_between(x_prev, h_prev, x_next, h_next):
        p0 = iso(x_prev + W, 0, z0 + h_prev / 2)
        p1 = iso(x_next, 0, z0 + h_next / 2)
        thick_arrow(ax, p0, p1)

    h_in = heights['in']
    arrow_between(xin, h_in, xs[0][0], xs[0][1])
    for j in range(len(xs) - 1):
        arrow_between(xs[j][0], xs[j][1], xs[j + 1][0], xs[j + 1][1])
    arrow_between(xs[-1][0], xs[-1][1], xout, h_in)

    # ---- 通道注释 ----
    ax.text(*iso(xin + W / 2, -0.55, z0 - 0.55),
            'SIC, t2m, d2m, u10, v10, sp_grad_x/y,\nFDD2–FDD20, land_mask',
            ha='center', va='top', fontsize=7.5, color='0.3')

    # ---- 自回归回环 ----
    p_a = iso(xout + W / 2, D / 2, z0 + 2.4)
    p_b = iso(xin + W / 2, D / 2, z0 + 2.4)
    ax.add_patch(FancyArrowPatch(p_a, p_b, connectionstyle='arc3,rad=-0.15',
                                 arrowstyle='-|>', mutation_scale=14, lw=1.4,
                                 color='#b05a2a', linestyle='--', zorder=25))
    mid = iso((xin + xout + W) / 2, D / 2, z0 + 3.4)
    ax.text(mid[0], mid[1],
            'autoregressive rollout: predicted SIC replaces the SIC channel\n'
            '(meteorological channels: observed forcing, §3.6)',
            ha='center', va='bottom', fontsize=8, color='#b05a2a')

    # ---- 训练注脚 (数据坐标, 保证在所有元素下方) ----
    fx = iso(xin, 0, 0)[0]
    ax.text(fx, -2.4,
            'Training: Adam (lr 5e-5, weight decay 1e-4), ocean-masked MSE loss, '
            'batch size 8, 200 epochs max, ReduceLROnPlateau (patience 10), '
            'early stopping (patience 15)',
            fontsize=7.5, color='0.35', ha='left', va='top')

    ax.autoscale_view()
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'💾 {OUT}')


if __name__ == '__main__':
    main()
