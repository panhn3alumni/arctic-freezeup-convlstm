#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step13_ig_maps.py —— 论文 §4.3 的 IG 空间归因图 (CPU, 秒级; 数据来自 step12 已存盘的 npy)

读 results/ig_audit/{exp}/{target}/{date}.npy  (形状 (T,C,H,W), 时间步求和后画图)
叠加观测 T+10 冰缘等值线 (0.15), 通道可选; 另出一张 分区×通道 带内均值柱状图
(跨日期平均), 直接对应论文 §4.3 的 "短窗抑制/长窗促进/FDD8 主导".

用法:
  python step13_ig_maps.py                          # 默认日期与通道
  python step13_ig_maps.py --dates 2023-10-01 2024-10-16 --channels t2m fdd2 fdd8 fdd20
输出: results/ig_audit/{exp}/{target}/fig_ig_maps.png + fig_ig_region_bars.png
"""
import os
import argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

_names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

import cartopy.crs as ccrs

from region_defs import build_reg_masks, REG_EN

BAND_LO, BAND_HI = 0.02, 0.40
THRESH = 0.15

# 色盲友好配色 (Okabe-Ito) —— 替代 matplotlib 默认色板的红绿相邻组合 (TC 要求 CVD 可读)
CH_COLORS = {'SIC': '#0072B2', 't2m': '#D55E00', 'fdd2': '#009E73',
             'fdd8': '#CC79A7', 'fdd20': '#56B4E9'}
_MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
           'August', 'September', 'October', 'November', 'December']


def fmt_date(d):
    """'2023-10-01' -> '1 October 2023' (TC 日期格式: 25 July 2007)"""
    y, m, dd = str(d)[:10].split('-')
    return f'{int(dd)} {_MONTHS[int(m) - 1]} {y}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='第1次SIC修复全通道')
    ap.add_argument('--target', default='roll10_edge')
    ap.add_argument('--dates', nargs='*', default=['2023-10-01', '2024-10-01', '2025-10-01'])
    ap.add_argument('--channels', nargs='*', default=['t2m', 'fdd2', 'fdd8', 'fdd20'])
    args = ap.parse_args()

    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    dc = cfg['data']
    base = os.path.join('results', 'ig_audit', args.exp, args.target)
    ch_names = ['SIC', 't2m', 'd2m', 'u10', 'v10', 'sp_grad_x', 'sp_grad_y',
                'fdd2', 'fdd4', 'fdd6', 'fdd8', 'fdd10', 'fdd15', 'fdd20', 'land_mask']
    i_ch = {c: ch_names.index(c) for c in args.channels}

    land_full = np.load('./npy/land_mask_static.npy').astype(bool)
    land = land_full[0][:80, :720] if land_full.ndim == 3 else land_full[:80, :720]
    ocean = ~land
    H, W = land.shape
    lon = np.linspace(dc.get('lon_min', 30), dc.get('lon_max', 210), W)
    lat = np.linspace(dc.get('lat_max', 85), dc.get('lat_min', 65), H)
    lon2d, lat2d = np.meshgrid(lon, lat)
    # IHO 海区分划 (与 fig.1 一致, 见 region_defs.py)
    _, reg_masks = build_reg_masks(land)

    dates_np = [str(d)[:10] for d in np.load('./npy/sample_start_dates.npy', allow_pickle=True)]
    targets = np.load('./npy/targets_seasonal.npy', mmap_mode='r')

    # ---------- 图A: 各日期 × 通道 空间归因 (v3: 全图共用一个对称色标(右侧), 压缩空白) ----------
    proj = ccrs.NorthPolarStereo(central_longitude=120)
    pc = ccrs.PlateCarree()

    def _clean(a, tag):
        n_inf = int(np.isinf(a).sum()); n_nan = int(np.isnan(a).sum())
        if n_inf or n_nan:
            print(f"⚠ {tag}: {n_inf} inf / {n_nan} nan —— 旧版 float16 溢出所致, 请用 float32 版 step12 重跑!")
            a = np.nan_to_num(a, nan=0.0, posinf=60000.0, neginf=-60000.0)
        return a

    attrs = {}
    for d in args.dates:
        p = os.path.join(base, f'{d}.npy')
        if not os.path.exists(p):
            raise SystemExit(f'❌ 缺 {p} (先用 step12 生成)')
        raw = np.load(p)
        a = _clean(raw.astype(np.float32).sum(axis=0), d)   # (C,H,W) 时间求和
        attrs[d] = a
        per = []
        for ci, cn in enumerate(ch_names):
            v = a[ci][ocean]; v = v[np.isfinite(v)]
            per.append(f'{cn}={np.abs(v).max():.1e}' if v.size else f'{cn}=NA')
        print(f"[diag] {d}: shape={raw.shape} dtype={raw.dtype} | 逐通道 |max|: " + '  '.join(per))

    # 全图共用一个对称色标 (SymLogNorm): t2m 量级 (~1.4×10⁻⁴) 约为 FDD 通道 (~3-5×10⁻⁵)
    # 的 3-4 倍, 线性共享会把 FDD 面板洗白 —— 对称 log 色标 (±linthresh 内线性, 以外对数)
    # 让 FDD 特征保持醒目, t2m 大值压缩, 一根 colorbar 兼容全部通道。
    # 绘图数据统一缩放为 ×10⁻⁵ SIC 单位, 与 fig_ig_region_bars 一致。
    from matplotlib.colors import SymLogNorm, ListedColormap
    vmax = 0.0
    for d in args.dates:
        for c in args.channels:
            v = attrs[d][i_ch[c]][ocean]; v = v[np.isfinite(v)]
            if v.size:
                vmax = max(vmax, float(np.abs(v).max()))
    if vmax == 0.0:
        raise SystemExit('⚠ 所选通道归因全为 0 —— step12 端问题 (下溢/通道序错位), 先看 [diag] 行')
    vmax *= 1e5   # ×10⁻⁵ SIC units
    LINTHRESH = 1.0   # ±1×10⁻⁵ 以内线性
    norm = SymLogNorm(linthresh=LINTHRESH, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)

    nrow, ncol = len(args.dates), len(args.channels)
    # 极投影扇区面板宽:高 ≈ 1.86:1 (实测, extent [30,210,65,85])。constrained_layout
    # 按 aspect 收拢面板, 行配额若大于面板实际高就会留大块行间空白 —— 第22轮实测
    # 233px/行的白带, 故图高按 面板宽/1.86 精算, 行间只留 ~0.1 in。
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol + 0.75, 3.05 / 1.86 * nrow - 0.22),
                             subplot_kw={'projection': proj},
                             squeeze=False, constrained_layout=True)
    # 2026-10-13: 压缩三行之间的行距 (用户要求), 仅收紧纵向; 第22轮再缩 (hspace 0.01->0.008)
    try:
        fig.get_layout_engine().set(h_pad=0.01, hspace=0.008)
    except (AttributeError, TypeError):
        fig.set_constrained_layout_pads(h_pad=0.02, hspace=0.01)
    im = None
    for r, d in enumerate(args.dates):
        i = dates_np.index(d) if d in dates_np else None
        obs10 = np.asarray(targets[i, 9, 0], dtype=np.float32)[:H, :W] if i is not None else None
        for k, (c, ax) in enumerate(zip(args.channels, axes[r])):
            ax.set_extent([30, 210, 65, 85], crs=pc)
            ax.pcolormesh(lon2d, lat2d, np.where(land, 1.0, np.nan),
                          cmap=ListedColormap(['#b8b8b8']), vmin=0, vmax=2,
                          shading='auto', transform=pc, zorder=0)
            im = ax.pcolormesh(lon2d, lat2d, np.where(ocean, attrs[d][i_ch[c]], np.nan) * 1e5,
                               cmap='RdBu_r', norm=norm,
                               shading='auto', transform=pc, zorder=2)
            ax.coastlines(resolution='50m', lw=0.6, zorder=8)
            gl = ax.gridlines(draw_labels=False, linewidth=0.9, color='dimgray',
                              alpha=0.85, linestyle=':', zorder=10)
            if obs10 is not None:
                ax.contour(lon2d, lat2d, obs10, levels=[THRESH], colors='k',
                           linewidths=1.2, transform=pc, zorder=6)
                ax.contour(lon2d, lat2d, obs10, levels=[BAND_HI], colors='dimgray',
                           linewidths=0.8, linestyles='dashed', transform=pc, zorder=6)
            if r == 0:   # 列=通道, 标在最上一行上方 (矩阵图必须自带行列标识)
                ax.set_title(c.upper() if c.startswith('fdd') else c, fontsize=18, pad=4)
        # 行=起报日, 标在每行左侧 (ax.text 不裁剪, savefig tight bbox 会包含)
        axes[r][0].text(-0.07, 0.5, fmt_date(d), transform=axes[r][0].transAxes,
                        rotation=90, ha='right', va='center', fontsize=17)
    cb = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.9, pad=0.015,
                      ticks=[-10, -3, -1, 0, 1, 3, 10])
    cb.set_label(r'IG attribution ($\times 10^{-5}$ SIC units, symlog scale)', fontsize=17)
    cb.ax.tick_params(labelsize=17)
    out_a = os.path.join(base, 'fig_ig_maps.png')
    fig.savefig(out_a, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'💾 {out_a}  (共用色标 ±{vmax:.2f}×10⁻⁵, 列=通道标于顶部, 行=起报日标于左侧)')

    # ---------- 图B: 分区×通道 带内均值 (跨日期平均) ----------
    # 2026-10-19: 柱状图加入 SIC 系列 (题注声称 initialization-date concentration 主导,
    # 图中必须有该系列); 图例显示名统一为正文记号 (FDD 大写), 与 Table 1 / Fig.5 列标题一致
    bar_channels = ['SIC'] + [c for c in args.channels if c != 'SIC']
    i_ch = {c: ch_names.index(c) for c in set(args.channels) | set(bar_channels)}

    def disp(c):
        return 'FDD' + c[3:] if c.startswith('fdd') else c

    rows = {c: {n: [] for n in reg_masks} for c in bar_channels}
    for d in args.dates:
        i = dates_np.index(d)
        obs10 = np.asarray(targets[i, 9, 0], dtype=np.float32)[:H, :W]
        band = ocean & (obs10 >= BAND_LO) & (obs10 <= BAND_HI)
        for c in bar_channels:
            for n, m in reg_masks.items():
                sel = band & m
                if sel.sum():
                    rows[c][n].append(float(attrs[d][i_ch[c]][sel].mean()))
    n_reg = len(reg_masks)
    x = np.arange(n_reg)
    nb = len(bar_channels)
    fig2, ax = plt.subplots(figsize=(9, 4.2))
    for k, c in enumerate(bar_channels):
        vals = [np.mean(rows[c][n]) if rows[c][n] else np.nan for n in reg_masks]
        ax.bar(x + (k - nb / 2 + 0.5) * 0.8 / nb, np.array(vals) * 1e5, 0.8 / nb,
               label=disp(c), color=CH_COLORS.get(c))
    ax.set_xticks(x)
    ax.set_xticklabels([REG_EN[n] for n in reg_masks], fontsize=11)
    ax.tick_params(axis='y', labelsize=10.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel(r"band-mean IG attribution ($\times 10^{-5}$ SIC units)", fontsize=12)
    # 图内不写标题 (2026-10-13 规范): 标题信息移入 docx 图题
    ax.legend(fontsize=10.5)
    ax.grid(axis='y', alpha=0.3)
    out_b = os.path.join(base, 'fig_ig_region_bars.png')
    plt.tight_layout()
    fig2.savefig(out_b, dpi=300, bbox_inches='tight')
    print(f'💾 {out_b}')


if __name__ == '__main__':
    main()
