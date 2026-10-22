#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig6_missed_scatter_20261021.py —— 论文 Fig. 6: D−C 逐日漏警面积差三段折线
(2026-10-21 第21轮改折线: x=各年测试窗内日序, 消除无数据日期的空档)

口径 (与 step8/正文 Table 1 一致):
  - transition pixel = 起报日 SIC<0.15 且 观测 T+1..T+10 内首次 >=0.15
  - missed area = transition 且 模型 T+10 仍未冻结 的球面格点面积和 (km²)
  - 差值 = miss(D, 全通道) − miss(C, 无FDD); 正值 = D 漏得更多
  - 硬标准: 117 日均值必须复现正文 +13,993 km²/天
  - 三段折线: 按年着色; 年均值不再入图 (年均值数字由图注给出)

输出: results/figures/fig6_missed_scatter_20261021.png
      paper/TC_submission/figures/f06.png
"""
import os
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

THRESH = 0.15
NH = 10
D_DIR = 'checkpoints/第6次全通道重训'
C_DIR = 'checkpoints/第7次无fdd重训'
OUT1 = 'results/figures/fig6_missed_lines_20261021.png'
OUT2 = 'paper/TC_submission/figures/f06.png'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


def main():
    cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
    dc = cfg['data']
    base = dc.get('base_dir', './npy')
    seqs = np.load(os.path.join(base, dc['seq_file']), mmap_mode='r')
    land_full = np.load(os.path.join(base, dc['mask_file'])).astype(bool)
    land = land_full[0][:80, :720] if land_full.ndim == 3 else land_full[:80, :720]
    ocean = ~land
    H, W = land.shape
    R = 6371.0
    dlat = np.deg2rad((dc['lat_max'] - dc['lat_min']) / H)
    dlon = np.deg2rad((dc['lon_max'] - dc['lon_min']) / W)
    lat2d = np.repeat(np.linspace(dc['lat_max'], dc['lat_min'], H)[:, None], W, axis=1)
    area = (R * dlat) * (R * dlon * np.cos(np.deg2rad(lat2d)))

    D = np.load(os.path.join(D_DIR, 'test_preds.npy')).astype(np.float32)
    C = np.load(os.path.join(C_DIR, 'test_preds.npy')).astype(np.float32)
    obs = np.load(os.path.join(D_DIR, 'test_obs.npy')).astype(np.float32)
    dates = [str(x)[:10] for x in np.load(os.path.join(D_DIR, 'test_dates.npy'),
                                          allow_pickle=True)]
    if D.ndim == 5:
        D, C = D[:, :, 0], C[:, :, 0]
        obs = obs[:, :, 0] if obs.ndim == 5 else obs
    N = len(dates)
    test_start = int(0.9 * seqs.shape[0])

    def freeze_h(arr):
        frozen = (arr >= THRESH)
        return np.where(frozen.any(axis=0), frozen.argmax(axis=0) + 1, 99).astype(np.int16)

    diffs = []
    for j in range(N):
        i = test_start + j
        sic0 = np.asarray(seqs[i, -1, 0], dtype=np.float32)[:H, :W]
        tm = (sic0 < THRESH) & ocean & (freeze_h(obs[j]) <= NH)
        missD = tm & (freeze_h(D[j]) == 99)
        missC = tm & (freeze_h(C[j]) == 99)
        diffs.append(float((missD * area).sum() - (missC * area).sum()))
    diffs = np.array(diffs)
    mean = diffs.mean()
    print(f'117 日均值 D−C = {mean:,.0f} km²/day  (正文 +13,993)')
    assert abs(mean - 13993) < 800, f'均值偏离正文: {mean:,.0f}'

    df = pd.DataFrame({'date': pd.to_datetime(dates), 'diff': diffs})
    df['year'] = df['date'].dt.year
    df['doy'] = df.groupby('year').cumcount() + 1   # 测试窗内日序 (1..45)

    fig, ax = plt.subplots(figsize=(9.6, 4.9), dpi=300)
    ycols = {2023: '#0072B2', 2024: '#D55E00', 2025: '#009E73'}
    for y, g in df.groupby('year'):
        ax.plot(g['doy'], g['diff'], '-o', color=ycols[y], ms=3.5, lw=1.4,
                alpha=0.9, label=str(y), zorder=3)
    ax.axhline(mean, color='black', ls='--', lw=1.2, zorder=2)
    ax.annotate(f'all-date mean {mean:,.0f} km² day$^{{-1}}$',
                xy=(0.99, mean), xycoords=('axes fraction', 'data'),
                xytext=(0, 7), textcoords='offset points',
                ha='right', fontsize=10)
    ax.axhline(0, color='gray', lw=0.7, zorder=1)
    ax.set_ylabel('Excess missed first-freeze area, D $\\minus$ C\n(km² day$^{-1}$)', fontsize=12)
    ax.set_xlabel('Day of autumn test window', fontsize=12)
    ax.tick_params(labelsize=10.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(title='Start year', frameon=False, fontsize=10,
              title_fontsize=10.5, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=3)
    fig.tight_layout()
    fig.savefig(OUT1, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(OUT2, bbox_inches='tight', pad_inches=0.15)
    print(f'saved -> {OUT1}')
    print(f'saved -> {OUT2}')


if __name__ == '__main__':
    main()
