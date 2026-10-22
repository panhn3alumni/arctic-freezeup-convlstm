#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step15_rsgp_definition_figure.py —— 论文 Fig. 2: RSGP 客观定义示意图 (CPU, 秒级)

论文位置: §3.4 "Objective definition of the rapid sea-ice growth period (RSGP)"
正文引用: "Figure 2 illustrates the definition on the pan-Arctic SIE annual cycle."

逻辑与 step2_prepare_data_new_v3.py 完全一致 (直接复用其函数, 保证图文不脱节):
  - NSIDC 海冰范围 (N_seaice_extent_daily_v4.0.csv)
  - 7 天滑动差 dSIE/change; 向后找首个 change<0 -> D0; change 峰值 -> 爆发日
  - RSGP 窗口 = D0 前 10 天 ~ 爆发日后 10 天

输出: results/figures/fig2_rsgp_definition.png
  (a) 三个测试年 (2023/2024/2025) 的 SIE 年循环 + RSGP 窗口阴影 + D0/爆发日标记
  (b) 全部年份 (1996-2025) RSGP 起止日期的年际分布 (D0 与爆发日)
"""

import os
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import pandas as pd

from step2_prepare_data_new import (
    load_csv_extent, find_key_dates, get_yearly_windows,
    D0_BEFORE_DAYS, BURST_AFTER_DAYS,
)

# SCI 风格: 衬线字体
_names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

TEST_YEARS = [2023, 2024, 2025]
START_YEAR, END_YEAR = 1996, 2025
OUT = 'results/figures/fig2_rsgp_definition.png'


def resolve_csv(explicit=None):
    """定位 NSIDC extent CSV: --csv > 常用路径 > 递归搜索 (防服务器目录名不同)."""
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise SystemExit(f'❌ --csv 指定的文件不存在: {explicit}')
    candidates = [
        'data_nsidc.0p25/N_seaice_extent_daily_v4.0.csv',   # step2 默认路径
        'N_seaice_extent_daily_v4.0.csv',
        'data_nsidc.0p25/N_seaice_extent_daily_v3.0.csv',
        'data_nsidc/N_seaice_extent_daily_v4.0.csv',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # 递归搜索当前目录 (最多下探4层, 跳过 checkpoints/results)
    hits = [p for p in glob.glob('**/N_seaice_extent*.csv', recursive=True)
            if not p.startswith(('checkpoints', 'results', 'results_false'))]
    if hits:
        return sorted(hits, key=len)[0]
    raise SystemExit(
        '❌ 找不到 NSIDC extent CSV (step2 用的那张表).\n'
        '   已尝试: 默认路径 + 递归搜索.\n'
        '   请用 --csv <路径> 显式指定, 例如:\n'
        '   python step15_rsgp_definition_figure.py --csv /path/to/N_seaice_extent_daily_v4.0.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None, help='NSIDC extent CSV 路径 (缺省自动搜索)')
    args = ap.parse_args()
    csv_path = resolve_csv(args.csv)
    print(f'使用 extent 表: {csv_path}')
    os.makedirs('results/figures', exist_ok=True)

    df = load_csv_extent(csv_path, START_YEAR, END_YEAR)
    key_dates = find_key_dates(df)
    windows = get_yearly_windows(key_dates)

    # ---- (a) 测试年 SIE 年循环 ----
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                   gridspec_kw={'width_ratios': [1.35, 1]})
    # 色盲友好配色 (Okabe-Ito): 蓝/朱红/蓝绿 —— 替代 tab10 红绿组合 (TC 要求 CVD 可读)
    colors = {2023: '#0072B2', 2024: '#D55E00', 2025: '#009E73'}
    for yr in TEST_YEARS:
        g = df[df['Year'] == yr].sort_values('date')
        if g.empty:
            print(f'⚠ {yr} 无数据, 跳过')
            continue
        axa.plot(g['date'], g['Extent'], lw=1.4, color=colors[yr], label=str(yr))
        start, end = windows[yr]
        axa.axvspan(start, end, color=colors[yr], alpha=0.13)
        # D0 / 爆发日标记
        row = key_dates[key_dates['year'] == yr].iloc[0]
        axa.axvline(row['d0_date'], color=colors[yr], ls='--', lw=1.0, alpha=0.8)
        axa.axvline(row['burst_date'], color=colors[yr], ls=':', lw=1.2, alpha=0.9)

    axa.set_ylabel(r'Sea-ice extent (10$^6$ km$^2$)')
    axa.set_xlabel('Month')
    # y 轴向上延长, 给竖排图例留出净空, 避免与曲线/阴影相互遮挡
    axa.set_ylim(3.8, 19.6)
    axa.set_title('(a) SIE annual cycle and RSGP windows (test years)', fontsize=10.5)
    leg_a = axa.legend(frameon=True, fontsize=9, loc='upper left', ncol=1)
    leg_a.get_frame().set_edgecolor('black')
    leg_a.get_frame().set_facecolor('white')
    leg_a.get_frame().set_linewidth(0.8)
    leg_a.get_frame().set_alpha(1.0)
    axa.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axa.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axa.tick_params(labelsize=9)

    # ---- (b) 年际 D0 / 爆发日分布 ----
    kd = key_dates.sort_values('year')
    yrs = kd['year'].values
    d0 = pd.to_datetime(kd['d0_date'])
    burst = pd.to_datetime(kd['burst_date'])
    # 用 "月日" 的年内序数, 便于跨年比较
    d0_doy = d0.dt.dayofyear
    burst_doy = burst.dt.dayofyear
    axb.fill_between(yrs, d0_doy, burst_doy, color='#56B4E9', alpha=0.18,
                     label='RSGP span')
    axb.plot(yrs, d0_doy, 'o-', ms=3.2, lw=1.1, color='#0072B2', label='D0 (onset)')
    axb.plot(yrs, burst_doy, 's-', ms=3.2, lw=1.1, color='#D55E00', label='burst (max expansion)')
    # 测试年高亮
    for yr in TEST_YEARS:
        if yr in yrs:
            axb.axvline(yr, color='0.55', ls='--', lw=0.8, alpha=0.7)
    axb.set_ylabel('Day of year')
    axb.set_xlabel('Year')
    axb.set_title('(b) Interannual spread of RSGP key dates', fontsize=10.5)
    axb.set_xlim(START_YEAR - 0.7, END_YEAR + 0.7)
    # y 轴向上延长, 给图例留出净空
    axb.set_ylim(min(d0_doy.min(), burst_doy.min()) - 8,
                 max(d0_doy.max(), burst_doy.max()) + 30)
    leg_b = axb.legend(frameon=True, fontsize=8.5, loc='upper left')
    leg_b.get_frame().set_edgecolor('black')
    leg_b.get_frame().set_facecolor('white')
    leg_b.get_frame().set_linewidth(0.8)
    leg_b.get_frame().set_alpha(1.0)
    i2016 = int(np.where(yrs == 2016)[0][0])
    axb.annotate('2016', xy=(2016, 325), xytext=(2012.6, 331),
                 fontsize=8.5, color='#0072B2',
                 arrowprops=dict(arrowstyle='-', lw=0.7, color='0.4'))
    axb.tick_params(labelsize=9)

    for ax in (axa, axb):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'💾 {OUT}')

    # 打印窗口表 (供论文核对 / [AUTHOR] 确认)
    print('\nRSGP 窗口 (与 step2 训练窗口一致):')
    for yr in sorted(windows):
        s, e = windows[yr]
        print(f'  {yr}: {s.date()} -> {e.date()}')


if __name__ == '__main__':
    main()
