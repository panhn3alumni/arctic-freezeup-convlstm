#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig5_leadtime_20261020.py —— 新增 Figure 5: A − persistence 逐时效配对差曲线
口径与 verify10_leadtime_check.py / make_fig4_ladder.py 完全一致:
  - 80x720 网格, 球面格点面积 (R=6371, lat 85..65, lon 30..210)
  - 海面掩膜 = 正式陆地掩膜非陆 & 观测非NaN
  - persistence = 起报日(序列最后一天)SIC平推
  - 逐日配对差 d_i(k) = IIEE_A(i,k) − IIEE_pers(i,k)
  - 95% CI = bootstrap 10000, seed=42, 同一组重采样索引用于全部时效
    (保证 T+10 与 verify10/正文 CI[+34,708,+166,005] 分毫不差)
硬标准: T+10 均值 = +100,839 ±2 km²/日; T+3..T+6 为负; T+7..T+10 单调递增。
任一不满足则不写图并报错退出。
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats as st

THRESH = 0.15
print('== fig5_leadtime: 读取配置与数据 ==')
cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
base = cfg['data'].get('base_dir', './npy')
seqs = np.load(os.path.join(base, cfg['data']['seq_file']), mmap_mode='r')

H, W = 80, 720
lat_min, lat_max, lon_min, lon_max = 65.0, 85.0, 30.0, 210.0
R = 6371.0
dlat = np.deg2rad((lat_max - lat_min) / H)
dlon = np.deg2rad((lon_max - lon_min) / W)
lat2d = np.repeat(np.linspace(lat_max, lat_min, H)[:, None], W, axis=1)
area = (R * dlat) * (R * dlon * np.cos(np.deg2rad(lat2d))).astype(np.float64)

obs = np.load('checkpoints/第6次全通道重训/test_obs.npy', mmap_mode='r')
n_dates, n_leads = obs.shape[0], obs.shape[1]
land_full = np.load(os.path.join(base, cfg['data']['mask_file']))
land_bool = land_full[0][:H, :W].astype(bool) if land_full.ndim == 3 else land_full[:H, :W].astype(bool)
sea = (~land_bool) & (~np.isnan(np.asarray(obs[0, 0])))
print(f'海面格点: {sea.sum()} / {H*W};  {n_dates} dates x {n_leads} leads')

test_start = int(0.9 * seqs.shape[0])
print('== persistence: 起报日SIC平推 ==')
pers = np.zeros((n_dates, n_leads, H, W), dtype=bool)
for j, i in enumerate(range(test_start, test_start + n_dates)):
    sic0 = np.asarray(seqs[i, -1, 0], dtype=np.float32)[:H, :W]
    pers[j] = sic0 >= THRESH
predA = np.asarray(np.load('checkpoints/第4次纯SIC/test_preds.npy', mmap_mode='r')) >= THRESH
assert predA.shape == pers.shape, (predA.shape, pers.shape)

print('== 逐日配对差 IIEE(A) − IIEE(persistence) ==')
d = np.zeros((n_dates, n_leads), dtype=np.float64)
for k in range(n_leads):
    ob = np.asarray(obs[:, k]) >= THRESH
    xa = np.logical_xor(predA[:, k], ob)[:, sea]
    xp = np.logical_xor(pers[:, k], ob)[:, sea]
    d[:, k] = (xa * area[sea]).sum(axis=1) - (xp * area[sea]).sum(axis=1)

means = d.mean(axis=0)
rng = np.random.default_rng(42)
idx = rng.integers(0, n_dates, size=(10000, n_dates))
lo = np.zeros(n_leads); hi = np.zeros(n_leads)
for k in range(n_leads):
    boot = d[idx, k].mean(axis=1)
    lo[k], hi[k] = np.percentile(boot, [2.5, 97.5])

print(f"{'lead':>5} | {'mean':>10} | {'CI lo':>10} | {'CI hi':>10} | Wilcoxon p")
for k in range(n_leads):
    try:
        pw = st.wilcoxon(d[:, k])[1]
    except ValueError:
        pw = 1.0
    print(f"  T+{k+1:<2} | {means[k]:+10,.0f} | {lo[k]:+10,.0f} | {hi[k]:+10,.0f} | {pw:.3g}")

# ---- 硬标准 ----
errs = []
if abs(means[9] - 100839) > 2: errs.append(f'T+10 mean {means[9]:+,.0f} != 100,839')
if abs(lo[9] - 34708) > 5 or abs(hi[9] - 166005) > 5:
    errs.append(f'T+10 CI [{lo[9]:+,.0f},{hi[9]:+,.0f}] != [34,708,166,005]')
if not all(means[k] < 0 for k in (2, 3, 4, 5)): errs.append('T+3..T+6 未全部为负')
mono = all(means[k + 1] > means[k] for k in (6, 7, 8))
if not mono: errs.append('T+7..T+10 非单调递增')
if errs:
    print('\n!! 硬标准未通过:'); [print('  -', e) for e in errs]
    raise SystemExit(1)
print('\n硬标准全部通过 ✓')

# ---- 绘图 (风格与图4一致: serif/Times, 300dpi, CVD 安全色) ----
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

RED = '#B2182B'
fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=300)
x = np.arange(1, n_leads + 1)
ax.axhline(0, color='#666666', lw=1.0, ls='--', zorder=1)
ax.fill_between(x, lo, hi, color=RED, alpha=0.15, lw=0, zorder=2,
                label='95 % bootstrap CI (10,000 resamples)')
ax.plot(x, means, color=RED, lw=2.0, marker='o', ms=4.5, zorder=3,
        label='A $\\minus$ persistence (paired daily mean)')

# 端点数值标注 (与正文 §4.1 的 100,839 一致)
ax.annotate(f'+{means[9]:,.0f}', xy=(10, means[9]), xytext=(-6, 6),
            textcoords='offset points', ha='right', fontsize=9, color=RED)

ax.text(4.5, -32000, 'A ahead of persistence', ha='center', fontsize=9.5,
        color='#444444')
ax.text(8.2, 78000, 'A behind persistence', ha='center', fontsize=9.5,
        color='#444444')

ax.set_xticks(x)
ax.set_xticklabels([f'T+{k}' for k in x])
ax.set_xlabel('Forecast lead time (days)')
ax.set_ylabel('Paired daily IIEE difference,\nA $\\minus$ persistence (km$^{2}$)')
ax.set_ylim(-40000, 175000)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax.tick_params(labelsize=9)
ax.legend(loc='upper left', fontsize=8.5, frameon=False)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

fig.tight_layout()
out = 'paper/TC_submission/figures/f05_leadtime.png'
fig.savefig(out, dpi=300)
print(f'\n已写出 {out}')
import PIL.Image
im = PIL.Image.open(out)
print('尺寸:', im.size, 'dpi:', im.info.get('dpi'))
