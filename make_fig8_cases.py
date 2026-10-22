#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig8_cases_20261021c.py —— 论文 Fig. 8: 两个规模相近、结果相反的
consolidation 事件的预警图 (CPU, 秒级)

第22轮终版选日标准 (用户: "务必找 10 月 1 日前起报的暖平流日"):
  - 两个案例都必须是暖平流日: 漏警(双模型)像元在观测首冻日的 t2m 距平
    高于预警像元 (ΔT > 0, step9/§3.8 口径)
  - 起报日 <= 9 月 30 日, T+10 落在 10 月 10 日前 -> 彻底封冻之前, 过渡带可见
  - 案例: 2023-09-24 (T+10 = 10-04, 域内 31.1% 转冻, D/C 捕获 81/82%, ΔT = +0.86 K)
          2025-09-28 (T+10 = 10-08, 域内 34.8% 转冻, D/C 捕获 67/69%, ΔT = +1.72 K)
  - 叙事: 两个暖平流日、规模相近, 暖平流信号翻倍 -> 捕获率降 14 个百分点
  - 2024 年候选已逐日核算 (见 scan_fig8_cases_20261021.py): 过渡像元 2001-2651
    均偏少, ΔT 0.01-0.63, 无更优组合, 未选
  - transition pixel = 起报日 SIC<0.15 且 观测在 T+1..T+10 内首次 >=0.15
  - 四类: both-warn 深蓝 / both-miss 深红 / D-miss-only 橙 / C-miss-only 青绿
    (C-only 由浅蓝 #6baed6 改 teal #19b5b5: 原色与海水底蓝 #d9e8f6 太近, 用户要求)
  - 投影: NorthPolarStereo(central_longitude=120), extent [30,210,65,85],
    与 Fig.1 / Fig.7 一致; 海岸线 50m + 灰色点线网格 (无标签, 置于最上层)
  - 第22轮: 无格内标题、无 suptitle (信息全部写入图注, 用户要求);
    legend 贴近图底 (bbox 0.5, 0.002), 图幅压缩

图层顺序 (第22轮): 底图 1/1.1/1.2 < 四分类 2 < 冰缘 6 < 海岸线 8 < 经纬网格 10

输出: results/figures/fig8_cases_20261021c.png
      paper/TC_submission/figures/f08.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

THRESH = 0.15
NH = 10
CASES = ['2023-09-24', '2025-09-28']
D_DIR = 'checkpoints/第6次全通道重训'
C_DIR = 'checkpoints/第7次无fdd重训'
OUT1 = 'results/figures/fig8_cases_20261021c.png'
OUT2 = 'paper/TC_submission/figures/f08.png'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


def load(d):
    preds = np.load(os.path.join(d, 'test_preds.npy')).astype(np.float32)
    obs = np.load(os.path.join(d, 'test_obs.npy')).astype(np.float32)
    dates = [str(x)[:10] for x in np.load(os.path.join(d, 'test_dates.npy'),
                                          allow_pickle=True)]
    if preds.ndim == 5:
        preds = preds[:, :, 0]
        obs = obs[:, :, 0] if obs.ndim == 5 else obs
    return preds, obs, dates


def freeze_h(arr):
    frozen = (arr >= THRESH)
    return np.where(frozen.any(axis=0), frozen.argmax(axis=0) + 1, 99).astype(np.int16)


def main():
    D, obs, dates = load(D_DIR)
    C = np.load(os.path.join(C_DIR, 'test_preds.npy')).astype(np.float32)
    if C.ndim == 5:
        C = C[:, :, 0]
    N, _, H, W = D.shape
    import yaml
    cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
    seqs = np.load(os.path.join(cfg['data'].get('base_dir', './npy'), cfg['data']['seq_file']),
                   mmap_mode='r')
    test_start = int(0.9 * seqs.shape[0])

    land_full = np.load(os.path.join(cfg['data'].get('base_dir', './npy'),
                                     cfg['data']['mask_file'])).astype(bool)
    land = land_full[0][:H, :W] if land_full.ndim == 3 else land_full[:H, :W]
    ocean = ~land
    lon = np.linspace(30, 210, W)
    lat = np.linspace(85, 65, H)
    lon2d, lat2d = np.meshgrid(lon, lat)

    proj = ccrs.NorthPolarStereo(central_longitude=120)
    pc = ccrs.PlateCarree()

    # 4 类: 0 both-warn, 1 both-miss, 2 D-miss-only(C warn), 3 C-miss-only(D warn)
    cmap = ListedColormap(['#1f4e79', '#8b0000', '#e8710a', '#19b5b5'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9),
                             subplot_kw={'projection': proj})
    for ax_i, (ax, cd) in enumerate(zip(axes, CASES)):
        j = dates.index(cd)
        i = test_start + j
        assert j + NH <= N - 1, 'T+10 must be inside the test set'
        t10 = dates[j + NH]  # T+10 日历日 (= 晚 10 天起报的样本起始日)
        assert t10 < cd[:4] + '-10-10', 'T+10 must fall before 10 Oct'
        sic0 = np.asarray(seqs[i, -1, 0], dtype=np.float32)[:H, :W]
        f_obs = freeze_h(obs[j]); f_D = freeze_h(D[j]); f_C = freeze_h(C[j])
        tm = (sic0 < THRESH) & ocean & (f_obs <= NH)
        wD = f_D <= NH; wC = f_C <= NH
        cls = np.full((H, W), np.nan)
        cls[tm & wD & wC] = 0
        cls[tm & ~wD & ~wC] = 1
        cls[tm & ~wD & wC] = 2
        cls[tm & wD & ~wC] = 3
        cls[land] = np.nan
        n_tm = int(tm.sum())
        pD = float((tm & wD).sum()) / n_tm * 100
        pC = float((tm & wC).sum()) / n_tm * 100
        dom_frz = float((f_obs <= NH).sum()) / (H * W) * 100
        print(f'{cd} (T+10 = {t10}): transition={n_tm} ({n_tm/ocean.sum()*100:.1f}% ocean)  '
              f'domain-frozen-in-window={dom_frz:.1f}%  '
              f'D-capture={pD:.1f}%  C-capture={pC:.1f}%')
        assert abs(pD - {CASES[0]: 80.6, CASES[1]: 66.7}[cd]) < 0.2, cd
        assert abs(pC - {CASES[0]: 81.9, CASES[1]: 68.8}[cd]) < 0.2, cd

        # 底图: 海洋浅蓝 / 陆地灰 / T+10 观测海冰白 + 黑色冰缘等值线
        ax.pcolormesh(lon2d, lat2d, np.where(ocean, 1.0, np.nan),
                      cmap=ListedColormap(['#d9e8f6']), vmin=0, vmax=2,
                      transform=pc, shading='auto', zorder=1)
        ax.pcolormesh(lon2d, lat2d, np.where(land, 1.0, np.nan),
                      cmap=ListedColormap(['#b8b8b8']), vmin=0, vmax=2,
                      transform=pc, shading='auto', zorder=1.1)
        obs10 = obs[j][9]
        ax.contourf(lon2d, lat2d, np.where(obs10 >= THRESH, 1.0, np.nan),
                    levels=[0.5, 1.5], colors=['#ffffff'], alpha=0.95,
                    transform=pc, zorder=1.2)
        im = ax.pcolormesh(lon2d, lat2d, cls, cmap=cmap, norm=norm,
                           transform=pc, shading='auto', zorder=2)
        ax.contour(lon2d, lat2d, obs10, levels=[THRESH], colors='k',
                   linewidths=1.6, transform=pc, zorder=6)
        ax.coastlines(resolution='50m', lw=0.7, zorder=8)
        gl = ax.gridlines(draw_labels=False, linewidth=0.9, color='dimgray',
                          alpha=0.85, linestyle=':', zorder=10)
        ax.set_extent([30, 210, 65, 85], crs=pc)
        # (a)/(b) 子图标签 (第23轮: 用户要求与 Fig.2 一致的字母标注)
        ax.text(0.035, 0.93, f'({chr(97 + ax_i)})', transform=ax.transAxes,
                fontsize=15, fontweight='bold', color='black',
                bbox=dict(facecolor='white', edgecolor='none', pad=2.0, alpha=0.75),
                zorder=20)
        # 第22轮: 不设格内标题 (信息写入图注)

    handles = [Patch(facecolor='#1f4e79', label='warned by both'),
               Patch(facecolor='#8b0000', label='missed by both'),
               Patch(facecolor='#e8710a', label='missed by full model (D) only'),
               Patch(facecolor='#19b5b5', label='missed by no-FDD control (C) only')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               fontsize=11.5, bbox_to_anchor=(0.5, 0.002))
    # 第22轮: 无 suptitle; legend 贴近图底
    fig.tight_layout(rect=[0, 0.065, 1, 1])
    os.makedirs(os.path.dirname(OUT1), exist_ok=True)
    fig.savefig(OUT1, dpi=300, bbox_inches='tight', pad_inches=0.12)
    fig.savefig(OUT2, dpi=300, bbox_inches='tight', pad_inches=0.12)
    print(f'saved -> {OUT1}')
    print(f'saved -> {OUT2}')


if __name__ == '__main__':
    main()
