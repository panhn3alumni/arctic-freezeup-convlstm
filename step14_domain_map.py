#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step14_domain_map.py —— 论文 Fig.1: 研究区图 (CPU, 秒级)

内容: 0.25° 分析网格 (30-210°E, 65-85°N), 陆地掩膜, IHO S-23 海区分划
(巴伦支/喀拉/拉普捷夫/东西伯利亚/楚科奇 + 中央北极, 多边形来自 MarineRegions.org
IHO sea areas, 与 npy/region_mask_iho.npy 一致), 东北航道 5 个咽喉点,
NEP 走向示意线. 图内全英文, Times 衬线字体.

用法: python step14_domain_map.py
输出: results/figures/fig1_domain_map.png
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

_names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import patheffects

# 白描边: 保证文字/航线在填色与灰陆地格上都清晰可读
PE_TXT = patheffects.withStroke(linewidth=2.4, foreground='white')
PE_TXT_BIG = patheffects.withStroke(linewidth=3.2, foreground='white')
PE_LINE = [patheffects.Stroke(linewidth=4.4, foreground='white'), patheffects.Normal()]

# IHO S-23 海区多边形 (MarineRegions.org): (shp要素名, 图内标签, 填色, 标签经度, 标签纬度)
REG_FILL = [
    ('Barentsz Sea', 'Barents Sea', '#c6dbef', 40.0, 72.1),
    ('Kara Sea', 'Kara Sea', '#c7e9c0', 74.0, 74.6),
    ('Laptev Sea', 'Laptev Sea', '#fdd0a2', 127.0, 75.6),
    ('East Siberian Sea', 'East Siberian Sea', '#fcbba1', 155.0, 73.0),
    ('Chukchi Sea', 'Chukchi Sea', '#dadaeb', 194.0, 70.1),
    ('Arctic Ocean', 'Central Arctic', '#e5f5f9', 120.0, 83.6),
]
gdf = pd.concat([gpd.read_file('region_shp/iho6.geojson'),
                 gpd.read_file('region_shp/barentsz.geojson')], ignore_index=True)
# 咽喉点: (名称, 三角经度, 三角纬度, 文字经度, 文字纬度) —— 三角放航线拐点, 文字单独定位避让岛屿
CHOKE = [('Novaya Zemlya', 55.0, 72.5, 57.8, 72.9),
         ('Vilkitsky Strait', 106.5, 78.0, 106.5, 78.85),
         ('Dmitry Laptev Strait', 140.0, 72.9, 137.0, 73.6),
         ('Longa Strait', 179.5, 70.2, 179.5, 70.95),
         ('Bering Strait', 191.0, 66.45, 192.6, 66.1)]

land_full = np.load('./npy/land_mask_static.npy').astype(bool)
land = land_full[0] if land_full.ndim == 3 else land_full
H, W = land.shape[:2]
lat = np.linspace(85, 65, H)
lon = np.linspace(30, 210, W)
lon2d, lat2d = np.meshgrid(lon, lat)

# v2: 与 step4/step5 系列图统一投影 (NorthPolarStereo, central_longitude=120),
#     等经纬网会把高纬地区横向拉长, 极投影下形状更接近真实
pc = ccrs.PlateCarree()
proj = ccrs.NorthPolarStereo(central_longitude=120)
fig, ax = plt.subplots(figsize=(7.6, 7.4), subplot_kw={'projection': proj})
ax.set_extent([30, 210, 65, 85], crs=pc)
ax.add_feature(cfeature.LAND, facecolor='0.92', edgecolor='none', zorder=0)
# 用产品陆地掩膜叠加 (与管线一致), 半透明
land_plot = np.where(land, 1.0, np.nan)
ax.pcolormesh(lon2d, lat2d, land_plot, cmap='Greys', vmin=0, vmax=2, alpha=0.55,
              shading='auto', transform=pc, zorder=1)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle=':')
gl.top_labels = gl.right_labels = False

for name_shp, label, fc, lx, ly in REG_FILL:
    for _, row in gdf[gdf['name'] == name_shp].iterrows():
        # 边线不画: IHO S-23 中 East Siberian Sea 北界是沿 150°E 经线到极点的直线段,
        # 画出来像一根突兀的黑经线; 海区之间靠填色区分即可
        ax.add_geometries([row.geometry], crs=pc, facecolor=fc, edgecolor='none',
                          alpha=0.6, zorder=2)
    ax.text(lx, ly, label, fontsize=9, ha='center', va='center', color='0.15',
            transform=pc, zorder=7, path_effects=[PE_TXT])

# NEP 走向示意 (NSR 主流商路: 全程紧贴俄罗斯北极沿岸、串连各海峡; 西端自挪威北角外
#   从巴伦支海进入, 绕新地岛北端 Cape Zhelaniya 进喀拉海, 贴泰梅尔半岛西北-东北岸经
#   Vilkitsky Str., 沿泰梅尔东岸-勒拿三角洲外南下, 经 Dmitry Laptev Str. 出拉普捷夫海,
#   贴东西伯利亚大陆沿岸 72.5-70.4°N (Stolbovoy/Belkovsky/Aion/Medvezhy 诸岛均绕行),
#   弗兰格尔岛经 Longa Str., 绕楚科奇半岛北缘到 Bering Str.
#   依据: ABS NSR Advisory (standard routes follow the position of ice massifs and
#   open water along the coast) 与 CHNL 过境航迹 waypoint 序列)
cx = [30.3, 35.5, 40, 48, 56, 64, 70, 78, 85, 95, 104, 106.5, 112, 118, 124, 130, 133,
      137, 139.5, 140.5, 142, 143.5, 145, 146.5, 149, 150.5, 151.5, 154,
      157.5, 160, 163, 167, 169.5, 171, 173, 176, 179.5, 184.5, 188, 190.3]
cy = [70.9, 71.0, 71.2, 73.2, 76.2, 77.4, 77.2, 77.0, 77.0, 77.6, 77.9, 78.0, 77.0, 76.0,
      74.8, 73.8, 73.4, 72.6, 72.6, 72.9, 72.85, 72.85, 72.7, 72.6, 72.55, 72.0,
      71.6, 71.2, 71.3, 71.25, 70.4, 69.8, 70.1, 70.3, 70.35, 70.1, 70.2, 69.9, 69.2, 66.3]
ax.plot(cx, cy, '-', color='royalblue', lw=2.2, alpha=0.95, transform=pc, zorder=3,
        path_effects=PE_LINE, solid_capstyle='round')
ax.annotate('', xy=(190.4, 66.6), xytext=(189.6, 67.8),
            arrowprops=dict(arrowstyle='-|>', color='royalblue', lw=2), transform=pc, zorder=3)
ax.text(95, 79.8, 'Northeast Passage', fontsize=11, color='royalblue', weight='bold',
        ha='center', transform=pc, zorder=7, path_effects=[PE_TXT_BIG])

# 自检: 航线密集采样, 逐点检查是否落在陆地掩膜上
def _to_ij(lon_, lat_):
    return (int(round((85 - lat_) / 20 * (H - 1))), int(round((lon_ - 30) / 180 * (W - 1))))

_hits = []
for (x0, y0), (x1, y1) in list(zip(zip(cx, cy), zip(cx[1:], cy[1:]))):
    for t in np.linspace(0.02, 1, 250):
        x_, y_ = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        i_, j_ = _to_ij(x_, y_)
        if 0 <= i_ < H and 0 <= j_ < W and land[i_, j_]:
            _hits.append((round(float(x_), 1), round(float(y_), 1)))
print(f'航线陆地命中: {len(_hits)} 点' + (f' 示例 {_hits[:8]}' if _hits else ' (全程海上 ✓)'))

for name, x, y, tx, ty in CHOKE:
    ax.plot(x, y, 'v', color='darkorange', ms=7, mec='k', mew=0.5, transform=pc, zorder=6)
    ax.text(tx, ty, name, fontsize=8, ha='center', color='darkorange',
            transform=pc, zorder=7, path_effects=[PE_TXT])

ax.set_title('')
os.makedirs('results/figures', exist_ok=True)
out = 'results/figures/fig1_domain_map.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'💾 {out}')
