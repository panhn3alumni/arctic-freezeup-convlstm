#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
region_defs.py —— IHO S-23 海区分划统一接口 (全部 step 脚本共用, 2026-10-10 起)

分区依据: IHO 海洋与海洋界线界定 (Limits of Oceans and Seas, S-23, 1953),
多边形取自 Flanders Marine Institute (VLIZ) MarineRegions.org "IHO Sea Areas"
图层 (https://www.marineregions.org), 与论文 Fig.1 (step14_domain_map.py) 完全一致。

掩膜文件 npy/region_mask_iho.npy (int8, 形状 80x720, 与产品网格 30-210E/65-85N 对齐):
  0=其他海域  1=巴伦支海  2=喀拉海  3=拉普捷夫海  4=东西伯利亚海
  5=楚科奇海  6=中央北极  9=陆地
生成方式: IHO 多边形栅格化到产品网格后与产品陆地掩膜求交 (陆地处记 9)。
中文名见 npy/region_mask_iho_names.json; 多边形存 region_shp/iho6.geojson
+ region_shp/barentsz.geojson。
"""
import json
import os

import numpy as np

BASE_DIR = './npy'
MASK_FILE = os.path.join(BASE_DIR, 'region_mask_iho.npy')
NAMES_FILE = os.path.join(BASE_DIR, 'region_mask_iho_names.json')

# (区域键, 英文显示名, 掩膜码) —— 顺序即各表/图/报告的区域顺序
REGION_CODES = [
    ('barents', 'Barents', 1),
    ('kara', 'Kara', 2),
    ('laptev', 'Laptev', 3),
    ('esiberian', 'East Siberian', 4),
    ('chukchi', 'Chukchi', 5),
    ('arctic_central', 'Central Arctic', 6),
    ('other', 'Remaining', 0),
]
REG_EN = {k: en for k, en, _ in REGION_CODES}


def load_iho_mask():
    """返回 (80,720) int8 IHO 掩膜 (0/1-6 海区, 9 陆地)"""
    return np.load(MASK_FILE)


def region_names_cn():
    """返回 {掩膜码: 中文名}"""
    with open(NAMES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_reg_masks(land_bool=None, include_pan=False):
    """
    返回 (ocean, reg_masks)。

    参数:
      land_bool: (80,720) bool 产品陆地掩膜; None 时用 IHO 掩膜中的 9 判陆地。
      include_pan: True 时额外给出 'pan' (全域海洋) 键 (仅 step11 需要)。

    返回:
      ocean: (80,720) bool 海洋掩膜
      reg_masks: dict 区域键 -> (80,720) bool; 键为 REGION_CODES 的 7 个区域
                 (6 个 IHO 海区 + 'other'), include_pan 时末尾多一个 'pan'。
    """
    iho = load_iho_mask()
    if land_bool is None:
        ocean = iho != 9
    else:
        land_bool = np.asarray(land_bool, dtype=bool)
        ocean = ~land_bool
        n_bad = int(((iho == 9) != land_bool).sum())
        if n_bad:
            print(f"⚠ IHO 掩膜陆地像元与产品陆地掩膜不一致 {n_bad} 个, 以产品掩膜为准")
            iho = np.where(land_bool, 9, iho)
    reg_masks = {}
    for key, _en, code in REGION_CODES:
        reg_masks[key] = ocean & (iho == code)
    if include_pan:
        reg_masks['pan'] = ocean.copy()
    return ocean, reg_masks
