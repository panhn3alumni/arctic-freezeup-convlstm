#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_fdd_edge_analysis.py —— 把 FDD 的作用变成"看得见"的证据 (2026-10-13 口径统一)

口径 (2026-10-13 用户裁定):
  - 所有逐日期配对指标只取 T+10 单时效, 不再对 10 个时效平均
    (第1天与第10天预报准确性是鸿沟, 时效平均会稀释信号);
  - FN/FP/IIEE 一律按精确球面格点面积折算成 km², 不再使用格点数或等效冰缘位移 (km);
  - 不做快/中/慢三分位分组, 全部日期统一计算;
  - 统计: 日期级 mean±sd + 配对差 bootstrap 95%CI (10000次, seed=42)
    + 双侧 Wilcoxon 符号秩检验 (符号检验全面删除)。

用法 (两步, 与 step4_eval_paired 类似):
  模式1: 分别导出两组预测 (config.yaml 必须处于该组训练时的 use_channels 状态!)
      python step7_fdd_edge_analysis.py --eval 第1次SIC修复全通道
      python step7_fdd_edge_analysis.py --eval 第2次无fdd
      输出: checkpoints/{组名}/test_preds.npy  (float16, 日期x时效x80x720)

  模式2: 对比分析 (两组都导出后)
      python step7_fdd_edge_analysis.py --compare 第1次SIC修复全通道 第2次无fdd
      输出: checkpoints/配对比较_{A}_vs_{B}/edge_analysis_report.txt
            checkpoints/配对比较_{A}_vs_{B}/edge_stats.json (T+10 + km² 新口径)
            checkpoints/配对比较_{A}_vs_{B}/case_study_top3/*.png
  (注: 旧 composite_fastfreeze_diffmap.png 与等效冰缘位移节已删除; step18_build_tables.py
   依赖的旧 schema 随之作废, 论文正文与表格以本脚本新输出为准)
"""
import os
import json
import argparse

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

try:
    from step2_prepare_data_new import get_channel_indices
except ImportError:   # 该模块已不存在(更名 _v3); 仅 --eval 模式推导通道时才会真正调用
    def get_channel_indices(cfg):
        raise RuntimeError("step2_prepare_data_new.py 不存在, --eval 模式需要它; "
                           "--compare 模式不受影响")
from convlstm import SeaIcePredictor
from unet_model import UNet10
from region_defs import build_reg_masks

THRESH = 0.15
N_BOOT = 10000
SEED = 42
# 案例图放大区域 (lon_min, lon_max, lat_min, lat_max)
ZOOMS = [('Barents-Kara', 28, 102, 66, 82), ('Chukchi', 160, 212, 64, 76)]


def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def strip_prefix(sd):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}


def find_group_dir(name, need='test_preds.npy'):
    """优先 checkpoints_thadv (thadv 实验), 再找 checkpoints (旧实验)"""
    for root in ('checkpoints_thadv', 'checkpoints'):
        d = os.path.join(root, name)
        if os.path.exists(os.path.join(d, need)):
            return d
    raise SystemExit(f"❌ 在 checkpoints_thadv/ 和 checkpoints/ 下都找不到 {name}/{need}")


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    return {'n': int(x.size), 'mean': float(np.mean(x)),
            'sd': float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
            'median': float(np.median(x)), 'min': float(np.min(x)), 'max': float(np.max(x))}


def bootstrap_ci(d, n_boot=N_BOOT, seed=SEED):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = d.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_stats(a, b):
    """两模型描述统计 + 配对差 mean±sd + bootstrap 95%CI + 双侧 Wilcoxon 符号秩检验"""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    d = a - b
    lo, hi = bootstrap_ci(d)
    try:
        p_w = float(stats.wilcoxon(d)[1])
    except ValueError:
        p_w = 1.0
    return {'A': describe(a), 'B': describe(b),
            'diff': {'mean': float(d.mean()),
                     'sd': float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
                     'ci_lo': lo, 'ci_hi': hi,
                     'p_wilcoxon': p_w}}


# ============================================================
# 模式1: 导出一组的测试集预测
# ============================================================
def eval_group(out_name, cfg, device):
    base_dir = cfg['data'].get('base_dir', './npy')
    save_dir = find_group_dir(out_name, need='best_model.pth')
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device)
    saved_cfg = checkpoint['config']
    # UNet checkpoint (step3b 保存) 无 config['model']['input_channels'],
    # 键为顶层 unet_in_channels (= T*C 拍平) + keep_channels; 模型为 UNet10 one-shot
    is_unet = 'unet_in_channels' in checkpoint
    if is_unet:
        in_ch = int(checkpoint['unet_in_channels'])
        keep_ch = list(checkpoint['keep_channels'])
        met_keep_ch = keep_ch        # one-shot 不喂未来气象, 此处仅保持下方接口占位
    else:
        in_ch = saved_cfg['model']['input_channels']
        # 通道: 优先 checkpoint 里存的 data_channels (thadv 实验有存), 否则按当前 config 推导
        dc = checkpoint.get('data_channels', {})
        keep_ch = dc.get('keep_channels')
        met_keep_ch = dc.get('met_keep_channels')
        if keep_ch is None or met_keep_ch is None:
            keep_ch, met_keep_ch, use_names, _ = get_channel_indices(cfg)
    expect = len(keep_ch) * (10 if is_unet else 1)
    if in_ch != expect:
        raise SystemExit(f"❌ {out_name}: checkpoint是{in_ch}通道, 但推导出了{expect}个通道 —— "
                         f"旧实验请用默认 config.yaml 且处于该组训练时的 use_channels 状态; "
                         f"thadv 实验请加 --config config_thadv.yaml")

    if is_unet:
        model = UNet10(in_channels=in_ch, n_lead=10,
                       base_ch=int(saved_cfg.get('model', {}).get('base_ch', 64))).to(device)
    else:
        model = SeaIcePredictor(
            input_channels=in_ch,
            hidden_dim=saved_cfg['model']['hidden_dim'],
            kernel_size=saved_cfg['model']['kernel_size'],
            num_layers=saved_cfg['model']['num_layers'], pred_len=1).to(device)
    model.load_state_dict(strip_prefix(checkpoint['model_state_dict']))
    model.eval()
    n_c = len(keep_ch)
    print(f"✅ [{out_name}] epoch {checkpoint.get('epoch', '?')} 加载成功 "
          f"({'UNet10 one-shot, ' + str(in_ch // n_c) + 'x' if is_unet else ''}{n_c}通道, "
          f"keep_channels={keep_ch})")

    data_cfg = cfg['data']
    seqs = np.load(os.path.join(base_dir, data_cfg['seq_file']), mmap_mode='r')
    targets = np.load(os.path.join(base_dir, data_cfg['target_file']), mmap_mode='r')
    targets_met = np.load(os.path.join(base_dir, 'targets_met_seasonal.npy'), mmap_mode='r')
    land_full = np.load(os.path.join(base_dir, data_cfg['mask_file'])).astype(bool)
    dates = np.load(os.path.join(base_dir, 'sample_start_dates.npy'), allow_pickle=True)

    land_bool = land_full[0][:80, :720] if land_full.ndim == 3 else land_full[:80, :720]
    H, W = land_bool.shape
    n_roll = cfg['data']['prediction_length']
    test_start = int(0.9 * seqs.shape[0])
    test_idx = list(range(test_start, seqs.shape[0]))
    mask_tensor = torch.from_numpy(land_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    preds = np.zeros((len(test_idx), n_roll, H, W), dtype=np.float16)
    obs = np.zeros((len(test_idx), n_roll, H, W), dtype=np.float16)

    with torch.no_grad():
        for j, i in enumerate(test_idx):
            x = torch.from_numpy(np.array(seqs[i]).astype(np.float32))[None]
            x = x[:, :, keep_ch, :H, :W].to(device)
            y = torch.from_numpy(np.array(targets[i]).astype(np.float32))[None]
            y = y[:, :, :, :H, :W]
            if is_unet:
                # UNet10 one-shot: (1,T,C,H,W) -> 拍平 (1,T*C,H,W) -> (1,10,H,W); 无自回归
                p = model(x.flatten(1, 2))
                preds[j] = p[0].cpu().numpy().astype(np.float16)
            else:
                y_met = torch.from_numpy(np.array(targets_met[i]).astype(np.float32))[None]
                y_met = y_met[:, :, met_keep_ch, :H, :W].to(device)
                cur = x
                for k in range(n_roll):
                    p = model(cur, mask=mask_tensor)[:, :, 0]
                    preds[j, k] = p[0, 0].cpu().numpy().astype(np.float16)
                    if k < n_roll - 1:
                        nxt = torch.cat([p.unsqueeze(2), y_met[:, k].unsqueeze(1),
                                         cur[:, -1:, -1:, :, :]], dim=2)
                        cur = torch.cat([cur[:, 1:], nxt], dim=1)
            obs[j] = y[0, :, 0].cpu().numpy().astype(np.float16)
            print(f"  [{out_name}] {str(dates[i])[:10]} 完成 ({j+1}/{len(test_idx)})", end='\r')
    print()

    np.save(os.path.join(save_dir, 'test_preds.npy'), preds)
    np.save(os.path.join(save_dir, 'test_obs.npy'), obs)
    np.save(os.path.join(save_dir, 'test_dates.npy'), np.array([str(dates[i])[:10] for i in test_idx]))
    print(f"💾 预测已保存: {save_dir}/test_preds.npy  (float16, {preds.shape})")


# ============================================================
# 模式2: 对比分析
# ============================================================
def grid_geometry(lat_min, lat_max, lon_min, lon_max, H, W):
    """逐像元精确球面面积 (km^2), R=6371; 用于 FN/FP/IIEE 的 km² 折算"""
    lats = np.linspace(lat_max, lat_min, H)
    lons = np.linspace(lon_min, lon_max, W)
    R = 6371.0
    dlat = np.deg2rad((lat_max - lat_min) / H)
    dlon = np.deg2rad((lon_max - lon_min) / W)
    lat2d = np.repeat(lats[:, None], W, axis=1)
    area = (R * dlat) * (R * dlon * np.cos(np.deg2rad(lat2d)))   # (H,W) km^2
    return lats, lons, area


def compare(a_name, b_name, cfg):
    device = 'cpu'
    base_dir = cfg['data'].get('base_dir', './npy')
    dir_a = find_group_dir(a_name, need='test_preds.npy')
    dir_b = find_group_dir(b_name, need='test_preds.npy')
    A = np.load(os.path.join(dir_a, 'test_preds.npy')).astype(np.float32)
    B = np.load(os.path.join(dir_b, 'test_preds.npy')).astype(np.float32)
    obs = np.load(os.path.join(dir_a, 'test_obs.npy')).astype(np.float32)
    dates = list(np.load(os.path.join(dir_a, 'test_dates.npy'), allow_pickle=True))
    dates_b = list(np.load(os.path.join(dir_b, 'test_dates.npy'), allow_pickle=True))
    if dates != dates_b or A.shape != B.shape:
        raise SystemExit("❌ 两组预测的日期/形状不一致, 请确认用同一数据版本导出")

    data_cfg = cfg['data']
    seqs = np.load(os.path.join(base_dir, data_cfg['seq_file']), mmap_mode='r')
    land_full = np.load(os.path.join(base_dir, data_cfg['mask_file'])).astype(bool)
    land_bool = land_full[0][:80, :720] if land_full.ndim == 3 else land_full[:80, :720]
    ocean = ~land_bool
    H, W = land_bool.shape
    N, Nh = A.shape[0], A.shape[1]

    # 冻结强度 (与 step4_eval_stratify 相同定义)
    test_start = int(0.9 * seqs.shape[0])
    fz = []
    for j, i in enumerate(range(test_start, test_start + N)):
        sic0 = np.asarray(seqs[i, -1, 0], dtype=np.float32)[:80, :720]
        sic9 = np.asarray(obs[j, 9], dtype=np.float32)
        fz.append(float((sic9 - sic0)[ocean].mean()))
    fz = np.array(fz)

    # 经纬网格与几何
    lats, lons, cell_area = grid_geometry(
        data_cfg.get('lat_min', 65.0), data_cfg.get('lat_max', 85.0),
        data_cfg.get('lon_min', 30.0), data_cfg.get('lon_max', 210.0), H, W)
    lon2d = np.repeat(lons[None, :], H, axis=0)
    lat2d = np.repeat(lats[:, None], W, axis=1)

    # IHO 海区分划 (与 fig.1 一致, 见 region_defs.py)
    _, reg_masks = build_reg_masks(land_bool)

    # ---- T+10 单时效: 只取第 k=Nh-1 天预报 (2026-10-13 裁定, 不对 10 个时效平均) ----
    k = Nh - 1
    Ak, Bk, ok = A[:, k], B[:, k], obs[:, k]      # (N,H,W)
    pA = Ak >= THRESH; pB = Bk >= THRESH; ob = ok >= THRESH
    ob_ocean = ob & ocean
    fnA = (~pA & ob_ocean)   # A 漏报封冻 (航道风险)
    fnB = (~pB & ob_ocean)
    fpA = (pA & ~ob & ocean)
    fpB = (pB & ~ob & ocean)
    iieeA = fnA | fpA
    iieeB = fnB | fpB

    # 逐日期面积 (km²/日, 精确球面格点面积)
    fA_km = (fnA.astype(np.float64) * cell_area * ocean).sum(axis=(1, 2))
    fB_km = (fnB.astype(np.float64) * cell_area * ocean).sum(axis=(1, 2))
    qA_km = (fpA.astype(np.float64) * cell_area * ocean).sum(axis=(1, 2))
    qB_km = (fpB.astype(np.float64) * cell_area * ocean).sum(axis=(1, 2))
    iA_km = (iieeA.astype(np.float64) * cell_area * ocean).sum(axis=(1, 2))
    iB_km = (iieeB.astype(np.float64) * cell_area * ocean).sum(axis=(1, 2))
    d_iiee_km = iA_km - iB_km
    maeA = np.abs(Ak - ok)[:, ocean].mean(axis=1) * 100.0   # ×10⁻² SIC
    maeB = np.abs(Bk - ok)[:, ocean].mean(axis=1) * 100.0

    lines = [f"A={a_name} (含FDD)  vs  B={b_name} (无FDD)",
             f"日期: {N} 个; 口径: T+{k+1} 单时效, 面积单位 km²/日 (精确球面格点面积, R=6371)",
             "FN=漏报封冻(模型说没封实际封了, 航道风险), FP=虚报封冻; 差=A-B, 负值=FDD更好",
             "统计: mean±sd + 配对差 bootstrap95%CI(10000次, seed=42) + 双侧Wilcoxon符号秩检验", ""]
    n_px = float(ocean.sum())
    edge_json = {'A': a_name, 'B': b_name, 'n_dates': int(N), 'lead': f'T+{k+1}',
                 'n_ocean_px': n_px, 'pan': {}, 'regional': {},
                 'lag_subsample': {}, 'win_top': [], 'transition_top': []}

    # == 1) 泛北极 T+10 主指标 (日期级配对; 喂论文 §4.1) ==
    lines.append("== 1) 泛北极 T+10 主指标 (日期级配对) ==")
    for mname, xa, xb in [('FN(漏报,km²)', fA_km, fB_km), ('FP(虚报,km²)', qA_km, qB_km),
                          ('IIEE(km²)', iA_km, iB_km), ('MAE(×10⁻²)', maeA, maeB)]:
        st = paired_stats(xa, xb)
        edge_json['pan'][mname] = st
        lines.append(f"  {mname:>14s}: A={st['A']['mean']:9.2f}±{st['A']['sd']:8.2f}  "
                     f"B={st['B']['mean']:9.2f}±{st['B']['sd']:8.2f} "
                     f"| 差={st['diff']['mean']:+8.2f}±{st['diff']['sd']:7.2f} "
                     f"95%CI[{st['diff']['ci_lo']:+.2f},{st['diff']['ci_hi']:+.2f}] "
                     f"| Wilcoxon p={st['diff']['p_wilcoxon']:.3g}")

    # == 2) 分海区 FN/FP 率 (占海区海面格点 %, pp; 喂论文 §4.2, 替代旧 alldate_supplement.json) ==
    lines.append("")
    lines.append("== 2) 分海区 T+10 FN/FP 率 (pp; 差=A-B, 负=FDD漏报更少) ==")
    for rn, m in reg_masks.items():
        ent = {}
        for tag, tA, tB in (('FN', fnA, fnB), ('FP', fpA, fpB)):
            ra = tA[:, m].mean(axis=1) * 100.0
            rb = tB[:, m].mean(axis=1) * 100.0
            st = paired_stats(ra, rb)
            rel = st['diff']['mean'] / max(st['B']['mean'], 1e-9)
            ent[tag] = {**st['diff'], 'A_mean_pp': st['A']['mean'], 'B_mean_pp': st['B']['mean'],
                        'rel_pct': rel * 100.0}
            lines.append(f"  {rn:>14s} {tag}: A={st['A']['mean']:6.3f}±{st['A']['sd']:5.3f}  "
                         f"B={st['B']['mean']:6.3f}±{st['B']['sd']:5.3f} pp "
                         f"| 差={st['diff']['mean']:+.3f} 95%CI[{st['diff']['ci_lo']:+.3f},{st['diff']['ci_hi']:+.3f}] "
                         f"({rel:+.1%}) | Wilcoxon p={st['diff']['p_wilcoxon']:.3g}")
        edge_json['regional'][rn] = ent

    # == 3) 滞后去相关子样本 (间隔≥10天的起报日, 10天预报窗口互不重叠; 方向性复核, 喂论文统计段) ==
    LAG_DATES = ['2023-09-22', '2023-10-02', '2023-10-12',
                 '2024-09-13', '2024-09-23', '2024-10-03', '2024-10-13', '2024-10-23',
                 '2025-09-12', '2025-09-22', '2025-10-02', '2025-10-12', '2025-10-22']
    lag_idx = np.array([dates.index(d) for d in LAG_DATES if d in dates], dtype=int)
    lines.append("")
    lines.append(f"== 3) 滞后去相关子样本 (间隔≥10天, n={lag_idx.size}; 只看方向) ==")
    if lag_idx.size >= 2:
        fnA_pp = fnA[:, ocean].mean(axis=1) * 100.0
        fnB_pp = fnB[:, ocean].mean(axis=1) * 100.0
        ent = {'n': int(lag_idx.size), 'dates': [str(dates[i])[:10] for i in lag_idx]}
        for mname, xa, xb in (('FN_pp', fnA_pp, fnB_pp), ('IIEE_km2', iA_km, iB_km),
                              ('MAE_e2', maeA, maeB)):
            st = paired_stats(xa[lag_idx], xb[lag_idx])
            ent[mname] = st
            lines.append(f"  {mname:>9s}: 差={st['diff']['mean']:+.3f} "
                         f"95%CI[{st['diff']['ci_lo']:+.3f},{st['diff']['ci_hi']:+.3f}] "
                         f"| Wilcoxon p={st['diff']['p_wilcoxon']:.3g}")
        ent['spearman_mae_vs_fz'] = float(stats.spearmanr(fz[lag_idx], (maeA - maeB)[lag_idx])[0])
        ent['spearman_iiee_vs_fz'] = float(stats.spearmanr(fz[lag_idx], d_iiee_km[lag_idx])[0])
        lines.append(f"  强度关系方向: Spearman(MAE差,fz)={ent['spearman_mae_vs_fz']:+.3f}  "
                     f"Spearman(IIEE差,fz)={ent['spearman_iiee_vs_fz']:+.3f}")
        edge_json['lag_subsample'] = ent

    # == 4) 转冻日与 FDD 优势日列表 (案例挑选; km² 口径) ==
    fix_km = ((fnB & ~fnA).astype(np.float64) * cell_area * ocean).sum(axis=(1, 2))
    trans = np.zeros(N)
    for j, i in enumerate(range(test_start, test_start + N)):
        sic0 = np.asarray(seqs[i, -1, 0], dtype=np.float32)[:80, :720]
        open0 = (sic0 < THRESH) & ocean
        trans[j] = float((open0 & (ok[j] >= THRESH)).sum()) / max(ocean.sum(), 1)

    win_order = np.argsort(d_iiee_km)   # 最负 = FDD 赢最多
    lines.append("")
    lines.append("== 4) FDD 优势最大的 10 个日期 (T+10 IIEE差 km², 候选案例日) ==")
    for i in win_order[:10]:
        lines.append(f"  {dates[i]}  强度={fz[i]:+.4f}  IIEE差={d_iiee_km[i]:+9.1f} km²")

    lines.append("")
    lines.append("== 5) 封冻转变率最大的 15 个日期 (起报日开阔, 10天内大范围封冻 -> 预警关键日) ==")
    lines.append("   (转冻占比 = 起报日SIC<0.15 且 T+10>=0.15 的海面格点比例; FN修复 = FDD在T+10报对而无FDD漏报的面积)")
    for i in np.argsort(-trans)[:15]:
        lines.append(f"  {dates[i]}  转冻占比={trans[i]:6.2%}  FN修复={fix_km[i]:7.0f} km²  "
                     f"IIEE差={d_iiee_km[i]:+9.1f} km²")
    edge_json['win_top'] = [{'date': str(dates[i])[:10], 'fz': float(fz[i]),
                             'd_iiee_km2': float(d_iiee_km[i])} for i in win_order[:10]]
    edge_json['transition_top'] = [{'date': str(dates[i])[:10], 'trans': float(trans[i]),
                                    'fix_km2': float(fix_km[i]),
                                    'd_iiee_km2': float(d_iiee_km[i])}
                                   for i in np.argsort(-trans)[:15]]

    # (旧"快封冻合成差异图 + 逐像元符号检验打点"已删除: 三分位与符号检验均弃用;
    #  论文 Fig.5 的逐像元差异图由 step10_tercile_check_maps.py 出)
    out_dir = os.path.join(os.path.dirname(dir_a), f'配对比较_{a_name}_vs_{b_name}')
    os.makedirs(out_dir, exist_ok=True)
    proj = ccrs.NorthPolarStereo(central_longitude=120)
    data_crs = ccrs.PlateCarree()
    extent = [lons[0], lons[-1], lats[-1], lats[0]]

    # == 6) 案例日冰缘图 (按封冻转变率挑: 起报日开阔, 10天内大范围转冻) ==
    case_idx, seen_year = [], set()
    for i in np.argsort(-trans):
        y = str(dates[i])[:4]
        if y not in seen_year:
            case_idx.append(i); seen_year.add(y)
        if len(case_idx) == 3:
            break
    case_dir = os.path.join(out_dir, 'case_study_top3')
    os.makedirs(case_dir, exist_ok=True)

    legend_elems = [
        Line2D([0], [0], color='k', lw=1.8, label='Observed 0.15 ice edge'),
        Line2D([0], [0], color='r', lw=1.3, ls='--', label='FDD (full model) edge'),
        Line2D([0], [0], color='b', lw=1.3, ls=':', label='No-FDD edge'),
        Patch(facecolor='orange', alpha=0.65, label='Freeze-up missed by No-FDD, caught by FDD (FN fixed)'),
    ]
    for rank, i in enumerate(case_idx):
        fig = plt.figure(figsize=(21, 6.8))
        panel_titles = ['All domain', 'Barents-Kara', 'Chukchi']
        boxes = [extent] + [(lo, hi, la, lb) for (_, lo, hi, la, lb) in ZOOMS]
        for col in range(3):
            ax = fig.add_subplot(1, 3, col + 1, projection=proj)
            ax.set_extent(boxes[col], crs=data_crs)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            # 橙色阴影: 无FDD漏报、有FDD报对的格点 (T+10)
            miss_fixed = np.where(ocean, (fnB[i] & ~fnA[i]).astype(float), np.nan)
            ax.pcolormesh(lon2d, lat2d, miss_fixed, transform=data_crs,
                          cmap='Oranges', vmin=0, vmax=1, shading='auto')
            ax.contour(lon2d, lat2d, np.where(ocean, ok[i], np.nan), levels=[THRESH],
                       colors='k', linewidths=1.8, transform=data_crs)
            ax.contour(lon2d, lat2d, np.where(ocean, Ak[i], np.nan), levels=[THRESH],
                       colors='r', linewidths=1.3, linestyles='--', transform=data_crs)
            ax.contour(lon2d, lat2d, np.where(ocean, Bk[i], np.nan), levels=[THRESH],
                       colors='b', linewidths=1.3, linestyles=':', transform=data_crs)
            ax.set_title(f'{panel_titles[col]}  (T+{k+1})', fontsize=12)
        fig.suptitle(f'{dates[i]}  (open water at init: {(~(ok[i] >= THRESH) & ocean).sum() / ocean.sum():.0%} -> '
                     f'{trans[i]:.0%} freezes by T+{k+1});  FN fixed by FDD at T+{k+1}: {fix_km[i]:.0f} km²',
                     fontsize=13)
        fig.legend(handles=legend_elems, loc='lower center', ncol=4, fontsize=11, frameon=False,
                   bbox_to_anchor=(0.5, 0.02))
        plt.subplots_adjust(top=0.88, bottom=0.14, wspace=0.15)
        plt.savefig(os.path.join(case_dir, f'case{rank+1}_{dates[i]}.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(out_dir, 'edge_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    with open(os.path.join(out_dir, 'edge_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(edge_json, f, ensure_ascii=False, indent=1)
    print(f"\n💾 报告: {out_dir}/edge_analysis_report.txt")
    print(f"💾 统计JSON: {out_dir}/edge_stats.json")
    print(f"💾 案例图: {case_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', metavar='GROUP')
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B'))
    ap.add_argument('--config', default='config.yaml',
                    help='thadv 实验用 --config config_thadv.yaml; 旧实验用默认 config.yaml')
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.eval:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        eval_group(args.eval, cfg, device)
        print("\n下一组: 旧实验用 config.yaml, thadv 用 --config config_thadv.yaml, "
              "运行 python step7_fdd_edge_analysis.py --eval 另一组名")
    elif args.compare:
        compare(args.compare[0], args.compare[1], cfg)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
