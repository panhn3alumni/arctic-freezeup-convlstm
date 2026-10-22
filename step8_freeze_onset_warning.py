#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step8_freeze_onset_warning.py —— 封冻预警提前量/误差类型分析 (无需重训, 秒级)

核心思想: 航道关心的不是冰缘位置准不准, 而是"封冻开始前, 模型是否及时、正确地发出预警"。
对每个测试日期, 找出"起报日开阔、未来10天内冻结"的转冻像元, 比较两组模型的:
  - 预警时机误差: 模型首次预报 SIC>=0.15 的时效 - 观测首次冻结时效 (天)
      >0 晚报(漏警, 航道危险), <0 早报(虚警, 保守), =0 完美
  - 漏警率: 10天内观测冻结但模型从未预报冻结的比例
  - 误警率: 10天内观测未冻结但模型预报冻结的比例 (虚警)

用法 (需先用 step7_fdd_edge_analysis.py --eval 导出两组 test_preds.npy):
  python step8_freeze_onset_warning.py --compare 第1次SIC修复全通道 第2次无fdd
输出: checkpoints/配对比较_{A}_vs_{B}/onset_warning_report.txt + onset_stats.json + 两张图

统计规范 (2026-10-13): 日期级配对样本报告 mean±sd 与配对差的 bootstrap 95% CI
  (10000次重采样, seed=42) + 双侧 Wilcoxon 符号秩检验; 符号检验已删除。
  P90 与 |t|>=2天占比 不再做"事件格点池化"统计 (池化把同一日期的大量格点当独立样本,
  会夸大样本量), 全部改为"逐日期先聚合、再跨日期配对"的口径。
"""
import os
import json
import argparse

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import wilcoxon

from region_defs import build_reg_masks

THRESH = 0.15
N_BOOT = 10000
SEED = 42


def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


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
        p_w = float(wilcoxon(d)[1])
    except ValueError:
        p_w = 1.0
    return {'A': describe(a), 'B': describe(b),
            'diff': {'mean': float(d.mean()),
                     'sd': float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
                     'ci_lo': lo, 'ci_hi': hi,
                     'p_wilcoxon': p_w}}


def first_freeze_horizon(arr, valid):
    """arr: (Nh,H,W) SIC; valid: (H,W) bool 参与统计的像元. 返回 (H,W) 首次冻结时效(1-based), 不冻结=99"""
    Nh = arr.shape[0]
    frozen = (arr >= THRESH) & valid[None]
    h = np.where(frozen.any(axis=0), frozen.argmax(axis=0) + 1, 99).astype(np.int16)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare', nargs=2, required=True, metavar=('A', 'B'))
    args = ap.parse_args()
    a_name, b_name = args.compare
    cfg = load_config()
    data_cfg = cfg['data']

    def find_group_dir(name):
        """优先 checkpoints_thadv (thadv 实验), 再找 checkpoints (旧实验)"""
        for root in ('checkpoints_thadv', 'checkpoints'):
            d = os.path.join(root, name)
            if os.path.exists(os.path.join(d, 'test_preds.npy')):
                return d
        raise SystemExit(f"❌ 在 checkpoints_thadv/ 和 checkpoints/ 下都找不到 {name}/test_preds.npy "
                         f"(先用 step7 --eval 导出)")

    dir_a = find_group_dir(a_name)
    dir_b = find_group_dir(b_name)
    A = np.load(os.path.join(dir_a, 'test_preds.npy')).astype(np.float32)
    B = np.load(os.path.join(dir_b, 'test_preds.npy')).astype(np.float32)
    obs = np.load(os.path.join(dir_a, 'test_obs.npy')).astype(np.float32)
    dates = list(np.load(os.path.join(dir_a, 'test_dates.npy'), allow_pickle=True))
    if list(np.load(os.path.join(dir_b, 'test_dates.npy'), allow_pickle=True)) != dates:
        raise SystemExit("❌ 两组预测日期不一致")

    seqs = np.load(os.path.join('./npy', data_cfg['seq_file']), mmap_mode='r')
    land_full = np.load(os.path.join('./npy', data_cfg['mask_file'])).astype(bool)
    land_bool = land_full[0][:80, :720] if land_full.ndim == 3 else land_full[:80, :720]
    ocean = ~land_bool
    H, W = land_bool.shape
    N, Nh = A.shape[0], A.shape[1]

    test_start = int(0.9 * seqs.shape[0])
    lats = np.linspace(data_cfg.get('lat_max', 85.0), data_cfg.get('lat_min', 65.0), H)
    lons = np.linspace(data_cfg.get('lon_min', 30.0), data_cfg.get('lon_max', 210.0), W)
    lon2d = np.repeat(lons[None, :], H, axis=0)
    lat2d = np.repeat(lats[:, None], W, axis=1)
    # IHO 海区分划 (与 fig.1 一致, 见 region_defs.py)
    _, reg_masks = build_reg_masks(land_bool)

    # ---- 逐日期计算: 转冻像元集 + 两组的预警时机 ----
    # eA/eB: (N,H,W) int16, 99=未预警; trans_mask: (N,H,W) bool
    eA = np.full((N, H, W), 99, dtype=np.int16)
    eB = np.full((N, H, W), 99, dtype=np.int16)
    trans_mask = np.zeros((N, H, W), dtype=bool)
    false_warn_A = np.zeros((N, H, W), dtype=bool)
    false_warn_B = np.zeros((N, H, W), dtype=bool)
    for j, i in enumerate(range(test_start, test_start + N)):
        sic0 = np.asarray(seqs[i, -1, 0], dtype=np.float32)[:80, :720]
        open0 = (sic0 < THRESH) & ocean
        obs_f = first_freeze_horizon(obs[j], ocean)
        tmask = open0 & (obs_f <= Nh)                       # 10天内转冻的像元
        trans_mask[j] = tmask
        hA = first_freeze_horizon(A[j], ocean)              # 全海面预报首冻时效
        hB = first_freeze_horizon(B[j], ocean)
        # 确实报冻 -> 时机误差; 10天内从未报冻 -> 99 (漏警)
        eA[j] = np.where(tmask, np.where(hA <= Nh, hA - obs_f, 99), 99)
        eB[j] = np.where(tmask, np.where(hB <= Nh, hB - obs_f, 99), 99)
        # 误警: 观测10天内未冻结, 模型却报冻结
        nofreeze = open0 & (obs_f > Nh)
        false_warn_A[j] = nofreeze & (A[j] >= THRESH).any(axis=0)
        false_warn_B[j] = nofreeze & (B[j] >= THRESH).any(axis=0)

    tag_a = '(无FDD)' if '无fdd' in a_name.lower() else '(含FDD)'
    tag_b = '(无FDD)' if '无fdd' in b_name.lower() else '(含FDD)'
    # 精确球面格点面积 (km²), 与 step7/step19 同式 (R=6371)
    lat_min = data_cfg.get('lat_min', 65.0); lat_max = data_cfg.get('lat_max', 85.0)
    lon_min = data_cfg.get('lon_min', 30.0); lon_max = data_cfg.get('lon_max', 210.0)
    R = 6371.0
    dlat = np.deg2rad((lat_max - lat_min) / H)
    dlon = np.deg2rad((lon_max - lon_min) / W)
    _lat2d = np.repeat(np.linspace(lat_max, lat_min, H)[:, None], W, axis=1)
    cell_area = (R * dlat) * (R * dlon * np.cos(np.deg2rad(_lat2d)))   # (H,W) km²

    lines = [f"A={a_name} {tag_a}  vs  B={b_name} {tag_b}",
             f"日期 {N} 个; 预警时机误差 = 模型首次报冻时效 - 观测首冻时效 (天, +晚/-早, 99=10天内从未报冻)",
             "", "== 转冻像元上的预警表现 (分海区, 事件格点合并, 排除漏警99; 此表仅作总览) =="]
    lines.append(f"  {'海区':>10s} | {'漏警率 A/B':>16s} | {'时机均值(天)':>13s} | "
                 f"{'MAE(天)':>11s} | {'误警率 A/B':>16s}")

    def region_metric(mask3d, warn):
        """漏警率(99占比)与时机统计(排除99): 有符号均值 / MAE"""
        m = trans_mask & mask3d[None]
        miss = (warn == 99) & m
        ok = m & (warn != 99)
        vals = np.abs(warn[ok]).astype(float)          # 事件格点 |时机误差| (天)
        return m, miss, ok, vals

    for rn, m2d in [('全域', ocean)] + list(reg_masks.items()):
        stats = []
        for warn in (eA, eB):
            m, miss, ok, vals = region_metric(m2d, warn)
            miss_rate = miss.sum() / max(m.sum(), 1)
            if vals.size:
                stats.append((miss_rate, warn[ok].mean(), vals.mean()))
            else:
                stats.append((miss_rate, np.nan, np.nan))
        # 误警率
        fwA = false_warn_A[:, m2d].mean(); fwB = false_warn_B[:, m2d].mean()
        lines.append(f"  {rn:>10s} | {stats[0][0]:7.2%} /{stats[1][0]:7.2%} | "
                     f"{stats[0][1]:+6.2f} /{stats[1][1]:+6.2f} | "
                     f"{stats[0][2]:5.2f} /{stats[1][2]:5.2f} | {fwA:7.2%} /{fwB:7.2%}")

    # ---- 日期级配对统计 (全域 + 关键海区): 逐日期先聚合, 再跨日期配对 ----
    lines.append("")
    lines.append("== 日期级配对统计 (每日期先聚合, 独立样本; 差=A-B, 负值=A更早/更少) ==")
    lines.append("   漏警数同时给出格点数与 km² 面积 (精确球面格点面积); "
                 "P90 与 |t|>=2天占比 在'双方都预警'的像元上逐日期计算 (不再池化)")
    onset_json = {'A': a_name, 'B': b_name, 'n_dates': int(N), 'date_level': {}}
    for rn, m2d in [('全域', ocean)] + list(reg_masks.items()):
        sel = trans_mask & m2d[None]
        missA = ((eA == 99) & sel).sum(axis=(1, 2)).astype(float)
        missB = ((eB == 99) & sel).sum(axis=(1, 2)).astype(float)
        st_m = paired_stats(missA, missB)
        # 漏警面积 (km²/日): 漏警像元 × 格点面积
        areaA = (((eA == 99) & sel).astype(np.float64) * cell_area).sum(axis=(1, 2))
        areaB = (((eB == 99) & sel).astype(np.float64) * cell_area).sum(axis=(1, 2))
        st_area = paired_stats(areaA, areaB)
        ok = sel & (eA != 99) & (eB != 99)     # 双方都预警的像元
        cnt = ok.sum(axis=(1, 2))
        have = cnt > 0
        tA = np.full(N, np.nan); tB = np.full(N, np.nan)
        tA[have] = np.where(ok, eA, 0).sum(axis=(1, 2))[have] / cnt[have]
        tB[have] = np.where(ok, eB, 0).sum(axis=(1, 2))[have] / cnt[have]
        st_t = paired_stats(tA[have], tB[have])
        # 日期级时机 MAE (|timing| 按日期平均): 有符号均值的正负抵消不暴露绝对误差
        aA = np.full(N, np.nan); aB = np.full(N, np.nan)
        aA[have] = np.where(ok, np.abs(eA), 0).sum(axis=(1, 2))[have] / cnt[have]
        aB[have] = np.where(ok, np.abs(eB), 0).sum(axis=(1, 2))[have] / cnt[have]
        st_a = paired_stats(aA[have], aB[have])
        # 日期级 P90 与 |t|>=2天占比 (双方都预警像元, 逐日期聚合后配对)
        pA = np.full(N, np.nan); pB = np.full(N, np.nan)
        gA = np.full(N, np.nan); gB = np.full(N, np.nan)
        for j in range(N):
            if not have[j]:
                continue
            va = np.abs(eA[j][ok[j]]).astype(float)
            vb = np.abs(eB[j][ok[j]]).astype(float)
            pA[j] = np.percentile(va, 90); pB[j] = np.percentile(vb, 90)
            gA[j] = float((va >= 2).mean()); gB[j] = float((vb >= 2).mean())
        st_p = paired_stats(pA[have], pB[have])
        st_g = paired_stats(gA[have], gB[have])
        onset_json['date_level'][rn] = {'miss': st_m, 'miss_area_km2': st_area,
                                        'timing': st_t, 'timing_abs': st_a,
                                        'p90': st_p, 'ge2d_share': st_g,
                                        'n_timing': int(have.sum())}
        lines.append(f"  {rn:>10s}: 漏警数/日 A={st_m['A']['mean']:6.1f}±{st_m['A']['sd']:5.1f} "
                     f"B={st_m['B']['mean']:6.1f}±{st_m['B']['sd']:5.1f} "
                     f"| 面积/日 A={st_area['A']['mean']:7.0f}±{st_area['A']['sd']:6.0f} "
                     f"B={st_area['B']['mean']:7.0f}±{st_area['B']['sd']:6.0f} km² "
                     f"| 差(A-B)={st_area['diff']['mean']:+7.0f} "
                     f"95%CI[{st_area['diff']['ci_lo']:+.0f},{st_area['diff']['ci_hi']:+.0f}] km² "
                     f"| Wilcoxon p={st_m['diff']['p_wilcoxon']:.4g}")
        lines.append(f"  {'':>10s}  时机误差(天) A={st_t['A']['mean']:+.2f}±{st_t['A']['sd']:.2f} "
                     f"B={st_t['B']['mean']:+.2f}±{st_t['B']['sd']:.2f} "
                     f"| 差(A-B)={st_t['diff']['mean']:+.2f} 95%CI[{st_t['diff']['ci_lo']:+.2f},{st_t['diff']['ci_hi']:+.2f}] "
                     f"| Wilcoxon p={st_t['diff']['p_wilcoxon']:.4g} (n={int(have.sum())}日)")
        lines.append(f"  {'':>10s}  时机MAE(天) A={st_a['A']['mean']:.2f}±{st_a['A']['sd']:.2f} "
                     f"B={st_a['B']['mean']:.2f}±{st_a['B']['sd']:.2f} "
                     f"| 差(A-B)={st_a['diff']['mean']:+.2f} 95%CI[{st_a['diff']['ci_lo']:+.2f},{st_a['diff']['ci_hi']:+.2f}] "
                     f"| Wilcoxon p={st_a['diff']['p_wilcoxon']:.4g}")
        lines.append(f"  {'':>10s}  P90(|时机|,天) A={st_p['A']['mean']:.2f}±{st_p['A']['sd']:.2f} "
                     f"B={st_p['B']['mean']:.2f}±{st_p['B']['sd']:.2f} "
                     f"| 差(A-B)={st_p['diff']['mean']:+.2f} 95%CI[{st_p['diff']['ci_lo']:+.2f},{st_p['diff']['ci_hi']:+.2f}] "
                     f"| Wilcoxon p={st_p['diff']['p_wilcoxon']:.4g}")
        lines.append(f"  {'':>10s}  |t|>=2天占比 A={st_g['A']['mean']:6.1%}±{st_g['A']['sd']:5.1%} "
                     f"B={st_g['B']['mean']:6.1%}±{st_g['B']['sd']:5.1%} "
                     f"| 差(A-B)={st_g['diff']['mean']:+6.1%} "
                     f"95%CI[{st_g['diff']['ci_lo']:+.1%},{st_g['diff']['ci_hi']:+.1%}] "
                     f"| Wilcoxon p={st_g['diff']['p_wilcoxon']:.4g}")

    # ================= 图1: 分海区柱状图 =================
    reg_names = list(reg_masks.keys())
    x = np.arange(len(reg_names))
    missA = [((eA == 99) & (trans_mask & reg_masks[rn][None])).sum() /
             max((trans_mask & reg_masks[rn][None]).sum(), 1) for rn in reg_names]
    missB = [((eB == 99) & (trans_mask & reg_masks[rn][None])).sum() /
             max((trans_mask & reg_masks[rn][None]).sum(), 1) for rn in reg_names]
    timA = [eA[(trans_mask & reg_masks[rn][None]) & (eA != 99)].mean() for rn in reg_names]
    timB = [eB[(trans_mask & reg_masks[rn][None]) & (eB != 99)].mean() for rn in reg_names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(x - 0.18, missA, 0.36, label='FDD (full)', color='tab:red')
    axes[0].bar(x + 0.18, missB, 0.36, label='No-FDD', color='tab:blue')
    axes[0].set_xticks(x); axes[0].set_xticklabels(reg_names, rotation=20)
    axes[0].set_ylabel('Missed-warning rate (freezes in 10d, model never warns)')
    axes[0].set_title('Missed freeze-up warnings'); axes[0].legend(); axes[0].grid(alpha=0.3, axis='y')

    axes[1].bar(x - 0.18, timA, 0.36, label='FDD (full)', color='tab:red')
    axes[1].bar(x + 0.18, timB, 0.36, label='No-FDD', color='tab:blue')
    axes[1].set_xticks(x); axes[1].set_xticklabels(reg_names, rotation=20)
    axes[1].set_ylabel('Mean warning timing error (days)')
    axes[1].set_title('Warning timing (>0 = too late)')
    axes[1].axhline(0, color='k', lw=0.8); axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')
    plt.tight_layout()

    out_dir = os.path.join('checkpoints', f'配对比较_{a_name}_vs_{b_name}')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'onset_warning_bars.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ================= 图2: 逐像元时机误差差异合成图 =================
    proj = ccrs.NorthPolarStereo(central_longitude=120)
    data_crs = ccrs.PlateCarree()
    # 每像元: 双方都预警的样本上 (eB - eA) 的平均; >0 = No-FDD 预警更晚 (FDD 更早)
    # (逐像元符号检验打点已按 2026-10-13 裁定删除: 显著性只在日期级配对统计中报告)
    okall = trans_mask & (eA != 99) & (eB != 99)
    diff_stack = np.where(okall, (eB - eA).astype(np.float32), np.nan)
    with np.errstate(invalid='ignore'):
        diff_map = np.nanmean(diff_stack, axis=0).astype(np.float32)
    diff_map[okall.sum(axis=0) < 5] = np.nan      # 样本过少的像元不画

    fig = plt.figure(figsize=(9.5, 8))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([lons[0], lons[-1], lats[-1], lats[0]], crs=data_crs)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    vmax = np.nanpercentile(np.abs(diff_map[ocean]), 90)
    pcm = ax.pcolormesh(lon2d, lat2d, np.where(ocean, diff_map, np.nan), transform=data_crs,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    ax.set_title('Freeze-up warning timing diff (No-FDD minus FDD, days)\n'
                 'red = FDD warns earlier', fontsize=12)
    plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.06, aspect=40, shrink=0.8)
    plt.savefig(os.path.join(out_dir, 'onset_warning_timing_map.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(out_dir, 'onset_warning_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    with open(os.path.join(out_dir, 'onset_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(onset_json, f, ensure_ascii=False, indent=1)
    print(f"\n💾 报告: {out_dir}/onset_warning_report.txt")
    print(f"💾 统计JSON: {out_dir}/onset_stats.json")
    print(f"💾 柱状图: {out_dir}/onset_warning_bars.png")
    print(f"💾 时机差异图: {out_dir}/onset_warning_timing_map.png")


if __name__ == '__main__':
    main()
