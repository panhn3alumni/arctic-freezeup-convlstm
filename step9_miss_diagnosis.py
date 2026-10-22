#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_miss_diagnosis.py —— 封冻漏警的机制诊断: 暖湿平流假说检验 (无需重训, 秒级)

问题: step8 显示快速封冻前沿大量漏警 (Barents ~53%), 且 FDD 无增益。
假说: 首次封冻由天气尺度暖湿平流事件控制 (Crawford et al. 2025 的气旋机制),
      而非累积冷量 —— 因此任何热力记忆特征 (FDD) 都救不了 onset 预警。

检验: 对每个转冻像元 (起报日开阔, 10天内冻结), 取其"观测首冻日"的 2m 温度:
      与该像元训练期 (测试集之前全部样本) 的气候平均相减得到逐像元距平,
      比较 "两模型都漏警" vs "两模型都预警" 像元的距平。
      若漏警像元冻结日显著偏暖 (正距平) -> 暖平流事件压过累积冷量 -> 假说成立。
      注意: ΔT(漏警-预警) 在数学上与原始温度差相同 (气候平均在差值中抵消),
      改为距平只影响 T_miss / T_warn 两个绝对量的解释, 不改变 ΔT 与显著性。

用法: python step9_miss_diagnosis.py --compare 第1次SIC修复全通道 第2次无fdd
输出: checkpoints/配对比较_{A}_vs_{B}/miss_diagnosis_report.txt + miss_stats.json + 1 图

统计规范 (2026-10-13): 逐日期配对温度差报告 mean±sd + bootstrap 95% CI (10000次,
  seed=42) + 双侧 Wilcoxon 符号秩检验; 漏警/预警两组各自给 mean±sd; 符号检验已删除。
"""
import os
import json
import argparse

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from matplotlib import font_manager

# SCI 图: 衬线字体 (Times New Roman 优先, 与其余论文图一致)
_fn = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

from region_defs import build_reg_masks, REG_EN

THRESH = 0.15
N_BOOT = 10000
SEED = 42
# ERA5 气象变量顺序 (step2 生成 targets_met 的默认顺序, 不确定时看报告开头打印的名字)
DEFAULT_MET_NAMES = ['t2m', 'd2m', 'u10', 'v10', 'sp_grad_x', 'sp_grad_y']


def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def bootstrap_ci(d, n_boot=N_BOOT, seed=SEED):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = d.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def first_freeze_horizon(arr, valid):
    Nh = arr.shape[0]
    frozen = (arr >= THRESH) & valid[None]
    return np.where(frozen.any(axis=0), frozen.argmax(axis=0) + 1, 99).astype(np.int16)


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
    N, Nh = A.shape[0], A.shape[1]

    seqs = np.load(os.path.join('./npy', data_cfg['seq_file']), mmap_mode='r')
    met = np.load('./npy/targets_met_seasonal.npy', mmap_mode='r')
    land_full = np.load(os.path.join('./npy', data_cfg['mask_file'])).astype(bool)
    land_bool = land_full[0][:80, :720] if land_full.ndim == 3 else land_full[:80, :720]
    ocean = ~land_bool
    H, W = land_bool.shape

    # 气象通道索引
    met_names = data_cfg.get('met_channel_names', DEFAULT_MET_NAMES)
    print(f"📌 气象变量顺序: {met_names}")
    i_t2m = met_names.index('t2m')
    i_d2m = met_names.index('d2m') if 'd2m' in met_names else None

    test_start = int(0.9 * seqs.shape[0])
    # IHO 海区分划 (与 fig.1 一致, 见 region_defs.py)
    _, reg_masks = build_reg_masks(land_bool)

    # t2m 是标准化单位 (σ), 尝试加载 scaler 参数换算成 K
    sigma_K = None
    try:
        scaler = np.load(os.path.join('./npy', data_cfg['scaler_file']), allow_pickle=True).item()
        if 't2m' in scaler:
            sigma_K = float(scaler['t2m']['std'])
            print(f"📌 t2m 标准化 σ = {sigma_K:.2f} K")
    except Exception as e:
        print(f"⚠️ 未能加载 scaler 参数 ({e}), 温度差以 σ 为单位")
    km = sigma_K if sigma_K else 1.0
    unit = 'K' if sigma_K else 'σ'

    # ---- 训练期逐像元 t2m 气候平均 (归一化单位; seqs 通道序 = config.channel_names, 1='t2m') ----
    t2m_seq_ch = data_cfg['channel_names'].index('t2m')
    print(f'计算训练期逐像元 t2m 气候平均 (样本 0..{test_start}, 通道 {t2m_seq_ch}) ...')
    clim_sum = np.zeros((H, W), dtype=np.float64)
    clim_cnt = 0
    _blk = 64
    for s0 in range(0, test_start, _blk):
        s1 = min(s0 + _blk, test_start)
        arr = np.asarray(seqs[s0:s1, :, t2m_seq_ch], dtype=np.float32)[:, :, :H, :W]
        clim_sum += arr.sum(axis=(0, 1))
        clim_cnt += arr.shape[0] * arr.shape[1]
        if (s0 // _blk) % 4 == 0:
            print(f'  气候平均进度 {s1}/{test_start}', flush=True)
    pixel_clim = (clim_sum / clim_cnt).astype(np.float32)   # (H,W) 归一化单位
    print(f'  完成 ({clim_cnt} 个训练期日像元)')

    # ---- 逐日期: 转冻像元 + 首冻日温度 + 漏警/预警分组 (气象场每天只加载一次) ----
    obs_f_all = np.zeros((N, H, W), dtype=np.int16)
    trans_mask = np.zeros((N, H, W), dtype=bool)
    eA = np.full((N, H, W), 99, dtype=np.int16)
    eB = np.full((N, H, W), 99, dtype=np.int16)
    t2m_ff = np.zeros((N, H, W), dtype=np.float32)   # 各像元"首冻日"的 t2m
    for j, i in enumerate(range(test_start, test_start + N)):
        sic0 = np.asarray(seqs[i, -1, 0], dtype=np.float32)[:80, :720]
        open0 = (sic0 < THRESH) & ocean
        obs_f = first_freeze_horizon(obs[j], ocean)
        obs_f_all[j] = obs_f
        tmask = open0 & (obs_f <= Nh)
        trans_mask[j] = tmask
        hA = first_freeze_horizon(A[j], ocean)
        hB = first_freeze_horizon(B[j], ocean)
        eA[j] = np.where(tmask, np.where(hA <= Nh, hA - obs_f, 99), 99)
        eB[j] = np.where(tmask, np.where(hB <= Nh, hB - obs_f, 99), 99)
        metj = np.asarray(met[test_start + j], dtype=np.float32)[:, :, :80, :720]
        idx = np.clip(obs_f, 1, Nh) - 1              # 首冻日 met 索引 (0-based)
        t2m_ch = metj[:, i_t2m]                      # (10,H,W) t2m 时效序列
        t2m_ff[j] = np.take_along_axis(t2m_ch, idx[None].astype(np.int64), axis=0)[0]  # 逐像元取首冻日
        if (j + 1) % 10 == 0 or j == N - 1:
            print(f"  进度 {j+1}/{N}", flush=True)

    # ---- 逐像元距平: 首冻日 t2m 减去该像元训练期气候平均 (归一化), 再乘 σ 换算成 K ----
    t2m_anom = (t2m_ff - pixel_clim[None]) * km     # (N,H,W); km=1 时单位为 σ

    lines = [f"A={a_name} (含FDD)  vs  B={b_name} (无FDD)",
             "转冻像元 = 起报日SIC<0.15 且 10天内观测冻结; 漏警 = 10天内从未预报冻结",
             "", "== 漏警机制诊断: 漏警 vs 预警成功 像元在各自首冻日的 t2m 距平 (两模型结果合并) =="]
    lines.append(f"   距平 = 首冻日 t2m − 该像元训练期(测试集之前)气候平均, 单位 {unit}; "
                 "正值 = 比该像元同期气候态更暖")
    lines.append("   ΔT(漏警-预警) 数学上与原始温度差相同 (气候平均在差值中抵消); "
                 "正值 = 漏警发生在更暖的日子 -> 暖平流控制")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    reg_names = list(reg_masks.keys())
    dT_means, bothmiss_rates, warm_shares = [], [], []
    miss_json = {'A': a_name, 'B': b_name, 'sigma_K': sigma_K, 'unit': unit,
                 'anomaly_def': 'first-freeze-day t2m minus training-period per-pixel climatological mean',
                 'regions': {}}

    for rn, m2d in reg_masks.items():
        dTs, both_miss_frac = [], []
        tMiss, tWarn = [], []          # 各自首冻日 t2m 距平 (逐日期均值, K)
        for j in range(N):
            sel = trans_mask[j] & m2d
            if not sel.any():
                dTs.append(np.nan); both_miss_frac.append(np.nan)
                tMiss.append(np.nan); tWarn.append(np.nan); continue
            miss_both = sel & (eA[j] == 99) & (eB[j] == 99)
            warned = sel & (eA[j] != 99) & (eB[j] != 99)
            both_miss_frac.append(miss_both.sum() / sel.sum())
            if miss_both.sum() < 3 or warned.sum() < 3:
                dTs.append(np.nan); tMiss.append(np.nan); tWarn.append(np.nan); continue
            tm = float(t2m_anom[j][miss_both].mean()); tw = float(t2m_anom[j][warned].mean())
            tMiss.append(tm); tWarn.append(tw)
            dTs.append(tm - tw)
        dTs = np.array(dTs); valid = ~np.isnan(dTs)
        dTs_v = dTs[valid]
        tMiss_v = np.array(tMiss)[valid]; tWarn_v = np.array(tWarn)[valid]
        both = np.array(both_miss_frac); both = both[~np.isnan(both)]
        lo, hi = bootstrap_ci(dTs_v)
        try:
            p_w = float(wilcoxon(dTs_v)[1])
        except ValueError:
            p_w = 1.0
        dT_means.append(dTs_v.mean()); bothmiss_rates.append(both.mean())
        warm_frac = float((dTs_v > 0).mean())          # 漏警更暖的日期占比
        warm_shares.append(warm_frac)
        miss_json['regions'][rn] = {
            'unit': unit,
            'tmiss': {'mean': float(tMiss_v.mean()), 'sd': float(np.std(tMiss_v, ddof=1))},
            'twarn': {'mean': float(tWarn_v.mean()), 'sd': float(np.std(tWarn_v, ddof=1))},
            'diff': {'mean': float(dTs_v.mean()), 'sd': float(np.std(dTs_v, ddof=1)),
                     'ci_lo': lo, 'ci_hi': hi, 'p_wilcoxon': p_w},
            'warm_frac': warm_frac, 'n': int(dTs_v.size), 'both_miss_rate': float(both.mean())}
        lines.append(f"  {rn:>10s}: T_miss={tMiss_v.mean():+.3f}±{np.std(tMiss_v, ddof=1):.3f} {unit} "
                     f"T_warn={tWarn_v.mean():+.3f}±{np.std(tWarn_v, ddof=1):.3f} {unit}")
        lines.append(f"  {'':>10s}  ΔT(漏警-预警)={dTs_v.mean():+.3f}±{np.std(dTs_v, ddof=1):.3f} {unit} "
                     f"95%CI[{lo:+.3f},{hi:+.3f}] {unit} "
                     f"| Wilcoxon p={p_w:.4g} (更暖占比={warm_frac:5.1%}) "
                     f"| n={dTs_v.size}日 | 双模型漏警率={both.mean():6.2%}")

    # 图: 左 = 各海区漏警日 vs 预警日 t2m 距平差 (均值条); 右 = 配对案例更暖占比 (与论文 Fig.7 题注一致)
    axes[0].bar(range(len(reg_names)), dT_means, color='tab:orange', alpha=0.8)
    axes[0].axhline(0, color='k', lw=0.8)
    axes[0].set_xticks(range(len(reg_names)))
    axes[0].set_xticklabels([REG_EN[n] for n in reg_names], rotation=20, fontsize=11)
    axes[0].set_ylabel(f't2m anomaly on first-freeze day: missed − warned ({unit})', fontsize=12)
    axes[0].grid(alpha=0.3, axis='y')

    axes[1].bar(range(len(reg_names)), warm_shares, color='tab:purple', alpha=0.8)
    axes[1].set_xticks(range(len(reg_names)))
    axes[1].set_xticklabels([REG_EN[n] for n in reg_names], rotation=20, fontsize=11)
    axes[1].set_ylabel('Share of paired cases that are warmer', fontsize=12)
    axes[1].grid(alpha=0.3, axis='y')
    plt.tight_layout()

    out_dir = os.path.join('checkpoints', f'配对比较_{a_name}_vs_{b_name}')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'miss_diagnosis.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(out_dir, 'miss_diagnosis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    with open(os.path.join(out_dir, 'miss_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(miss_json, f, ensure_ascii=False, indent=1)
    print(f"\n💾 报告: {out_dir}/miss_diagnosis_report.txt")
    print(f"💾 统计JSON: {out_dir}/miss_stats.json")
    print(f"💾 图: {out_dir}/miss_diagnosis.png")


if __name__ == '__main__':
    main()
