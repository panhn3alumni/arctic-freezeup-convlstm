#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9b_advection.py —— 在 step9 温度距平基础上, 实算"温度平流"公式, 直接检验
"漏报是否集中在暖平流事件"这一假说 (step9 之前只算了 t2m 距平, 没算平流)。

温度平流(标准, 正=暖平流) = -V·∇T = -(u·∂T/∂x_east + v·∂T/∂y_north)
  met 数组纬度轴为降序(85→65), np.gradient(axis=-2) 得到的是"向南"梯度 ∂T/∂y_south
  = -(∂T/∂y)_north, 故:
      A = -(u·∂T/∂x_east + v·[-(∂T/∂y)_north])
        = -u·∂T/∂x_east + v·(∂T/∂y)_north
        = -u·dTdx + v·dTdy_south
  等价于 A = -thadv (step2 里 thadv = u·dTdx - v·dTdy_south = V·∇T, 正=冷平流)。
  met 通道已标准化(z-score), 故 A 为无量纲"暖平流指数", 符号与物理平流一致,
  量级受各变量 std 缩放 —— 本脚本只关心 missed vs warned 的符号/相对比较。

对每个转冻像元取其"观测首冻日"时效的 A, 比较 两模型都漏警 vs 都预警 的像元。
  ΔA = A_miss - A_warn > 0 且显著 -> 漏报确实发生在更强暖平流条件下 (假说坐实)
  ΔA 不显著 -> 只能说"漏报发生在更暖的温度条件", 不能断言是暖平流机制

用法: python step9b_advection.py --compare 第6次全通道重训 第7次无fdd重训
输出: checkpoints/配对比较_{A}_vs_{B}/miss_advection_report.txt
"""
import os
import json
import argparse
import numpy as np
import yaml
from scipy.stats import wilcoxon

from region_defs import build_reg_masks

THRESH = 0.15
N_BOOT = 10000
SEED = 42
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
        for root in ('checkpoints_thadv', 'checkpoints'):
            d = os.path.join(root, name)
            if os.path.exists(os.path.join(d, 'test_preds.npy')):
                return d
        raise SystemExit(f"找不到 {name}/test_preds.npy")

    dir_a = find_group_dir(a_name)
    dir_b = find_group_dir(b_name)
    A = np.load(os.path.join(dir_a, 'test_preds.npy')).astype(np.float32)
    B = np.load(os.path.join(dir_b, 'test_preds.npy')).astype(np.float32)
    obs = np.load(os.path.join(dir_a, 'test_obs.npy')).astype(np.float32)
    N, Nh = A.shape[0], A.shape[1]

    seqs = np.load(os.path.join('./npy', data_cfg['seq_file']), mmap_mode='r')
    met = np.load('./npy/targets_met_seasonal.npy', mmap_mode='r')
    land_full = np.load(os.path.join('./npy', data_cfg['mask_file'])).astype(bool)
    land_bool = land_full[0][:80, :720] if land_full.ndim == 3 else land_full[:80, :720]
    ocean = ~land_bool
    H, W = land_bool.shape

    met_names = data_cfg.get('met_channel_names', DEFAULT_MET_NAMES)
    print(f"📌 气象变量顺序: {met_names}")
    i_t2m = met_names.index('t2m')
    i_u10 = met_names.index('u10')
    i_v10 = met_names.index('v10')

    test_start = int(0.9 * seqs.shape[0])
    _, reg_masks = build_reg_masks(land_bool)

    obs_f_all = np.zeros((N, H, W), dtype=np.int16)
    trans_mask = np.zeros((N, H, W), dtype=bool)
    eA = np.full((N, H, W), 99, dtype=np.int16)
    eB = np.full((N, H, W), 99, dtype=np.int16)
    # 每个像元"首冻日"的暖平流指数 A (= -V·∇T, 正=暖平流), 由逐时效 2D 平流场按 obs_f 索引
    A_ff = np.zeros((N, H, W), dtype=np.float32)

    for j in range(N):
        sic0 = np.asarray(seqs[test_start + j, -1, 0], dtype=np.float32)[:80, :720]
        open0 = (sic0 < THRESH) & ocean
        obs_f = first_freeze_horizon(obs[j], ocean)
        obs_f_all[j] = obs_f
        tmask = open0 & (obs_f <= Nh)
        trans_mask[j] = tmask
        hA = first_freeze_horizon(A[j], ocean)
        hB = first_freeze_horizon(B[j], ocean)
        eA[j] = np.where(tmask, np.where(hA <= Nh, hA - obs_f, 99), 99)
        eB[j] = np.where(tmask, np.where(hB <= Nh, hB - obs_f, 99), 99)

        # 逐时效算 2D 暖平流场 A_all (10,H,W)
        metj = np.asarray(met[test_start + j], dtype=np.float32)[:, :, :80, :720]  # (10,C,H,W)
        T2 = metj[:, i_t2m]   # (10,H,W)
        U = metj[:, i_u10]
        V = metj[:, i_v10]
        dTdx = np.gradient(T2, axis=-1)        # 向东梯度 (lon 升序)
        dTdy = np.gradient(T2, axis=-2)        # 向南梯度 (lat 降序)
        A_all = -U * dTdx + V * dTdy           # 标准暖平流指数, 正=暖平流
        idx = np.clip(obs_f, 1, Nh) - 1
        A_ff[j] = np.take_along_axis(A_all, idx[None].astype(np.int64), axis=0)[0]
        if (j + 1) % 10 == 0 or j == N - 1:
            print(f"  进度 {j+1}/{N}", flush=True)

    lines = [f"A={a_name} (含FDD)  vs  B={b_name} (无FDD)",
             "转冻像元 = 起报日SIC<0.15 且 10天内观测冻结; 漏警 = 10天内从未预报冻结",
             "", "== 暖平流诊断: 漏警 vs 预警成功 像元在各自首冻日的温度平流指数 A=-V·∇T (归一化单位, 正=暖平流) ==",
             "   ΔA(漏警-预警)>0 且显著 => 漏报确发生在更强暖平流条件下 (step9 假说坐实)",
             "   ΔA 不显著 => 仅能说漏报在更暖温度条件下, 不能断言暖平流机制", ""]

    reg_names = list(reg_masks.keys())
    dA_means, warmadv_shares = [], []
    adv_json = {'A': a_name, 'B': b_name,
                'advection_def': '-V·∇T computed from normalized u10/v10/t2m in met targets; positive = warm advection',
                'regions': {}}
    for rn, m2d in reg_masks.items():
        dAs, aMiss, aWarn = [], [], []
        for j in range(N):
            sel = trans_mask[j] & m2d
            if not sel.any():
                dAs.append(np.nan); aMiss.append(np.nan); aWarn.append(np.nan); continue
            miss_both = sel & (eA[j] == 99) & (eB[j] == 99)
            warned = sel & (eA[j] != 99) & (eB[j] != 99)
            if miss_both.sum() < 3 or warned.sum() < 3:
                dAs.append(np.nan); aMiss.append(np.nan); aWarn.append(np.nan); continue
            am = float(A_ff[j][miss_both].mean()); aw = float(A_ff[j][warned].mean())
            aMiss.append(am); aWarn.append(aw); dAs.append(am - aw)
        dAs = np.array(dAs); valid = ~np.isnan(dAs)
        dAs_v = dAs[valid]
        aMiss_v = np.array(aMiss)[valid]; aWarn_v = np.array(aWarn)[valid]
        lo, hi = bootstrap_ci(dAs_v)
        try:
            p_w = float(wilcoxon(dAs_v)[1])
        except ValueError:
            p_w = 1.0
        dA_means.append(dAs_v.mean())
        warmadv_share = float((dAs_v > 0).mean())
        warmadv_shares.append(warmadv_share)
        adv_json['regions'][rn] = {
            'A_miss': {'mean': float(aMiss_v.mean()), 'sd': float(np.std(aMiss_v, ddof=1))},
            'A_warn': {'mean': float(aWarn_v.mean()), 'sd': float(np.std(aWarn_v, ddof=1))},
            'diff': {'mean': float(dAs_v.mean()), 'sd': float(np.std(dAs_v, ddof=1)),
                     'ci_lo': lo, 'ci_hi': hi, 'p_wilcoxon': p_w},
            'warm_advection_share': warmadv_share, 'n': int(dAs_v.size)}
        lines.append(f"  {rn:>14s}: A_miss={aMiss_v.mean():+.4f}±{np.std(aMiss_v, ddof=1):.4f} "
                     f"A_warn={aWarn_v.mean():+.4f}±{np.std(aWarn_v, ddof=1):.4f}")
        lines.append(f"  {'':>14s}  ΔA(漏警-预警)={dAs_v.mean():+.4f}±{np.std(dAs_v, ddof=1):.4f} "
                     f"95%CI[{lo:+.4f},{hi:+.4f}] | Wilcoxon p={p_w:.4g} "
                     f"(更强暖平流占比={warmadv_share:5.1%}) | n={dAs_v.size}日")
        verdict = "✓ 暖平流更强(显著)" if (p_w < 0.05 and dAs_v.mean() > 0) else (
                  "✗ 无显著暖平流信号" if p_w >= 0.05 else "⚠ 方向相反")
        lines.append(f"  {'':>14s}  => {verdict}")

    out_dir = os.path.join('checkpoints', f'配对比较_{a_name}_vs_{b_name}')
    os.makedirs(out_dir, exist_ok=True)
    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(out_dir, 'miss_advection_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    with open(os.path.join(out_dir, 'miss_advection_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(adv_json, f, ensure_ascii=False, indent=1)
    print(f"\n💾 报告: {out_dir}/miss_advection_report.txt")
    print(f"💾 统计JSON: {out_dir}/miss_advection_stats.json")


if __name__ == '__main__':
    main()
