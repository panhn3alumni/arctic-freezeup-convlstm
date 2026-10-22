#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_nep_shipping_synthesis.py —— 东北航道(NEP)航运导向的合成分析（今晚必跑，CPU，不重训）
===========================================================================================
用途：把"第1次SIC修复全通道(含FDD)" vs "第2次无fdd" 的**已有预测结果**，从"画差异图"
      提升为"航运决策要用的问题"。输出的每个数字/图都直接可以写进论文。

你要画的图/表的原始数据都已存在（checkpoints/{组名}/test_preds.npy 等），本脚本只读不走训练。

它在 3 个问题上给你论文弹药 (2026-10-13 口径: onset 漏警率只取 T+10 单时效;
快封冻三分位复核节已删除 —— 论文不再引用):
  1) 东北航道每段海(Kara/Laptev/ESiberian/Chukchi/Barents)在起报后第几天"封死"？
     （开阔水→冻结 的"通航段关闭时间线"，这是航运最关心的）
  2) 首次封冻 onset 预警：每个海区在 T+10 的漏警率（观测已封冻但模型没报）
  3) 关键科学问题的廉价证据：FDD 和 t2m 到底多冗余？
     （"LSTM 自己能积 t2m，那 FDD 是不是没用"——用逐像元相关直接量给你看）

用法（工程目录下，和 step7 一样）：
  python step10_nep_shipping_synthesis.py
输入两张表： 第1次SIC修复全通道  第2次无fdd   （可改下面前两个字符串）
输出：
  checkpoints/配对比较_第1次SIC修复全通道_vs_第2次无fdd/nep_shipping_report.txt
  checkpoints/配对比较_第1次SIC修复全通道_vs_第2次无fdd/*.png
"""
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from region_defs import build_reg_masks, REG_EN

# SCI 图: 衬线字体 (Times New Roman 优先, 服务器无则回退度量兼容的 Liberation Serif)
_names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# ============ 可以手动改的两行 ============
A_NAME = "第6次全通道重训"        # 含 FDD (完整训练版 A')
B_NAME = "第7次无fdd重训"         # 无 FDD (完整训练版 B')
# =========================================

THRESH = 0.15                     # 冻结的 SIC 判定阈值（与 step7 一致，航道“不可通”也可再试 0.5）


def load_group(name):
    A_dir = f'checkpoints/{name}'
    preds = np.load(os.path.join(A_dir, 'test_preds.npy')).astype(np.float32)
    obs = np.load(os.path.join(A_dir, 'test_obs.npy')).astype(np.float32)
    dates = list(np.load(os.path.join(A_dir, 'test_dates.npy'), allow_pickle=True))
    return preds, obs, dates


def main():
    A, obs, dates = load_group(A_NAME)
    B, _, _ = load_group(B_NAME)

    # --- 空间与掩膜（与 step7 完全一致; IHO 海区分划见 region_defs.py）---
    land_full = np.load('./npy/land_mask_static.npy').astype(bool)
    land = land_full[0][:80, :720] if land_full.ndim == 3 else land_full[:80, :720]
    ocean = ~land
    H, W = land.shape
    N, Nh = A.shape[0], A.shape[1]        # 117 日期 × 10 时效
    _, reg_masks = build_reg_masks(land)
    sea_names = [k for k in reg_masks if k != 'pan']

    # --- 起报日开阔水（SIC<0.15）格点，用于"这段海会不会在10天内封死" ---
    seqs = np.load('./npy/sequences_seasonal.npy', mmap_mode='r')
    test_start = int(0.9 * seqs.shape[0])

    # 逐日"通航段关闭"：起报日为开阔水 & 观测在未来第 k 天已冻结 的占比（按海区、按时效）
    # 行=海区，列=时效
    close_obs = np.zeros((len(sea_names), Nh))
    close_fa  = np.zeros((len(sea_names), Nh))   # 含FDD模型预报的"即将冻结"
    close_fb  = np.zeros((len(sea_names), Nh))   # 无FDD模型预报的"即将冻结"
    froze_cnt = np.zeros((len(sea_names), Nh))   # 真封冻的格点总数(分母)
    miss_A    = np.zeros((len(sea_names), Nh))   # FDD漏掉真正封冻的格点数
    miss_B    = np.zeros((len(sea_names), Nh))

    for r, sea in enumerate(sea_names):
        m = reg_masks[sea]
        print(f"[1] 处理 {sea} ...")
        ns = int(m.sum())
        for lead in range(Nh):
            tot = 0.0
            for j in range(N):
                sic0 = np.asarray(seqs[test_start + j, -1, 0, :H, :W]).astype(np.float32)
                open0 = (sic0 < THRESH) & m
                n0 = int(open0.sum())
                tot += n0
                froze = obs[j, lead][open0] >= THRESH            # 观测封冻(限定在起报日开阔水的格点上)
                if n0 == 0:
                    continue
                pf_a = A[j, lead][open0] >= THRESH              # FDD模型报封
                pf_b = B[j, lead][open0] >= THRESH              # 无FDD模型报封
                close_obs[r, lead] += froze.sum()
                close_fa[r, lead] += pf_a.sum()
                close_fb[r, lead] += pf_b.sum()
                froze_cnt[r, lead] += int(froze.sum())
                miss_A[r, lead] += int((froze & ~pf_a).sum())   # FDD漏报封冻
                miss_B[r, lead] += int((froze & ~pf_b).sum())   # 无FDD漏报封冻
            # 平均成占比
            if tot > 0:
                close_obs[r, lead] /= tot
                close_fa[r, lead] /= tot
                close_fb[r, lead] /= tot
            if froze_cnt[r, lead] > 0:                          # 漏警率 = 漏报 / 真封冻
                miss_A[r, lead] /= froze_cnt[r, lead]
                miss_B[r, lead] /= froze_cnt[r, lead]

    # --- 组装报告 ---
    lines = []
    lines.append("=" * 78)
    lines.append(f"NEP 航运导向合成分析  A={A_NAME}(含FDD) vs B={B_NAME}(无FDD)")
    lines.append(f"测试集: {N} 个独立日期 x 时效 T+1..T+10, 冻结阈值 SIC>={THRESH}")
    lines.append("=" * 78)

    lines.append("\n[1] 通航段关闭时间线 (起报日为开阔水, 到第T+k天已封冻的占比)")
    lines.append("    行=海区, 列=T+1..T+10;  这是'哪段航道在第几天被封死'——航运第一问")
    lines.append("    格式:  观测 | FDD模型 | 无FDD模型")
    lines.append("    列: " + "  ".join([f"T+{k}" for k in range(1, Nh + 1)]))
    for r, sea in enumerate(sea_names):
        lines.append(f"  {sea:>10s} 观测: " + " ".join([f"{close_obs[r, k]:.2f}" for k in range(Nh)]))
        lines.append(f"  {'':>10s} FDD:  " + " ".join([f"{close_fa[r, k]:.2f}" for k in range(Nh)]))
        lines.append(f"  {'':>10s} 无FDD: " + " ".join([f"{close_fb[r, k]:.2f}" for k in range(Nh)]))

    k10 = Nh - 1   # T+10 单时效 (2026-10-13 口径)
    lines.append("\n[2] 首次封冻预警(onset): 各海区 T+10 的漏警率")
    lines.append("    漏警率 = 观测 T+10 已封冻但模型 T+10 没报 / 观测 T+10 封冻总数  （航运最怕：船被突然封在里面）")
    lines.append(f"  {'海区':>10s} | {'FDD漏警':>8s} | {'无FDD漏警':>9s} | 降低量")
    for r, sea in enumerate(sea_names):
        a, b = miss_A[r, k10], miss_B[r, k10]
        lines.append(f"  {sea:>10s} | {a:8.3f} | {b:9.3f} | {(a-b)/max(b,1e-6)*100:+6.1f}%")

    lines.append("\n[3] FDD 与 t2m 冗余度诊断 (回答'LSTM自己能积t2m,FDD是否无用')")
    lines.append("    逐像元相关(前300样本/海洋格点): 相关越高=越冗余. FDD>0 的部分仍是强非线性")
    nsamp = min(300, seqs.shape[0])
    vec_t = np.asarray(seqs[:nsamp, -1, 1, :H, :W]).astype(np.float32)[:, ocean].ravel()
    for lab in ['fdd2', 'fdd4', 'fdd6', 'fdd8', 'fdd10', 'fdd15', 'fdd20']:
        idx = channel_id(lab)
        if idx is None:
            lines.append(f"    {lab}: 未在当前通道列表, 跳过")
            continue
        vec_f = np.asarray(seqs[:nsamp, -1, idx, :H, :W]).astype(np.float32)[:, ocean].ravel()
        r = float(np.corrcoef(vec_t, vec_f)[0, 1]) if len(vec_t) > 10 else float('nan')
        lines.append(f"    {lab} vs t2m: r = {r:+.3f}")

    report = '\n'.join(lines)
    print(report)
    os.makedirs(f'checkpoints/配对比较_{A_NAME}_vs_{B_NAME}', exist_ok=True)
    with open(f'checkpoints/配对比较_{A_NAME}_vs_{B_NAME}/nep_shipping_report.txt', 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f"\n💾 报告: checkpoints/配对比较_{A_NAME}_vs_{B_NAME}/nep_shipping_report.txt")

    # --- 图1: 通航段关闭时间线 (只画 NEP 沿线 5 海区, 不含中央北极/其他海域) ----
    NEP = ['barents', 'kara', 'laptev', 'esiberian', 'chukchi']
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    leads = np.arange(1, Nh + 1)
    for sea in NEP:
        r = sea_names.index(sea)
        ax.plot(leads, close_obs[r], marker='o', label=f'{REG_EN[sea]} (obs)')
    ax.set_xlabel('Forecast lead (days)', fontsize=12)
    ax.set_ylabel('Fraction of open-water cells that freeze', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=11)
    plt.tight_layout(); plt.savefig(f'checkpoints/配对比较_{A_NAME}_vs_{B_NAME}/nep_route_closure.png', dpi=180)
    plt.close(fig)

    # --- 图2: 各海区漏警率对比 (只画 NEP 沿线 5 海区, T+10 单时效) ----
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(NEP))
    w = 0.38
    idx_nep = [sea_names.index(s) for s in NEP]
    ax.bar(x - w / 2, miss_A[idx_nep, k10], w, label='FDD (full)')
    ax.bar(x + w / 2, miss_B[idx_nep, k10], w, label='No-FDD')
    ax.set_xticks(x); ax.set_xticklabels(NEP, rotation=30)
    ax.set_ylabel('First-freeze missed rate'); ax.set_title('Onset miss rate per NEP sea (T+10)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(f'checkpoints/配对比较_{A_NAME}_vs_{B_NAME}/nep_onset_miss.png', dpi=180)
    plt.close(fig)

    print(f"💾 图: nep_route_closure.png / nep_onset_miss.png")


def channel_id(lab):
    """根据 step2 通道顺序返回 fdd 通道号。固定 15 通道顺序。"""
    order = ['SIC', 't2m', 'd2m', 'u10', 'v10', 'sp_grad_x', 'sp_grad_y',
             'fdd2', 'fdd4', 'fdd6', 'fdd8', 'fdd10', 'fdd15', 'fdd20', 'land_mask']
    return order.index(lab) if lab in order else None


if __name__ == '__main__':
    main()