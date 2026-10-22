#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step18_build_tables.py —— 从统计 JSON 自动构建论文 Table 1-4 (CPU, 秒级)

论文位置与引用:
  Table 1  §4.2 "...in the rapid-freeze tercile (... Table 1; Fig. 4)."   <- stratify_stats.json
  Table 2  §4.2 "...concentrates along the Northeast-Passage corridor (Table 2; Fig. 5)." <- edge_stats.json
  Table 3  §4.2 caption after Table 2 (equivalent ice-edge displacement)  <- edge_stats.json
  Table 4  §4.4 "...misses occur on significantly warmer days (Table 4; Fig. 8)."  <- miss_stats.json

为什么有这个脚本 (红线: 手抄数字必错):
  表内所有数字直接从统计脚本输出的 JSON 读取, 不经人工转录; 缺文件会显式报
  MISSING 并提示重跑对应 step, 不允许静默编造。

统计规范 (2026-10-09, 全稿统一):
  两模型各自 mean±sd 与 min–max; 配对差 Δ = FDD − no-FDD, bootstrap 95% CI
  (10000 次重采样, seed=42); 显著性用双侧 Wilcoxon 符号秩检验; 不再使用
  "胜率/符号检验" 作为论文统计量。

用法:
  python step18_build_tables.py
  python step18_build_tables.py --cmp-dir "checkpoints/配对比较_第1次SIC修复全通道_vs_第2次无fdd"

输出: results/figures/tables_draft.md  (Markdown 表格 + 英文表题, 直接粘贴进正文)
"""

import os
import json
import argparse

OUT_DEFAULT = 'results/figures/tables_draft.md'


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} 不存在 — 请先重跑对应统计 step')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fmt_p(p):
    if p < 1e-4:
        return '<0.0001'
    if p < 1e-3:
        return '<0.001'
    return f'{p:.3f}'


def build_table1(st):
    """st = stratify_stats.json"""
    rows = []
    ter_en = {'slow': 'Slow-freeze (lower tercile)',
              'mid': 'Mid-freeze (middle tercile)',
              'rapid': 'Rapid-freeze (upper tercile)'}
    for key in ('slow', 'mid', 'rapid'):
        t = st['terciles'][key]
        for metric, scale, dec in (('MAE', 100.0, 2), ('IIEE', 1.0, 1)):
            s = t[metric]
            A, B, D = s['A'], s['B'], s['diff']
            rows.append('| ' + ' | '.join([
                ter_en[key] if metric == 'MAE' else '',
                f'{metric} (×10⁻²)' if metric == 'MAE' else f'{metric} (grid cells)',
                f'{A["mean"] * scale:.{dec}f} ± {A["sd"] * scale:.{dec}f}',
                f'{A["min"] * scale:.{dec}f}\u2013{A["max"] * scale:.{dec}f}',
                f'{B["mean"] * scale:.{dec}f} ± {B["sd"] * scale:.{dec}f}',
                f'{B["min"] * scale:.{dec}f}\u2013{B["max"] * scale:.{dec}f}',
                f'{D["mean"] * scale:+.{dec}f} [{D["ci_lo"] * scale:+.{dec}f}, {D["ci_hi"] * scale:+.{dec}f}]',
                fmt_p(D['p_wilcoxon'])]) + ' |')
    return rows


def build_table2(ej):
    """ej = edge_stats.json (快封冻期分海区 FN, IHO 分区)"""
    reg_en = {'barents': 'Barents', 'kara': 'Kara', 'laptev': 'Laptev',
              'esiberian': 'East Siberian', 'chukchi': 'Chukchi',
              'arctic_central': 'Central Arctic', 'other': 'Other (open ocean)'}
    rows = []
    for rn in ('barents', 'kara', 'laptev', 'esiberian', 'chukchi',
               'arctic_central', 'other'):
        e = ej['regional_FN_fast'].get(rn)
        if e is None:
            rows.append(f'| {reg_en[rn]} | MISSING | MISSING | MISSING | MISSING | '
                        f'MISSING (重跑 step7 --compare) |')
            continue
        s, D, rel = e['stats'], e['stats']['diff'], e['rel_pct']
        rows.append('| ' + ' | '.join([
            reg_en[rn],
            f'{s["A"]["mean"] * 100:.2f}',
            f'{s["B"]["mean"] * 100:.2f}',
            f'{D["mean"] * 100:+.2f} [{D["ci_lo"] * 100:+.2f}, {D["ci_hi"] * 100:+.2f}]',
            f'{rel:+.1f}',
            fmt_p(D['p_wilcoxon'])]) + ' |')
    return rows


def build_table3(ej):
    """ej = edge_stats.json (等效冰缘位移)"""
    ter_en = {'slow': 'Slow', 'mid': 'Mid', 'rapid': 'Rapid'}
    rows = []
    for key in ('slow', 'mid', 'rapid'):
        d = ej['displacement'][key]
        rows.append('| ' + ' | '.join([
            ter_en[key],
            f'{d["A_km"]:.2f}',
            f'{d["B_km"]:.2f}',
            f'{d["diff_km"]:+.2f} [{d["ci_lo"]:+.2f}, {d["ci_hi"]:+.2f}]',
            f'{d["rel_pct"]:+.1f}']) + ' |')
    return rows


def build_table4(mj):
    """mj = miss_stats.json (IHO 分区)"""
    reg_en = {'barents': 'Barents', 'kara': 'Kara', 'laptev': 'Laptev',
              'esiberian': 'East Siberian', 'chukchi': 'Chukchi',
              'arctic_central': 'Central Arctic', 'other': 'Open ocean (other)'}
    km = mj.get('sigma_K') or 1.0
    rows = []
    for rn in ('barents', 'kara', 'laptev', 'esiberian', 'chukchi',
               'arctic_central', 'other'):
        e = mj['regions'].get(rn)
        if e is None:
            rows.append(f'| {reg_en[rn]} | MISSING | MISSING | MISSING | MISSING | '
                        f'MISSING | MISSING (重跑 step9 --compare) |')
            continue
        D = e['diff']
        rows.append('| ' + ' | '.join([
            reg_en[rn],
            f'{e["tmiss"]["mean"] * km:+.2f} ± {e["tmiss"]["sd"] * km:.2f}',
            f'{e["twarn"]["mean"] * km:+.2f} ± {e["twarn"]["sd"] * km:.2f}',
            f'{D["mean"]:+.3f}',
            f'{D["mean"] * km:+.2f}',
            fmt_p(D['p_wilcoxon']),
            str(e['n'])]) + ' |')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cmp-dir', default='checkpoints/配对比较_第1次SIC修复全通道_vs_第2次无fdd')
    args = ap.parse_args()
    d = args.cmp_dir
    os.makedirs('results/figures', exist_ok=True)

    st = load_json(os.path.join(d, 'stratify_stats.json'))
    ej = load_json(os.path.join(d, 'edge_stats.json'))
    mj = load_json(os.path.join(d, 'miss_stats.json'))

    L = []
    L.append('# 论文 Table 1-4 草表 (由 step18 自动从统计 JSON 读取, 数字未手抄)\n')
    L.append(f'数据来源目录: {d}\n')
    L.append('统计规范: mean ± sd 与 min–max; Δ = FDD − no-FDD; bootstrap 95% CI (10 000 resamples, '
             'seed=42); 双侧 Wilcoxon 符号秩检验。\n')

    # ---- Table 1 ----
    L.append('---\n')
    L.append('Table 1: Paired FDD-reference versus no-FDD-control comparison of 10-day SIC '
             'forecasts stratified by freeze-intensity tercile (117 date-level samples, 39 per '
             'tercile; MAE in units of 10⁻² SIC fraction, IIEE in grid cells; mean ± sd across '
             'dates with the min–max range; Δ is the mean paired difference FDD − no-FDD with '
             'the bootstrap 95 % CI (10 000 resamples); p from two-sided Wilcoxon signed-rank '
             'tests).\n')
    L.append('| Tercile | Metric | FDD mean ± sd | FDD min–max | No-FDD mean ± sd | No-FDD min–max | Δ [95 % CI] | p |')
    L.append('|---|---|---|---|---|---|---|---|')
    L.extend(build_table1(st))
    L.append('')
    c = st['corr']
    L.append(f"Freeze-intensity versus FDD-advantage correlation: "
             f"MAE Pearson {c['MAE']['pearson']:+.3f} (Spearman {c['MAE']['spearman']:+.3f}); "
             f"IIEE Pearson {c['IIEE']['pearson']:+.3f} (Spearman {c['IIEE']['spearman']:+.3f}).\n")

    # ---- Table 2 ----
    L.append('---\n')
    L.append('Table 2: Missed-freeze (FN) rates by region during the rapid-freeze tercile, '
             'FDD reference versus no-FDD control. FN rate is the fraction of ocean cells in '
             'the region; Δ is the mean paired difference in percentage points (pp) of ocean '
             'cells with the bootstrap 95 % CI (10 000 resamples); Δ (%) is the relative '
             'change; p from two-sided Wilcoxon signed-rank tests on date-level paired '
             'differences.\n')
    L.append('| Region | FN rate, FDD (%) | FN rate, no-FDD (%) | Δ (pp) [95 % CI] | Δ (%) | p |')
    L.append('|---|---|---|---|---|---|')
    L.extend(build_table2(ej))
    L.append('')

    # ---- Table 3 ----
    L.append('---\n')
    L.append('Table 3: Equivalent ice-edge displacement by freeze-intensity tercile (IIEE '
             'divided by the observed 15 % ice-edge length; mean over dates and lead times). '
             'Δ is FDD − no-FDD with a date-level bootstrap 95 % CI (10 000 resamples).\n')
    L.append('| Tercile | FDD (km) | No-FDD (km) | Δ (km) [95 % CI] | Δ (%) |')
    L.append('|---|---|---|---|---|')
    L.extend(build_table3(ej))
    L.append('')

    # ---- Table 4 ----
    L.append('---\n')
    L.append('Table 4: T2m anomaly (relative to the pixel climatological mean) at the '
             'first-freeze day of missed-versus-warned pixels (both models pooled): regional '
             'mean ± sd of each group in K, mean paired difference ΔT in σ and K with the '
             'bootstrap 95 % CI reported in the text, two-sided Wilcoxon signed-rank p, and '
             'the number of dates n.\n')
    L.append('| Region | Missed-day T (K) | Warned-day T (K) | ΔT (σ) | ΔT (K) | p | n |')
    L.append('|---|---|---|---|---|---|---|')
    L.extend(build_table4(mj))
    L.append('')
    mj_d = {r: mj['regions'][r]['diff'] for r in mj['regions']}
    L.append('Bootstrap 95 % CI of ΔT (σ): ' +
             '; '.join(f'{r} [{v["ci_lo"]:+.3f}, {v["ci_hi"]:+.3f}]' for r, v in mj_d.items()) + '\n')

    out = OUT_DEFAULT
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'💾 {out}')
    print('⚠ 请逐表核对方向与数值后再写入正文。')


if __name__ == '__main__':
    main()
