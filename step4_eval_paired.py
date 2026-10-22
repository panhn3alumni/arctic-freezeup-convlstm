#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_eval_paired.py —— 逐日期配对评估 + 分海区统计 (FDD 增量的决定性证据)

两种模式:
  模式1 (评估单组, config.yaml 需处于该组状态):
      python step4_eval_paired.py
      python step4_eval_paired.py --output_path 消融_no_fdd
      输出: checkpoints/{output_path}/perdate_metrics.csv
            每行 = (日期, 时效) x [模型与persistence的 TS/IIEE/AEE/MAE + 分海区IIEE]

  模式2 (两组配对比较, 需先各自跑过模式1):
      python step4_eval_paired.py --compare 第1次SIC修复全通道 消融_no_fdd
      输出: checkpoints/配对比较_{A}_vs_{B}/paired_report.txt
            每时效: IIEE 差值均值/胜率/符号检验 p 值 + 分海区对比

海区定义: IHO S-23 官方海区分划 (region_defs.build_reg_masks, 与 fig.1 一致):
  Barents / Kara / Laptev / East Siberian / Chukchi / Central Arctic / Other,
  多边形来自 MarineRegions.org, 掩膜 npy/region_mask_iho.npy
"""
import os
import argparse

import numpy as np
import torch
import yaml

from step2_prepare_data_new import get_channel_indices
from convlstm import SeaIcePredictor
from region_defs import build_reg_masks

THRESH = 0.15


def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def strip_prefix(sd):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}


def eval_one_group(out_name, cfg, device):
    """评估一组, 返回 perdate 记录列表"""
    keep_ch, met_keep_ch, use_names, _ = get_channel_indices(cfg)

    save_dir = os.path.join('checkpoints', out_name)
    best_path = os.path.join(save_dir, 'best_model.pth')
    checkpoint = torch.load(best_path, map_location=device)
    saved_cfg = checkpoint['config']
    in_ch = saved_cfg['model']['input_channels']
    if in_ch != len(keep_ch):
        raise SystemExit(f"❌ {out_name}: checkpoint是{in_ch}通道, config只选了{len(keep_ch)}个 —— "
                         f"请把 config.yaml 调回该组训练时的 use_channels 再跑")

    model = SeaIcePredictor(
        input_channels=in_ch,
        hidden_dim=saved_cfg['model']['hidden_dim'],
        kernel_size=saved_cfg['model']['kernel_size'],
        num_layers=saved_cfg['model']['num_layers'], pred_len=1).to(device)
    model.load_state_dict(strip_prefix(checkpoint['model_state_dict']))
    model.eval()
    print(f"✅ [{out_name}] epoch {checkpoint.get('epoch', '?')} 加载成功 ({in_ch}通道)")

    data_cfg = cfg['data']
    seqs = np.load(os.path.join('./npy', data_cfg['seq_file']), mmap_mode='r')
    targets = np.load(os.path.join('./npy', data_cfg['target_file']), mmap_mode='r')
    targets_met = np.load('./npy/targets_met_seasonal.npy', mmap_mode='r')
    land_full = np.load(os.path.join('./npy', data_cfg['mask_file'])).astype(bool)
    dates = np.load('./npy/sample_start_dates.npy', allow_pickle=True)

    # 与可视化脚本一致的空间裁剪 (80x720); IHO 海区分划统一取自 region_defs.py
    land_bool = land_full[0][:80, :720] if land_full.ndim == 3 else land_full[:80, :720]
    H, W = land_bool.shape
    ocean, reg_masks = build_reg_masks(land_bool)
    n_roll = cfg['data']['prediction_length']
    test_start = int(0.9 * seqs.shape[0])
    test_idx = list(range(test_start, seqs.shape[0]))
    mask_tensor = torch.from_numpy(land_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    # 经度带 -> 各区域格点数 (用于列说明)
    reg_names = list(reg_masks.keys())

    rows = []
    with torch.no_grad():
        for i in test_idx:
            x = torch.from_numpy(np.array(seqs[i]).astype(np.float32))[None]
            x = x[:, :, keep_ch, :H, :W].to(device)      # 筛通道 + 裁剪到 (80,720)
            y = torch.from_numpy(np.array(targets[i]).astype(np.float32))[None]  # (1,10,1,H,W)
            y = y[:, :, :, :H, :W]
            y_met = torch.from_numpy(np.array(targets_met[i]).astype(np.float32))[None]
            y_met = y_met[:, :, met_keep_ch, :H, :W].to(device)

            sic_last = x[:, -1, 0:1]  # persistence
            cur = x
            preds = []
            for k in range(n_roll):
                p = model(cur, mask=mask_tensor)[:, :, 0]
                preds.append(p[0, 0].cpu().numpy())  # (H,W)
                if k < n_roll - 1:
                    nxt = torch.cat([p.unsqueeze(2), y_met[:, k].unsqueeze(1),
                                     cur[:, -1:, -1:, :, :]], dim=2)
                    cur = torch.cat([cur[:, 1:], nxt], dim=1)

            obs_all = y[0, :, 0].cpu().numpy()          # (10,H,W)
            pers = sic_last[0, 0].cpu().numpy()          # (H,W)
            date_str = str(dates[i])[:10]

            for k in range(n_roll):
                pred_k = preds[k]
                obs_k = obs_all[k]
                row = {'date': date_str, 'horizon': k + 1}

                def pack(p):
                    pb = (p >= THRESH) & ocean
                    ob = (obs_k >= THRESH) & ocean
                    tp = int((pb & ob).sum()); fp = int((pb & ~ob).sum()); fn = int((~pb & ob).sum())
                    ts = tp / max(tp + fp + fn, 1)
                    iiee = fp + fn
                    aee = fp - fn
                    mae = float(np.abs(p[ocean] - obs_k[ocean]).mean())
                    return ts, iiee, aee, mae

                row['model_TS'], row['model_IIEE'], row['model_AEE'], row['model_MAE'] = pack(pred_k)
                row['pers_IIEE'], row['pers_AEE'], row['pers_MAE'] = pack(pers)[1:]

                # 分海区 IIEE (模型)
                for rn in reg_names:
                    m = reg_masks[rn]
                    pb = (pred_k >= THRESH) & m
                    ob = (obs_k >= THRESH) & m
                    row[f'{rn}_IIEE'] = int((pb ^ ob).sum())  # FP+FN = 异或
                rows.append(row)
            print(f"  [{out_name}] {date_str} 完成", end='\r')
    print()
    return rows, reg_names, use_names


def csv_write(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"💾 已保存: {path}")


def compare(mode_args):
    a_name, b_name = mode_args
    a_path = os.path.join('checkpoints', a_name, 'perdate_metrics.csv')
    b_path = os.path.join('checkpoints', b_name, 'perdate_metrics.csv')
    for p in (a_path, b_path):
        if not os.path.exists(p):
            raise SystemExit(f"❌ 缺少 {p} —— 请先用模式1分别评估两组")

    import csv
    def load(p):
        with open(p, 'r', encoding='utf-8') as f:
            return {(r['date'], int(r['horizon'])): r for r in csv.DictReader(f)}
    A, B = load(a_path), load(b_path)
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit("❌ 两组没有共同日期 (数据版本不一致?)")
    print(f"📊 配对样本: {len(common)} 个 (日期×时效)")

    reg_names = [k[:-5] for k in A[common[0]] if k.endswith('_IIEE')]
    lines = [f"配对比较: A={a_name}  vs  B={b_name}  (指标越小越好, 除TS/精度类)",
             f"配对数: {len(common)}", ""]

    def paired(metric, key=lambda r: float(r[metric])):
        da = np.array([key(A[c]) for c in common])
        db = np.array([key(B[c]) for c in common])
        diff = da - db
        win = float((diff < 0).mean())  # A 优于 B 的比例
        # 符号检验 (排除平局)
        pos = int((diff < 0).sum()); neg = int((diff > 0).sum())
        n = pos + neg
        from math import comb
        # 双侧二项检验 p 值
        k = min(pos, neg)
        if n > 0:
            pval = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
            pval = min(pval, 1.0)
        else:
            pval = 1.0
        return diff.mean(), win, pos, neg, pval

    lines.append("== 全域 IIEE (冰缘误差, 格点数; A<B 表示A更好) ==")
    for h in sorted(set(c[1] for c in common)):
        sub = [c for c in common if c[1] == h]
        da = np.array([float(A[c]['model_IIEE']) for c in sub])
        db = np.array([float(B[c]['model_IIEE']) for c in sub])
        d = da - db
        pos = int((d < 0).sum()); neg = int((d > 0).sum()); n = pos + neg
        from math import comb
        pval = min(1.0, sum(comb(n, i) for i in range(0, min(pos, neg) + 1)) / 2 ** n * 2) if n else 1.0
        lines.append(f"  T+{h:<2d} | A均值={da.mean():9.0f} B均值={db.mean():9.0f} | "
                     f"差={d.mean():+8.0f} | A胜率={pos/max(pos+neg,1):5.1%} | p={pval:.4f}")

    lines.append("")
    lines.append("== 全时效汇总 ==")
    for metric in ['model_IIEE', 'model_TS', 'model_MAE', 'pers_IIEE']:
        mean_d, win, pos, neg, pval = paired(metric)
        lines.append(f"  {metric:>12s}: A-B差均值={mean_d:+.4f} | A胜率={win:5.1%} | 符号检验 p={pval:.4f}")

    lines.append("")
    lines.append("== 分海区 IIEE (全时效汇总, A vs B) ==")
    for rn in reg_names:
        da = np.array([float(A[c][f'{rn}_IIEE']) for c in common])
        db = np.array([float(B[c][f'{rn}_IIEE']) for c in common])
        d = da - db
        pos = int((d < 0).sum()); neg = int((d > 0).sum()); n = pos + neg
        from math import comb
        pval = min(1.0, sum(comb(n, i) for i in range(0, min(pos, neg) + 1)) / 2 ** n * 2) if n else 1.0
        lines.append(f"  {rn:>10s}: A={da.mean():8.1f} B={db.mean():8.1f} | 差={d.mean():+7.1f} | "
                     f"A胜率={pos/max(pos+neg,1):5.1%} | p={pval:.4f}")

    report = '\n'.join(lines)
    print(report)
    out_dir = os.path.join('checkpoints', f'配对比较_{a_name}_vs_{b_name}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'paired_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f"\n💾 报告已保存: {os.path.join(out_dir, 'paired_report.txt')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_path', default=None)
    ap.add_argument('--compare', nargs=2, metavar=('GROUP_A', 'GROUP_B'))
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return

    cfg = load_config()
    out_name = args.output_path or cfg['infer']['output_path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, reg_names, use_names = eval_one_group(out_name, cfg, device)
    csv_write(rows, os.path.join('checkpoints', out_name, 'perdate_metrics.csv'))
    print(f"\n下一步(两组都评完后): python step4_eval_paired.py --compare {out_name} 消融_no_fdd")


if __name__ == '__main__':
    main()
