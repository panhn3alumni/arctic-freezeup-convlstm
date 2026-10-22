#!/usr/bin/env python3
"""
prepare_data_burst_windows_v3.py (step2_prepare_data_new_v3)
基于海冰爆发日与变化率转正日确定每年窗口，加入动力特征。
窗口: D0 - 10 天 → 爆发 + 10 天
通道: SIC, t2m, d2m, u10, v10, sp_grad_x, sp_grad_y, fdd系列, land_mask

★ v3 改动 (相对 v2, 修复内存溢出):
  - 只加载窗口内日期的 ERA5 数据 (含 FDD 所需前 20 天 padding),
    不再加载全 30 年 (10958 天 → ~2400 天)
  - sp 梯度与 FDD 改为逐年窗口内计算, 算完即释放
  - 数值结果与 v2 完全一致:
      * np.gradient 只沿空间轴 (axis=-2/-1), 时间切片不影响
      * FDD 按日历日期回溯, padding 保证与全序列计算逐值相同
  - 覆盖提醒列表沿用 v2 的 7 个文件
"""

import os
import gc
import yaml
import numpy as np
import xarray as xr
import pandas as pd

# ================= 固定配置 =================
CSV_PATH = "data_nsidc.0p25/N_seaice_extent_daily_v4.0.csv"
ERA5_ROOT = r"\\10.152.252.137\yjzxdata\METDATA\data_era5.0p25\00Z"
CONFIG_PATH = "config.yaml"

SMOOTH_WINDOW = 7
D0_BEFORE_DAYS = 10
BURST_AFTER_DAYS = 10
FDD_WINDOWS = [2, 4, 6, 8, 10, 15, 20]   # 最长窗口决定 padding 天数
FDD_PAD_DAYS = max(FDD_WINDOWS)          # 20
T_FREEZE = -1.9

# 通道顺序 (必须与 v2 的 build_variable_dict 完全一致, 下游通道索引依赖此顺序)
VAR_ORDER = ['sic', 't2m', 'd2m', 'u10', 'v10',
             'sp_grad_x', 'sp_grad_y',
             'fdd2', 'fdd4', 'fdd6', 'fdd8', 'fdd10', 'fdd15', 'fdd20',
             'land_mask']


def get_channel_indices(cfg):
    """返回 (keep_channels, met_keep_channels, use_channels, met_use_names)。

    通道布局以 VAR_ORDER 为准 (15 通道: 0=sic, 1..6=气象, 7..13=FDD, 14=land_mask)。
    met 目标通道 (targets_met_seasonal.npy) 由 main() 取 yr_data[..., 1:-1, ...] 得到,
    即 VAR_ORDER 中除 sic 与 land_mask 外的 13 个气象通道 (下标 1..13)。

    cfg['data']['use_fdd']=False 时返回无 FDD 消融布局 (8 序列通道 / 6 met 通道),
    以便与第7次无-fdd 重训数据集配套; 默认 use_fdd=True = 全通道 (with-FDD) 主模型。
    """
    fdd_idx = [VAR_ORDER.index('fdd%d' % w) for w in FDD_WINDOWS]   # 7..13
    met_idx = [i for i, v in enumerate(VAR_ORDER)
               if v not in ('sic', 'land_mask')]                   # 1..13
    use_fdd = True
    try:
        use_fdd = bool(cfg['data'].get('use_fdd', True))
    except Exception:
        pass
    if use_fdd:
        keep = list(range(len(VAR_ORDER)))            # 0..14
        met_keep = list(range(len(met_idx)))           # 0..12
        use = list(met_idx)                           # 1..13
        names = [VAR_ORDER[i] for i in met_idx]
    else:
        keep_met = [i for i in met_idx if i not in fdd_idx]   # 1..6
        keep = [0] + keep_met + [len(VAR_ORDER) - 1]         # sic + 6 met + land_mask
        met_keep = list(range(len(keep_met)))                # 0..5
        use = list(keep_met)                                # 1..6
        names = [VAR_ORDER[i] for i in keep_met]
    return keep, met_keep, use, names


def load_csv_extent(csv_path, start_year, end_year):
    df = pd.read_csv(csv_path, skiprows=[1])
    df = df.iloc[:, :4]
    df.columns = ['Year', 'Month', 'Day', 'Extent']
    for col in ['Year', 'Month', 'Day', 'Extent']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Year', 'Extent'])
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(int))
    df = df.sort_values('date').reset_index(drop=True)
    df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()

    change_list = []
    for year, grp in df.groupby('Year'):
        grp = grp.sort_values('date').reset_index(drop=True)
        extent = grp['Extent'].values
        change = np.zeros(len(extent))
        if len(extent) > SMOOTH_WINDOW:
            change[SMOOTH_WINDOW:] = (extent[SMOOTH_WINDOW:] - extent[:-SMOOTH_WINDOW]) / SMOOTH_WINDOW
        change_list.append(change)
    df['change'] = np.concatenate(change_list)
    return df


def find_key_dates(df):
    results = []
    for year, grp in df.groupby('Year'):
        grp = grp.sort_values('date')
        change = grp['change'].values
        dates = grp['date'].values
        burst_idx = np.argmax(change)
        burst_date = pd.Timestamp(dates[burst_idx])
        d0_date = None
        for i in range(burst_idx - 1, -1, -1):
            if change[i] < 0:
                d0_idx = min(i + 1, burst_idx)
                d0_date = pd.Timestamp(dates[d0_idx])
                break
        if d0_date is None:
            d0_date = burst_date
        results.append({'year': year, 'burst_date': burst_date, 'd0_date': d0_date})
    return pd.DataFrame(results)


def get_yearly_windows(key_dates_df):
    windows = {}
    for _, row in key_dates_df.iterrows():
        start = row['d0_date'] - pd.Timedelta(days=D0_BEFORE_DAYS)
        end = row['burst_date'] + pd.Timedelta(days=BURST_AFTER_DAYS)
        windows[int(row['year'])] = (start, end)
        print(f"  {row['year']}: D0={row['d0_date'].date()}, 爆发={row['burst_date'].date()}, "
              f"窗口={start.date()} → {end.date()}")
    return windows


def load_sic(sic_path, lat_slice, lon_slice, sic_var_name):
    ds = xr.open_dataset(sic_path)
    ds = ds.sel(lat=lat_slice, lon=lon_slice)
    raw_sic = ds[sic_var_name].values
    dates = pd.to_datetime(ds['time'].values).tolist()
    print(f"🧊 SIC: {raw_sic.shape}, {dates[0]} ~ {dates[-1]}")
    return raw_sic, dates


def load_era5_for_dates(era5_root, start_year, end_year, need_dates, lat_slice, lon_slice, req_vars):
    """v3: 只加载 need_dates (datetime64[D] 升序数组) 对应日期的 ERA5 数据"""
    dates_np = np.asarray(need_dates, dtype='datetime64[D]')
    total_days = len(dates_np)
    # 空间尺寸
    H = W = None
    for yr in range(start_year, end_year + 1):
        nc_file = os.path.join(era5_root, str(yr), f"{yr}_surf_00.nc")
        if os.path.exists(nc_file):
            with xr.open_dataset(nc_file) as ds_tmp:
                ds_tmp = ds_tmp.rename({'latitude': 'lat', 'longitude': 'lon'})
                ds_tmp = ds_tmp.assign_coords(lon=(ds_tmp.lon % 360)).sortby('lon')
                ds_tmp = ds_tmp.sel(lat=lat_slice, lon=lon_slice)
                H, W = ds_tmp.sizes['lat'], ds_tmp.sizes['lon']
                break
    if H is None:
        raise RuntimeError("找不到任何 ERA5 文件, 请检查 ERA5_ROOT")

    data = {var: np.full((total_days, H, W), np.nan, dtype=np.float32) for var in req_vars}
    for yr in range(start_year, end_year + 1):
        nc_file = os.path.join(era5_root, str(yr), f"{yr}_surf_00.nc")
        if not os.path.exists(nc_file):
            print(f"⚠️ 缺失: {nc_file}")
            continue
        ds = xr.open_dataset(nc_file)
        if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
        ds = ds.sel(lat=lat_slice, lon=lon_slice)
        era_np = pd.to_datetime(ds['time'].values).normalize().values.astype('datetime64[D]')
        # 只取需要的日期 (两侧均升序)
        _, pos_idx, era_idx = np.intersect1d(dates_np, era_np, return_indices=True)
        if len(pos_idx) == 0:
            ds.close()
            continue
        sub = ds.isel(time=era_idx)
        for var in req_vars:
            vals = sub[var].values
            if vals.ndim == 4 and vals.shape[1] == 1:   # 兼容 expver 维度
                vals = vals[:, 0]
            data[var][pos_idx] = vals.squeeze()
        ds.close()
        print(f"  {yr}: 命中 {len(pos_idx)}/{total_days} 天")
    # 填充
    for var in req_vars:
        arr = data[var]
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            print(f"⚠️ {var} 缺失 {nan_mask.sum()}，填充")
            arr_2d = arr.reshape(total_days, -1)
            arr_2d = pd.DataFrame(arr_2d).fillna(method='ffill').fillna(method='bfill').values.reshape(total_days, H, W)
            data[var] = arr_2d.astype(np.float32)
    print("✅ ERA5 加载完成")
    return data, dates_np


def build_window_chunk(idx, eidx, era5_data, sic_clean, land_mask, era_dates, start_dt, end_dt):
    """对一个年窗口计算全部 15 通道 (含梯度/FDD), 返回 dict + 窗口日期"""
    n = len(idx)
    H, W = sic_clean.shape[1:]

    # ERA5 padding 段的连续性校验
    pad_start = start_dt - pd.Timedelta(days=FDD_PAD_DAYS)
    expected = np.arange(np.datetime64(pad_start.date(), 'D'),
                         np.datetime64(end_dt.date(), 'D') + np.timedelta64(1, 'D'),
                         dtype='datetime64[D]')
    if not np.array_equal(era_dates[eidx], expected):
        raise RuntimeError(f"ERA5 日期不连续/缺失 ({pad_start.date()} ~ {end_dt.date()}), 请检查 ERA5 文件")

    # t2m (含 padding) 与空间梯度 (只沿空间轴, 与全序列计算逐值一致)
    t2m_pad = era5_data['t2m'][eidx] - 273.15
    sp = era5_data['sp'][eidx]
    sp_grad_y = np.gradient(sp, axis=-2)
    sp_grad_x = np.gradient(sp, axis=-1)
    del sp
    # 对称 log 压缩，抑制极端梯度值，同时保留方向
    sp_grad_x = np.sign(sp_grad_x) * np.log1p(np.abs(sp_grad_x))
    sp_grad_y = np.sign(sp_grad_y) * np.log1p(np.abs(sp_grad_y))

    # FDD: 窗口第 j 天在 padding 序列中的行号 = FDD_PAD_DAYS + j
    w_off = FDD_PAD_DAYS
    chunk = {}
    for w in FDD_WINDOWS:
        arr = np.zeros((n, H, W), dtype=np.float32)
        for j in range(n):
            g = w_off + j
            s0 = max(0, g - w + 1)
            arr[j] = np.sum(np.maximum(0.0, T_FREEZE - t2m_pad[s0:g + 1]), axis=0)
        chunk[f'fdd{w}'] = arr

    # 通道顺序必须与 VAR_ORDER 一致
    sl = slice(w_off, w_off + n)
    chunk['sic'] = sic_clean[idx].astype(np.float32)
    chunk['t2m'] = t2m_pad[sl].astype(np.float32)
    chunk['d2m'] = era5_data['d2m'][eidx][sl].astype(np.float32)
    chunk['u10'] = era5_data['u10'][eidx][sl].astype(np.float32)
    chunk['v10'] = era5_data['v10'][eidx][sl].astype(np.float32)
    chunk['sp_grad_x'] = sp_grad_x[sl].astype(np.float32)
    chunk['sp_grad_y'] = sp_grad_y[sl].astype(np.float32)
    chunk['land_mask'] = np.broadcast_to(land_mask[None, ...], (n, H, W)).astype(np.float32)

    dates = list(pd.to_datetime(era_dates[eidx][w_off:w_off + n]))
    return chunk, dates


def normalize_all(vars_dict, land_mask, scaler_path):
    scaler_params = {}
    for name, data in vars_dict.items():
        if name == 'land_mask':
            continue
        if name == 'sic':
            mask_data = data.copy()
            mask_data[:, land_mask] = np.nan
            valid = mask_data[~np.isnan(mask_data)]
            min_v, max_v = (0., 1.) if len(valid) == 0 else (np.min(valid), np.max(valid))
            norm = (data - min_v) / (max_v - min_v) if max_v - min_v > 1e-6 else np.zeros_like(data)
            norm[:, land_mask] = 0.
            scaler_params[name] = {'min': float(min_v), 'max': float(max_v)}
        else:
            ocean = data[:, ~land_mask]
            mean_v = np.mean(ocean)
            std_v = np.std(ocean)
            if std_v < 1e-8: std_v = 1.0
            norm = (data - mean_v) / std_v
            scaler_params[name] = {'mean': float(mean_v), 'std': float(std_v)}
        vars_dict[name] = norm.astype(np.float32)
        gc.collect()
        print(f"  [{name}] 归一化完成")
    np.save(scaler_path, scaler_params)
    return vars_dict


def main():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    version = data_cfg['version']
    start_year = data_cfg['start_year']
    end_year = data_cfg['end_year']
    pole = data_cfg['pole']
    lat_slice = slice(data_cfg["lat_max"], data_cfg["lat_min"])
    lon_slice = slice(data_cfg["lon_min"], data_cfg["lon_max"])
    output_dir = './npy'
    seq_len = data_cfg["sequence_length"]
    pred_len = data_cfg["prediction_length"]
    os.makedirs(output_dir, exist_ok=True)

    sic_filename = f"data_raw/{pole}_{start_year}0101_{end_year}1231_0051_{version}_on_latlon_025deg.nc"
    sic_var = f"{version}_ICECON"

    # 备份提醒 (7 个文件, 与 v2 一致)
    print("⚠️ 以下文件将被覆盖，请确认已备份：")
    overwrite_files = [
        data_cfg['seq_file'], data_cfg['target_file'],
        data_cfg['scaler_file'], data_cfg['mask_file'],
        'targets_met_seasonal.npy',
        'sample_start_dates.npy',
        'sample_start_dates.txt',
    ]
    for f in overwrite_files:
        p = os.path.join(output_dir, f)
        if os.path.exists(p): print(f"  - {p}")
    input("按 Enter 继续...")

    # 1. 关键日期
    df_extent = load_csv_extent(CSV_PATH, start_year, end_year)
    key_dates = find_key_dates(df_extent)
    windows = get_yearly_windows(key_dates)

    # 2. 加载 SIC (全序列, 用于 land_mask 与窗口切片)
    raw_sic, dates_sic = load_sic(sic_filename, lat_slice, lon_slice, sic_var)
    valid = (raw_sic >= 0) & (raw_sic <= 1)
    land_mask = (~valid).all(axis=0)
    sic_clean = np.nan_to_num(raw_sic, nan=0.0).clip(0, 1)
    sic_clean[:, land_mask] = 0.0
    del raw_sic
    gc.collect()
    H, W = sic_clean.shape[1:]

    # 3. 需要的 ERA5 日期 = 各年窗口 ∪ 每个窗口前 20 天 (FDD padding)
    need_dates = np.zeros(0, dtype='datetime64[D]')
    for year in range(start_year, end_year + 1):
        if year not in windows: continue
        s, e = windows[year]
        s2 = np.datetime64(s.date(), 'D') - np.timedelta64(FDD_PAD_DAYS, 'D')
        e2 = np.datetime64(e.date(), 'D')
        need_dates = np.union1d(need_dates, np.arange(s2, e2 + 1, dtype='datetime64[D]'))
    print(f"📅 需要的 ERA5 日期数: {len(need_dates)} (含 FDD padding, v2 为 10958 天)")

    req_vars = ['t2m', 'd2m', 'u10', 'v10', 'sp']
    era5_data, era_dates = load_era5_for_dates(
        ERA5_ROOT, start_year, end_year, need_dates, lat_slice, lon_slice, req_vars)

    # 4. 预分配 combined (窗口总天数), 逐窗口填充, 避免拼接翻倍内存
    win_infos = []          # (offset, n)
    combined_dates = []
    n_total = 0
    for year in range(start_year, end_year + 1):
        if year not in windows: continue
        s, e = windows[year]
        mask = (pd.to_datetime(dates_sic) >= s) & (pd.to_datetime(dates_sic) <= e)
        n = int(mask.sum())
        if n > 0:
            win_infos.append((n_total, n))
            n_total += n
    print(f"窗口总天数: {n_total}")
    combined = {v: np.empty((n_total, H, W), dtype=np.float32) for v in VAR_ORDER}

    print("\n✂️ 逐窗口计算 (梯度/FDD 在窗口内完成, 算完即释放)...")
    win_counter = 0
    for year in range(start_year, end_year + 1):
        if year not in windows: continue
        start_dt, end_dt = windows[year]
        mask = (pd.to_datetime(dates_sic) >= start_dt) & (pd.to_datetime(dates_sic) <= end_dt)
        idx = np.where(mask)[0]
        if len(idx) == 0: continue
        # ERA5 行索引 (含 padding)
        pad_start = start_dt - pd.Timedelta(days=FDD_PAD_DAYS)
        emask = (era_dates >= np.datetime64(pad_start.date(), 'D')) & \
                (era_dates <= np.datetime64(end_dt.date(), 'D'))
        eidx = np.where(emask)[0]

        chunk, chunk_dates = build_window_chunk(
            idx, eidx, era5_data, sic_clean, land_mask, era_dates, start_dt, end_dt)
        chunk_offset, chunk_n = win_infos[win_counter]
        win_counter += 1
        for v in VAR_ORDER:
            combined[v][chunk_offset:chunk_offset + chunk_n] = chunk[v]
        del chunk
        combined_dates.extend(chunk_dates)
        print(f"  {year}: {chunk_n} 天 已写入")
        gc.collect()

    # 释放大数组
    del era5_data, sic_clean
    gc.collect()
    print("✅ 已释放 ERA5 与 SIC 原始数组")

    # 5. 归一化 (统计口径与 v2 相同: 全部窗口拼接后的海洋格点)
    scaler_path = os.path.join(output_dir, data_cfg['scaler_file'])
    combined = normalize_all(combined, land_mask, scaler_path)

    # 6. 生成样本 (逻辑与 v2 逐行等价, 仅按 win_infos 遍历)
    input_vars = list(combined.keys())   # VAR_ORDER 顺序
    n_met_channels = len([v for v in input_vars if v not in ('sic', 'land_mask')])

    print("\n🧩 生成样本...")
    total_samp = 0
    for offset, n in win_infos:
        if n >= seq_len + pred_len:
            total_samp += n - seq_len - pred_len + 1
    print(f"预计总样本数: {total_samp}")

    seq_path = os.path.join(output_dir, data_cfg['seq_file'])
    tgt_path = os.path.join(output_dir, data_cfg['target_file'])
    tgt_met_path = os.path.join(output_dir, 'targets_met_seasonal.npy')

    seq_shape = (total_samp, seq_len, len(input_vars), H, W)
    tgt_shape = (total_samp, pred_len, 1, H, W)
    tgt_met_shape = (total_samp, pred_len, n_met_channels, H, W)

    seqs = np.lib.format.open_memmap(seq_path, dtype='float16', mode='w+', shape=seq_shape)
    tgts = np.lib.format.open_memmap(tgt_path, dtype='float16', mode='w+', shape=tgt_shape)
    tgts_met = np.lib.format.open_memmap(tgt_met_path, dtype='float16', mode='w+', shape=tgt_met_shape)

    sample_idx = 0
    for offset, n in win_infos:
        if n < seq_len + pred_len:
            continue
        yr_data_dict = {var: combined[var][offset:offset + n] for var in input_vars}
        yr_data = np.stack([yr_data_dict[var] for var in input_vars], axis=1).astype('float16')
        del yr_data_dict
        n_samp = n - seq_len - pred_len + 1
        for i in range(n_samp):
            seqs[sample_idx] = yr_data[i:i + seq_len]
            tgts[sample_idx] = yr_data[i + seq_len:i + seq_len + pred_len, 0:1, :, :]
            tgts_met[sample_idx] = yr_data[i + seq_len:i + seq_len + pred_len, 1:-1, :, :]
            sample_idx += 1
        print(f"  窗口@{combined_dates[offset]}: {n_samp} 样本")

    # 7. 保存 mask 和日期文件
    np.save(os.path.join(output_dir, data_cfg['mask_file']), land_mask)
    date_strs = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in combined_dates]
    # 样本参考日期 = 每个样本序列最后一天 (与 v2 一致)
    sample_dates_list = []
    for offset, n in win_infos:
        if n < seq_len + pred_len:
            continue
        for i in range(n - seq_len - pred_len + 1):
            sample_dates_list.append(date_strs[offset + i + seq_len - 1])
    np.save(os.path.join(output_dir, 'sample_start_dates.npy'), np.array(sample_dates_list))
    with open(os.path.join(output_dir, 'sample_start_dates.txt'), 'w') as f:
        for d in sample_dates_list:
            f.write(d + '\n')

    print(f"✅ 完成！样本数: {total_samp}")
    print(f"   输入: {seq_shape}  目标SIC: {tgt_shape}  目标气象: {tgt_met_shape}")


if __name__ == "__main__":
    main()
