import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from datetime import datetime
from dataset import SeaIceDataset
from convlstm import SeaIcePredictor
from visualization import save_prediction_images, plot_loss_curve, plot_metric_histograms
from step2_prepare_data_new import get_channel_indices

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_binary_counts(pred, target, ocean_mask, threshold=0.15):
    """计算单个batch的TP, FP, FN（只统计海洋区域）"""
    pred_bin = (pred >= threshold).float() * ocean_mask
    target_bin = (target >= threshold).float() * ocean_mask
    TP = ((pred_bin == 1) & (target_bin == 1)).float().sum()
    FP = ((pred_bin == 1) & (target_bin == 0)).float().sum()
    FN = ((pred_bin == 0) & (target_bin == 1)).float().sum()
    return TP, FP, FN


def main():
    # scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda')

    # 1. 加载配置
    cfg = load_config()
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    # 2. 准备数据
    data_cfg = cfg['data']
    base_dir = './npy'

    keep_channels, met_keep_channels, use_channels, met_use_names = get_channel_indices(cfg)

    full_dataset = SeaIceDataset(
        seq_path=os.path.join(base_dir, data_cfg['seq_file']),
        tgt_path=os.path.join(base_dir, data_cfg['target_file']),
        mask_path=os.path.join(base_dir, data_cfg['mask_file']),
        met_path=os.path.join(base_dir, data_cfg.get('met_target_file', 'targets_met_seasonal.npy')),
        keep_channels=keep_channels,
        met_keep_channels=met_keep_channels
    )

    total_samples = len(full_dataset)

    # 2. 按时间顺序划分 (80% 训练, 10% 验证, 10% 测试)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_samples))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)  # ← 新增

    # 3. 创建 DataLoader（训练集可 shuffle，验证集不要 shuffle）
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        pin_memory=True  # 加速 GPU 数据传输
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=False,         # 验证时保持时间顺序
        num_workers=data_cfg['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=True
    )

    # 自动推断输入通道数
    sample_batch = next(iter(train_loader))
    input_channels = sample_batch['sequence'].shape[2]
    print(f"📊 检测到输入通道数: {input_channels}")

    print("🔍 检查数据范围:")
    print("Input seq min/max:", sample_batch['sequence'].min().item(), sample_batch['sequence'].max().item())

    # --- 新增：按通道和时间步统计 ---
    x_sample = sample_batch['sequence']  # (B, T, C, H, W)
    mask_sample = sample_batch['mask']  # (B, 1, H, W)  陆地=1，海洋=0
    ocean_mask = (mask_sample == 0).squeeze(1)  # (B, H, W)，海洋为True
    land_mask_bool = (mask_sample == 1).squeeze(1)  # (B, H, W)，陆地为True

    for c in range(x_sample.shape[2]):
        # 海洋统计
        ocean_vals = x_sample[:, :, c, :, :][ocean_mask.unsqueeze(1).expand(-1, x_sample.shape[1], -1, -1)]
        # 陆地统计
        land_vals = x_sample[:, :, c, :, :][land_mask_bool.unsqueeze(1).expand(-1, x_sample.shape[1], -1, -1)]

        if ocean_vals.numel() > 0:
            o_min, o_max = ocean_vals.min().item(), ocean_vals.max().item()
        else:
            o_min, o_max = float('nan'), float('nan')

        if land_vals.numel() > 0:
            l_min, l_max = land_vals.min().item(), land_vals.max().item()
        else:
            l_min, l_max = float('nan'), float('nan')

        print(f"  通道 {c}: 海洋 min={o_min:.4f}, max={o_max:.4f} | 陆地 min={l_min:.4f}, max={l_max:.4f}")

    print("Target min/max:", sample_batch['target'].min().item(), sample_batch['target'].max().item())
    if 'mask' in sample_batch:
        print("Mask unique values:", torch.unique(sample_batch['mask']))

    # 更新配置中的通道数 (以防 config 写错)
    cfg['model']['input_channels'] = input_channels

    # 3. 初始化模型
    model = SeaIcePredictor(
        input_channels=cfg['model']['input_channels'],
        hidden_dim=cfg['model']['hidden_dim'],
        kernel_size=cfg['model']['kernel_size'],
        num_layers=cfg['model']['num_layers'],
        pred_len=1
        # pred_len=cfg['data']['prediction_length']
    ).to(device)

    # model = SwinPredictor(
    #     input_channels=cfg['model']['input_channels'],  # 15
    #     T_in=cfg['data']['sequence_length'],  # 10
    #     hidden_dim=64,
    #     spatial_downsample=4  # 如果显存不够，可改为 8 或 16
    # ).to(device)

    # model = SwinConvLSTMPredictor(
    #     input_channels=cfg['model']['input_channels'],  # 15
    #     T_in=cfg['data']['sequence_length'],  # 10
    #     hidden_dim=64,
    #     lstm_hidden_dim=64,  # 可调
    #     spatial_downsample=2,  # 可调
    # ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['train']['learning_rate']),
                                 weight_decay=float(cfg['train']['weight_decay']))

    # --- 调度器在loss不下降时自动降低学习率 ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 创建检查点目录
    os.makedirs(cfg['train']['checkpoint_dir'], exist_ok=True)

    # 4. 训练循环
    best_loss = float('inf')
    epochs = cfg['train']['epochs']

    print(f"⏳ 开始训练 {epochs} 个 epoch...")

    train_losses = []
    val_losses = []

    patience = cfg['train']['patience']
    trigger_times = 0

    # save_dir = cfg['train']['checkpoint_dir']
    # 这样才能断开了RDP也不会训练终止，只能每次训练完再复制一份到本地了
    save_dir = os.path.join(cfg['train']['checkpoint_dir'], cfg['infer']['output_path'])
    # save_dir = os.path.join("/home/haonanp/thinclient_drives/./checkpoints", cfg['infer']['output_path'])
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_mae = 0.0  # 新增：用于累加 MAE

        train_TP = 0.0
        train_FP = 0.0
        train_FN = 0.0

        # # 用于保存最后一批数据的预测结果（用来画图）
        # last_batch_pred = None
        # last_batch_target = None

        for batch_idx, batch in enumerate(train_loader):
            x = batch['sequence'].to(device)  # (B, T_in, C, H, W)
            y = batch['target'].to(device)  # (B, T_out, 1, H, W)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            optimizer.zero_grad()

            # ===== 启用 AMP 的核心修改 =====
            with torch.amp.autocast(device_type='cuda'):  # ← 自动将计算转为 FP16
                # ===== 自回归滚动10步，每步独立反向传播 =====
                pred_list = []
                current_input = x
                loss = 0.0  # 仅用于打印，记录总损失

                y_met = batch.get('target_met', None)
                if y_met is not None:
                    y_met = y_met.to(device)  # (B, 10, n_met, H, W)

                for step in range(10):
                    # 1. 单步预测
                    pred_step = model(current_input, mask=mask)  # (B, 1, 1, H, W)
                    pred_list.append(pred_step.detach())  # 保存预测（断开梯度）

                    # 2. 当前步的标签
                    y_step = y[:, step:step + 1, :, :, :]
                    if mask is not None:
                        ocean_mask = (1 - mask).unsqueeze(1)
                        step_loss = (criterion(pred_step, y_step) * ocean_mask).sum() / ocean_mask.sum()
                    else:
                        step_loss = criterion(pred_step, y_step).mean()

                    # 3. 立即反向传播并更新参数（截断 BPTT）
                    optimizer.zero_grad()
                    scaler.scale(step_loss).backward()  # 单步反向传播
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    # 4. 累加损失（仅用于打印）
                    loss += step_loss.detach().item()

                    # 5. 构造下一步输入（必须 detach，切断梯度）
                    if step < 9:
                        # 预测的 SIC（已 detach）
                        next_sic = pred_step.detach().squeeze(2)  # (B, 1, H, W)
                        # 对应时次的真实气象
                        if y_met is not None:
                            next_met = y_met[:, step:step + 1, :, :, :]  # (B, 1, n_met, H, W)
                        else:
                            # 回退方案（如果没有 target_met）
                            next_met = current_input[:, -1, 1:-1, :, :].unsqueeze(1)
                        # 陆地掩膜直接复制当前输入的最后一天
                        next_landmask = current_input[:, -1:, -1:, :, :]  # (B, 1, 1, H, W)
                        # 拼接新时间步
                        next_time_step = torch.cat([next_sic.unsqueeze(2), next_met, next_landmask],
                                                   dim=2)  # (B, 1, C, H, W)
                        # 滑动窗口
                        current_input = torch.cat([current_input[:, 1:, :, :, :], next_time_step], dim=1)

                # 最终损失取平均（用于打印和记录）
                loss = loss / 10.0
                pred = torch.cat(pred_list, dim=1)  # (B, 10, 1, H, W)

            total_loss += loss

            # --- 新增：计算并累加当前 batch 的 MAE ---
            with torch.no_grad():  # 不需要梯度，节省计算
                # mae_batch = torch.mean(torch.abs(pred - y)).item()
                if mask is not None:
                    ocean_mask = (1 - mask).unsqueeze(1)
                    mae_batch = (torch.abs(pred - y) * ocean_mask).sum() / ocean_mask.sum()
                else:
                    mae_batch = torch.mean(torch.abs(pred - y))
                total_mae += mae_batch

                # 新增：累加二值指标
                if mask is not None:
                    ocean_mask_bin = (1 - mask).unsqueeze(1)  # (B,1,1,H,W)，海洋=1
                else:
                    ocean_mask_bin = torch.ones_like(pred)
                tp, fp, fn = compute_binary_counts(pred, y, ocean_mask_bin)
                train_TP += tp.item()
                train_FP += fp.item()
                train_FN += fn.item()
            # -----------------------------------------

            # 为了防止刷屏太厉害，这里只打印每 50 个 batch 和 最后一个 batch
            if batch_idx % 20 == 0 or batch_idx == len(train_loader) - 1:
                print(f"🔍 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                      f"[Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Pred: {pred.mean().item():.4f} | "    # 预测概率 (0~1)
                      f"True: {y.mean().item():.4f} | "            # 真实概率 (0~1)
                      f"Loss: {loss:.4f}")                  # 误差 (越小越好)

            # # 记录最后一个 batch 用于画图
            # if batch_idx == len(train_loader) - 1:
            #     last_batch_pred = pred.detach()
            #     last_batch_target = y.detach()

        train_avg_loss = total_loss / len(train_loader)
        train_avg_mae = total_mae / len(train_loader)  # 新增

        # --- 修改点 3: 每个 epoch 都打印，让你实时看到进度 ---
        # 并且直接读取 config 里的固定学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 计算训练二值指标
        train_TP = max(train_TP, 1e-8)
        train_FP = max(train_FP, 1e-8)
        train_FN = max(train_FN, 1e-8)
        AEE = train_FP - train_FN
        ME = 2 * min(train_FP, train_FN)
        IIEE = abs(AEE) + ME
        precision = train_TP / (train_TP + train_FP)
        recall = train_TP / (train_TP + train_FN)
        TS = train_TP / (train_TP + train_FP + train_FN)

        print(f"Epoch [{epoch + 1}/{epochs}]\n"
              f"✅ Train | Loss={train_avg_loss:.6f} MAE={train_avg_mae:.6f}, LR: {current_lr:.6f}\n"
              f"AEE={AEE:.1f} ME={ME:.1f} IIEE={IIEE:.1f} "
              f"Prec={precision:.4f} Rec={recall:.4f} TS={TS:.4f}")

        # # --- 修改点 2: 每个 Epoch 保存一次图片 ---
        # # 注意：这会增加一点训练时间，但能直观看到效果
        # if (epoch + 1) % 5 == 0 or epoch == 0: # 每5轮存一次，或者第一轮存一次
        #      save_prediction_images(epoch, last_batch_pred, last_batch_target)

        # ===== 新增：验证阶段 =====
        model.eval()
        val_total_loss = 0.0
        val_total_mae = 0.0
        val_sample_pred = None      # ← 必须先定义！
        val_sample_target = None    # ← 必须先定义！

        sample_metrics = {'TS': [], 'Prec': [], 'Rec': [], 'AEE': [], 'ME': [], 'IIEE': []}

        val_TP = 0.0  # 新增
        val_FP = 0.0  # 新增
        val_FN = 0.0  # 新增

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x = batch['sequence'].to(device)
                y = batch['target'].to(device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(device)

                with torch.amp.autocast(device_type='cuda'):
                    # pred = model(x, mask=mask)
                    # # loss = criterion(pred, y)
                    #
                    # if mask is not None:
                    #     ocean_mask = (1 - mask).unsqueeze(1)  # 现在 mask: 1=陆地, 0=海洋
                    #     loss = (criterion(pred, y) * ocean_mask).sum() / ocean_mask.sum()
                    #     mae = (torch.abs(pred - y) * ocean_mask).sum() / ocean_mask.sum()
                    # else:
                    #     loss = criterion(pred, y)
                    #     mae = torch.mean(torch.abs(pred - y))

                    # ===== 验证集自回归滚动10步 =====
                    pred_list_val = []
                    current_input_val = x

                    y_met = batch.get('target_met', None)
                    if y_met is not None:
                        y_met = y_met.to(device)

                    for step in range(10):
                        pred_step_val = model(current_input_val, mask=mask)  # (B, 1, 1, H, W)
                        pred_list_val.append(pred_step_val)

                        if step < 9:
                            next_sic_val = pred_step_val.squeeze(2)
                            # next_met_val = current_input_val[:, -1, 1:, :, :].unsqueeze(1)
                            # next_time_step_val = torch.cat([next_sic_val.unsqueeze(2), next_met_val], dim=2)
                            if y_met is not None:
                                next_met_val = y_met[:, step:step + 1, :, :, :]
                            else:
                                next_met_val = current_input_val[:, -1, 1:-1, :, :].unsqueeze(1)
                            next_landmask_val = current_input_val[:, -1:, -1:, :, :]
                            next_time_step_val = torch.cat([next_sic_val.unsqueeze(2), next_met_val, next_landmask_val],
                                                           dim=2)
                            current_input_val = torch.cat([current_input_val[:, 1:, :, :, :], next_time_step_val],
                                                          dim=1)

                    pred = torch.cat(pred_list_val, dim=1)  # (B, 10, 1, H, W)

                    # 计算整体损失（与训练一致，也可用累积单步loss，但整体计算等价）
                    if mask is not None:
                        ocean_mask_val = (1 - mask).unsqueeze(1)
                        loss = (criterion(pred, y) * ocean_mask_val).sum() / ocean_mask_val.sum()
                        mae = (torch.abs(pred - y) * ocean_mask_val).sum() / ocean_mask_val.sum()
                    else:
                        loss = criterion(pred, y).mean()
                        mae = torch.mean(torch.abs(pred - y))

                # mae = torch.mean(torch.abs(pred - y))

                val_total_loss += loss.item()
                val_total_mae += mae.item()

                # 新增：验证集二值指标累加
                if mask is not None:
                    ocean_mask_bin = (1 - mask).unsqueeze(1)
                else:
                    ocean_mask_bin = torch.ones_like(pred)
                tp, fp, fn = compute_binary_counts(pred, y, ocean_mask_bin)
                val_TP += tp.item()
                val_FP += fp.item()
                val_FN += fn.item()

                # 计算逐样本的 TP/FP/FN
                # 得到二值化的预测和真值
                pred_bin = (pred >= 0.15).float() * ocean_mask_bin
                target_bin = (y >= 0.15).float() * ocean_mask_bin

                # 对 T, C, H, W 维度求和，保留 batch 维度，得到每个样本的 TP/FP/FN
                dims_to_sum = tuple(range(1, pred_bin.dim()))
                tp_s = ((pred_bin == 1) & (target_bin == 1)).float().sum(dim=dims_to_sum).cpu().numpy()
                fp_s = ((pred_bin == 1) & (target_bin == 0)).float().sum(dim=dims_to_sum).cpu().numpy()
                fn_s = ((pred_bin == 0) & (target_bin == 1)).float().sum(dim=dims_to_sum).cpu().numpy()

                # 过滤掉全背景样本，防止计算指标时除以零
                eps = 1e-8
                valid = (tp_s + fp_s + fn_s) > 0
                if valid.any():
                    tp_v, fp_v, fn_v = tp_s[valid], fp_s[valid], fn_s[valid]
                    sample_metrics['TS'].extend((tp_v / (tp_v + fp_v + fn_v + eps)).tolist())
                    sample_metrics['Prec'].extend((tp_v / (tp_v + fp_v + eps)).tolist())
                    sample_metrics['Rec'].extend((tp_v / (tp_v + fn_v + eps)).tolist())
                    sample_metrics['AEE'].extend((fp_v - fn_v).tolist())
                    sample_metrics['ME'].extend((2.0 * np.minimum(fp_v, fn_v)).tolist())
                    sample_metrics['IIEE'].extend((fp_v + fn_v).tolist())

                # 只记录第一个 batch 用于画图（避免重复覆盖）
                if val_sample_pred is None:
                    val_sample_pred = pred.detach()
                    val_sample_target = y.detach()

        val_avg_loss = val_total_loss / len(val_loader)
        val_avg_mae = val_total_mae / len(val_loader)

        # 计算验证集二值指标
        val_TP = max(val_TP, 1e-8)
        val_FP = max(val_FP, 1e-8)
        val_FN = max(val_FN, 1e-8)
        v_AEE = val_FP - val_FN
        v_ME = 2 * min(val_FP, val_FN)
        v_IIEE = abs(v_AEE) + v_ME
        v_prec = val_TP / (val_TP + val_FP)
        v_rec = val_TP / (val_TP + val_FN)
        v_TS = val_TP / (val_TP + val_FP + val_FN)

        print(f"✅ Val | Loss={val_avg_loss:.6f} MAE={val_avg_mae:.6f}\n"
              f"AEE={v_AEE:.1f} ME={v_ME:.1f} IIEE={v_IIEE:.1f} "
              f"Prec={v_prec:.4f} Rec={v_rec:.4f} TS={v_TS:.4f}")

        train_losses.append(train_avg_loss)
        val_losses.append(val_avg_loss)

        scheduler.step(val_avg_loss)

        # ✅ 改为使用验证集样本绘图
        # if (epoch + 1) % 5 == 0 or epoch == 0:
        if val_sample_pred is not None:
            save_prediction_images(epoch, val_sample_pred, val_sample_target, land_mask=mask)
            plot_metric_histograms(sample_metrics, epoch)
        else:
            print("⚠️ 警告：未获取到验证样本，跳过绘图")
        # =========================

        # 保存最佳模型
        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            trigger_times = 0
            save_path = os.path.join(save_dir, 'best_model.pth')
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_avg_loss,
                'config': cfg
            }, save_path)
            print(f"💾 保存最佳模型到 {save_path} (Loss: {best_loss:.6f})")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("🚨 Early stopping!")
                break

    plot_loss_curve(train_losses, val_losses, save_dir)

    # ===== 测试集评估 =====
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    checkpoint = torch.load(best_model_path)
    state_dict = checkpoint['model_state_dict']

    # ✅ 核心修复：智能适配 DataParallel
    # 检查当前 model 是否是 DataParallel
    is_dp = isinstance(model, nn.DataParallel)

    # 获取 state_dict 中的第一个 key 来判断保存时是否带了 module. 前缀
    saved_keys_have_module = any(k.startswith('module.') for k in state_dict.keys())

    if is_dp and not saved_keys_have_module:
        # 情况1：当前模型是DP，但保存的权重没有module.前缀 -> 加上前缀
        print("🔧 检测到DataParallel模型加载无前缀权重，自动添加 'module.' 前缀...")
        new_state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not is_dp and saved_keys_have_module:
        # 情况2：当前模型不是DP，但保存的权重有module.前缀 -> 去掉前缀
        print("🔧 检测到单卡模型加载带前缀权重，自动去除 'module.' 前缀...")
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        # 情况3：两者一致，直接使用
        new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    model.eval()

    test_total_loss = 0.0
    test_total_mae = 0.0
    test_TP, test_FP, test_FN = 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in test_loader:
            x = batch['sequence'].to(device)
            y = batch['target'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            # 自回归滚动10步
            pred_list = []
            current_input = x
            y_met = batch.get('target_met', None)
            if y_met is not None:
                y_met = y_met.to(device)

            for step in range(10):
                pred_step = model(current_input, mask=mask)
                pred_list.append(pred_step)
                if step < 9:
                    next_sic = pred_step.squeeze(2)
                    if y_met is not None:
                        next_met = y_met[:, step:step + 1, :, :, :]
                    else:
                        next_met = current_input[:, -1, 1:-1, :, :].unsqueeze(1)
                    next_landmask = current_input[:, -1:, -1:, :, :]
                    next_step = torch.cat([next_sic.unsqueeze(2), next_met, next_landmask], dim=2)
                    current_input = torch.cat([current_input[:, 1:, :, :, :], next_step], dim=1)

            pred = torch.cat(pred_list, dim=1)

            # 计算 loss、MAE、二值指标（与验证阶段完全相同）
            if mask is not None:
                ocean_mask = (1 - mask).unsqueeze(1)
                loss = (criterion(pred, y) * ocean_mask).sum() / ocean_mask.sum()
                mae = (torch.abs(pred - y) * ocean_mask).sum() / ocean_mask.sum()
                ocean_mask_bin = ocean_mask
            else:
                loss = criterion(pred, y).mean()
                mae = torch.mean(torch.abs(pred - y))
                ocean_mask_bin = torch.ones_like(pred)

            test_total_loss += loss.item()
            test_total_mae += mae.item()

            tp, fp, fn = compute_binary_counts(pred, y, ocean_mask_bin)
            test_TP += tp.item()
            test_FP += fp.item()
            test_FN += fn.item()

    test_avg_loss = test_total_loss / len(test_loader)
    test_avg_mae = test_total_mae / len(test_loader)

    test_TP = max(test_TP, 1e-8)
    test_FP = max(test_FP, 1e-8)
    test_FN = max(test_FN, 1e-8)
    test_AEE = test_FP - test_FN
    test_ME = 2 * min(test_FP, test_FN)
    test_IIEE = abs(test_AEE) + test_ME
    test_prec = test_TP / (test_TP + test_FP)
    test_rec = test_TP / (test_TP + test_FN)
    test_TS = test_TP / (test_TP + test_FP + test_FN)

    print(f"\n🧪 测试集结果（基于最佳模型）")
    print(f"Loss={test_avg_loss:.6f} MAE={test_avg_mae:.6f}")
    print(f"AEE={test_AEE:.1f} ME={test_ME:.1f} IIEE={test_IIEE:.1f} "
          f"Prec={test_prec:.4f} Rec={test_rec:.4f} TS={test_TS:.4f}")

    print("✅ 训练完成！")


if __name__ == "__main__":
    main()
