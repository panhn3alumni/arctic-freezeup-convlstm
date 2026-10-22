import torch
import torch.nn as nn
import torch.nn.init as init


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        h, w = image_size
        state = torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
        return (state, state)


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, input_tensor, hidden_state=None):
        # input: (B, T, C, H, W)
        b, t, c, h, w = input_tensor.size()
        device = input_tensor.device

        if hidden_state is None:
            hidden_state = [cell.init_hidden(b, (h, w), device) for cell in self.cell_list]

        layer_output = input_tensor
        last_states = []

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_seq = []
            for t_idx in range(t):
                h, c = self.cell_list[layer_idx](layer_output[:, t_idx, :, :, :], (h, c))
                output_seq.append(h)
            layer_output = torch.stack(output_seq, dim=1)  # (B, T, H_dim, H, W)
            last_states.append((h, c))

        return layer_output, last_states


class SeaIcePredictor(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, num_layers, pred_len, output_mode='regression'):
        super(SeaIcePredictor, self).__init__()
        self.pred_len = pred_len
        self.output_mode = output_mode   # 'regression' 或 'binary'

        # Encoder
        self.encoder = ConvLSTM(input_channels, hidden_dim, kernel_size, num_layers)

        # Decoder Head: 直接将最后时刻的隐藏状态映射到 (Pred_Len, 1, H, W)
        # 这里使用一个 3D 卷积或者 TimeDistributed 2D 卷积
        # 方案：取 Encoder 最后一个时间步的输出 (B, Hidden, H, W)，通过卷积生成 (B, Pred_Len, 1, H, W)
        self.regressor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, pred_len, kernel_size=1),  # 输出通道数 = 预测天数
            nn.Sigmoid()
        )

        # --- 在这里添加偏置初始化 ---
        # self.regressor[2] 是最后一层 Conv2d
        # 将偏置初始化为 0，这样 Sigmoid(0) = 0.5
        # 这能让模型在训练初期输出 0.5 左右的概率，而不是全黑 (0)

        # 偏置初始化使 sigmoid(bias) ≈ 0.15
        init.constant_(self.regressor[2].bias, -1.734)  # ln(0.15/0.85) ≈ -1.734
        # ---------------------------

    def forward(self, x, mask=None):
        # === 修改点 ===
        # 原代码保留所有时间步，非常消耗显存：
        # x: (B, T_in, C, H, W)
        encoded, _ = self.encoder(x)

        # 取最后一个时间步的特征: (B, Hidden, H, W)
        last_feature = encoded[:, -1, :, :, :]

        # # 改为从状态元组中获取最后一层的最后一个时刻的隐藏状态 h
        # # _ 是层状态列表，每个元素是 (h, c) 元组
        # _, last_states = self.encoder(x)
        # # 取最后一层 (索引 -1) 的 h 状态 (索引 0)
        # last_feature = last_states[-1][0]
        # =============

        # 回归预测: (B, Pred_Len, H, W)
        pred = self.regressor(last_feature)

        # 调整形状为 (B, Pred_Len, 1, H, W) 以匹配 Target
        pred = pred.unsqueeze(2)

        # if self.output_mode == 'regression':  # 二值化 loss 就可以跳过这一步了
        #     # # Sigmoid 约束输出在 0-1 (海冰浓度)
        #     # pred = torch.sigmoid(pred)
        #     pred = torch.clamp(pred, 0.0, 1.0)

        # if mask is not None:
        #     # mask: (B, 1, H, W) -> 扩展为 (B, T, 1, H, W)
        #     pred = pred * mask.unsqueeze(1)

        return pred