import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
import logging
import glob
import re
import joblib
import random
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 基础工具函数与类定义
# ==========================================

def calculate_gating_indicators(df, window=30):
    dt_series = pd.to_datetime(df['hms'], errors='coerce')
    time_objs = dt_series.dt.time
    seconds = np.array([t.hour * 3600 + t.minute * 60 + t.second for t in time_objs])
    decay_minutes = np.full(len(df), 9999.0)
    impulse_minutes = np.full(len(df), 9999.0)
    mask_night_1 = (seconds >= 75600)
    # 夜盘
    if np.any(mask_night_1):
        dt = seconds[mask_night_1] - 75600
        decay_minutes[mask_night_1] = dt / 60.0
        impulse_minutes[mask_night_1] = dt / 60.0
    mask_night_2 = (seconds < 28800) # 小于 08:00
    if np.any(mask_night_2):
        # 距离 21:00 的秒数 = 当前秒 + (24小时 - 21小时)
        dt = seconds[mask_night_2] + (24 * 3600 - 75600)
        decay_minutes[mask_night_2] = dt / 60.0
        impulse_minutes[mask_night_2] = dt / 60.0
    # 日盘
    mask_am = (seconds >= 32400) & (seconds < 41400) # 09:00 - 11:30
    if np.any(mask_am):
        curr_secs = seconds[mask_am]
        decay_minutes[mask_am] = (curr_secs - 32400) / 60.0
        is_after_break = (curr_secs >= 37800)
        imp_mins = (curr_secs - 32400) / 60.0
        imp_mins[is_after_break] = (curr_secs[is_after_break] - 37800) / 60.0
        impulse_minutes[mask_am] = imp_mins

    mask_pm = (seconds >= 48600) & (seconds <= 15*3600)
    if np.any(mask_pm):
        dt = seconds[mask_pm] - 48600
        decay_minutes[mask_pm] = dt / 60.0
        impulse_minutes[mask_pm] = dt / 60.0
        
    sigma = 10.0
    df['gate_session_decay'] = 1.0 / np.sqrt(decay_minutes + 1.0)
    df['gate_open_impulse'] = np.exp(- (impulse_minutes ** 2) / (2 * sigma ** 2))
    df['mid_price'] = (df['ap1'] + df['bp1']) / 2
    df.loc[df['bp1']==0,'mid_price'] = df.loc[df['bp1']==0,'ap1']
    df.loc[df['ap1']==0,'mid_price'] = df.loc[df['ap1']==0,'bp1']
    df['mid_price'] = df['mid_price'].replace(0.0,np.nan)
    df['delta_vol'] = df['volume'].diff().fillna(0)
    df.loc[0,'delta_vol'] = df.loc[0,'volume']
    df['delta_vol_5'] = df['delta_vol'].rolling(10).sum()
    df['delta_oi'] = df['openinterest'].diff().fillna(0)
    df['spread'] = df['ap1'] - df['bp1']
    df.loc[df['bp1']==0,'spread'] = np.nan
    df.loc[df['ap1']==0,'spread'] = np.nan
    df['sum_bv5'] = df[['bv1', 'bv2', 'bv3', 'bv4', 'bv5']].sum(axis=1)
    df['sum_av5'] = df[['av1', 'av2', 'av3', 'av4', 'av5']].sum(axis=1)
    df['depth'] = np.log1p(df['sum_bv5'] + df['sum_av5'])
    df['log_ret_5'] = np.log(df['mid_price'] / df['mid_price'].shift(10)).fillna(0)
    df['rv_5'] = df['log_ret_5'].rolling(window=window).std()
    rolling_max = df['mid_price'].rolling(window=window).max()
    rolling_min = df['mid_price'].rolling(window=window).min()
    df['range'] = (rolling_max - rolling_min) / df['mid_price']
    df['l1_imbalance'] = (df['bv1'] - df['av1']) / (df['bv1'] + df['av1'] + 1e-6)
    df['l5_imbalance'] = (df['sum_bv5'] - df['sum_av5']) / (df['sum_bv5'] + df['sum_av5'] + 1e-6)
    total_order_cnt = df['ordercnt_bid1'] + df['ordercnt_ask1'] + 1e-6
    df['order_imbalance'] = (df['ordercnt_bid1'] - df['ordercnt_ask1']) / total_order_cnt
    df['oi_change'] = df['delta_oi'].rolling(10).sum()
    df['trade_intensity'] = np.log1p(df['delta_vol_5'])
    df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
    up = df['log_ret'].clip(lower=0)
    down = -1 * df['log_ret'].clip(upper=0)
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()
    rsi = 100 * ma_up / (ma_up + ma_down + 1e-9)
    df['rsi'] = rsi / 100.0
    df['dist_avg'] = (df['last'] - df['avg']) / df['avg']
    df['trend'] = (df['mid_price'] - df['mid_price'].shift(window)) / df['mid_price'].shift(window)
    gate_cols = [
        'spread', 'depth', 
        'rv_5', 'range',
        'l1_imbalance', 'l5_imbalance', 'order_imbalance',
        'oi_change', 'trade_intensity',
        'rsi', 'dist_avg', 'trend','gate_session_decay','gate_open_impulse'
    ]

    df_gate = df[['timestamp']+gate_cols].fillna(0).copy()
    return df_gate

def seed_everything(seed: int = 42):
    """设置全局随机种子，保证结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")

class SelectedIndicesDataset(torch.utils.data.Dataset):
    def __init__(self, features, indicators, labels, sites, target_code, seq_len, dilation=1):
        self.features = features
        self.indicators = indicators
        self.labels = labels
        self.seq_len = seq_len
        self.dilation = dilation
        self.span_offset = (seq_len - 1) * dilation
        
        valid_mask = (sites == target_code)
        all_indices = np.arange(len(sites))
        self.valid_anchors = all_indices[valid_mask & (all_indices >= self.span_offset)]
        
    def __len__(self):
        return len(self.valid_anchors)
    
    def __getitem__(self, idx):
        row_end = self.valid_anchors[idx]
        x_feat = self.features[row_end - self.span_offset : row_end + 1 : self.dilation]
        x_ind = self.indicators[row_end]
        y = self.labels[row_end]
        return x_feat, x_ind, y

def create_timestamp_fast(df, date_col='ExchActionDay', 
                          time_col='ExchUpdateTime', 
                          millisec_col='ExchUpdateMillisec'):
    df['date_part'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d')
    def time_to_seconds(time_str):
        if isinstance(time_str, str):
            parts = time_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:  # HH:MM
                h, m = map(int, parts)
                return h * 3600 + m * 60
        return 0
    df['time_seconds'] = df[time_col].apply(time_to_seconds)
    df['millisec_seconds'] = df[millisec_col] / 1000.0
    df['total_seconds'] = df['time_seconds'] + df['millisec_seconds']
    df['timestamp'] = df['date_part'] + pd.to_timedelta(df['total_seconds'], unit='s')
    temp_cols = ['date_part', 'time_seconds', 'millisec_seconds', 'total_seconds']
    df = df.drop(columns=temp_cols)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return

# ==========================================
# 2. 模型定义 (保持不变)
# ==========================================
@torch.jit.script
def slstm_scan_jit(i_gate: torch.Tensor, f_gate: torch.Tensor, z_gate: torch.Tensor, o_gate: torch.Tensor, init_c: torch.Tensor, init_n: torch.Tensor, init_m: torch.Tensor):
    B, T, H, D = i_gate.shape
    h_seq = []
    c_prev = init_c
    n_prev = init_n
    m_prev = init_m
    for t in range(T):
        i_t = i_gate[:, t]
        f_t = f_gate[:, t]
        z_t = z_gate[:, t]
        o_t = o_gate[:, t]
        m_t = torch.max(f_t + m_prev, i_t)
        i_prime = torch.exp(i_t - m_t)
        f_prime = torch.exp(f_t + m_prev - m_t)
        c_t = f_prime * c_prev + i_prime * z_t
        n_t = f_prime * n_prev + i_prime
        h_t = torch.sigmoid(o_t) * (c_t / (n_t + 1e-6))
        h_seq.append(h_t)
        c_prev = c_t
        n_prev = n_t
        m_prev = m_t
    return torch.stack(h_seq, dim=1)
class SoftmaxScaleGating(nn.Module):
    def __init__(self, context_dim, feature_dim, temperature=1.0, hidden_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, context):
        logits = self.gate_net(context)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        weights = probs * self.feature_dim
        return weights

class sLSTMBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, groups=hidden_dim, padding=3)
        self.conv_act = nn.SiLU()
        self.wx = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.norm = nn.GroupNorm(num_heads, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv(x_conv)[:, :, :T]
        x_conv = self.conv_act(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        gates = self.wx(x_conv)
        gates = gates.view(B, T, 4, self.num_heads, self.head_dim)
        i_gate, f_gate, z_gate, o_gate = gates.unbind(dim=2)
        z_gate = torch.tanh(z_gate)
        
        init_c = torch.zeros(B, self.num_heads, self.head_dim, device=x.device)
        init_n = torch.ones(B, self.num_heads, self.head_dim, device=x.device)
        init_m = torch.zeros(B, self.num_heads, self.head_dim, device=x.device)
        h_out = slstm_scan_jit(i_gate, f_gate, z_gate, o_gate, init_c, init_n, init_m)
        
        h_out = h_out.reshape(B, T, C)
        h_out = h_out.permute(0, 2, 1)
        h_out = self.norm(h_out)
        h_out = h_out.permute(0, 2, 1)
        out = self.out_proj(h_out)
        return self.dropout(out) + x

class ContextAwareAttentionGating(nn.Module):
    def __init__(self, context_dim, feature_dim, internal_dim=64, dropout=0.1):
        """
        基于注意力的特征门控网络 (Lightweight Attention Gating)
        
        Args:
            context_dim: 市场指标维度 (Input)
            feature_dim: 原始特征维度 (Output gates count)
            internal_dim: 注意力计算的内部维度 (d_model)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = internal_dim ** -0.5 # 缩放因子
        
        # 1. Query 生成器: 将市场 Context 映射为 Query 向量
        self.q_proj = nn.Sequential(
            nn.Linear(context_dim, internal_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Key 生成器: 为每个特征学习一个固定的"身份向量"
        # 这是一个可学习参数矩阵: (Feature_Dim, Internal_Dim)
        # 它代表了模型对"第i个特征是什么"的理解
        self.feature_embeddings = nn.Parameter(torch.randn(feature_dim, internal_dim))
        
        # 初始化参数以加速收敛
        nn.init.xavier_uniform_(self.feature_embeddings)

    def forward(self, context):
        """
        Args:
            context: (Batch, Context_Dim)
        Returns:
            weights: (Batch, Feature_Dim) - 每个特征的权重系数
        """
        batch_size = context.size(0)
        
        # --- Step A: 生成 Query ---
        # (Batch, 1, Internal_Dim)
        Q = self.q_proj(context).unsqueeze(1)
        K = self.feature_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        weights = 2.0 * torch.sigmoid(attn_logits).squeeze(1)
        return weights

# --- 主预测模型 ---

class ContextAwareSLSTMPredictor(nn.Module):
    def __init__(self, 
                 feature_dim,    # 409
                 context_dim,    # 20
                 hidden_dim,     # 256
                 num_layers=2, 
                 num_heads=4,
                 dropout=0.1):
        super().__init__()
        
        # 1. 使用新的 Attention Gating 替代简单的 MLP
        self.attention_gating = ContextAwareAttentionGating(
            context_dim=context_dim,
            feature_dim=feature_dim,
            internal_dim=64 # 这里的维度不需要很大，足够编码特征属性即可
        )
        
        # 2. 特征投影 (Embedding)
        self.embedding = nn.Linear(feature_dim, hidden_dim)
        
        # 3. sLSTM 骨干网络 (保持不变)
        self.layers = nn.ModuleList([
            sLSTMBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_feat, x_context):
        """
        x_feat:    (Batch, Time, Feature_Dim)
        x_context: (Batch, Context_Dim)
        """
        
        # --- 1. 计算特征权重 (基于 Attention) ---
        # gates: (Batch, Feature_Dim)
        gates = self.attention_gating(x_context)
        x_gated = x_feat * gates.unsqueeze(1)
        x = self.embedding(x_gated)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.head(x[:, -1, :])

class ContextGatedSingleGRU(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 context_dim,
                 hidden_dim=64, 
                 num_layers=2, 
                 dropout=0.1,
                 gate_internal_dim=64):
        super().__init__()
        self.context_gate = ContextAwareAttentionGating(
            context_dim=context_dim,
            feature_dim=feature_dim,
            internal_dim=gate_internal_dim,
            dropout=dropout
        )
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_feat, x_context):
        weights = self.context_gate(x_context)
        x_gated = x_feat * weights.unsqueeze(1)
        out, _ = self.gru(x_gated)
        last_step_feat = self.norm(out[:, -1, :])
        prediction = self.head(last_step_feat)
        
        return prediction

class ContextGatedMultiScaleGRU(nn.Module):
    def __init__(self, 
                 input_dim,         # 原始时序特征维度
                 context_dim,       # 上下文维度
                 hidden_dim=64, 
                 num_layers=1,      # 每个分支 GRU 的层数
                 dropout=0.1,
                 scales=[1, 3, 6],  # 定义不同的时间尺度分支
                 gate_internal_dim=64):
        super().__init__()
        
        self.scales = scales
        num_branches = len(scales)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.branch_gate = ContextAwareAttentionGating(
            context_dim=context_dim,
            feature_dim=num_branches,  # <--- 控制 N 个分支的权重
            internal_dim=gate_internal_dim,
            dropout=dropout
        )
        self.grus = nn.ModuleList()
        for _ in scales:
            self.grus.append(
                nn.GRU(
                    input_size=input_dim, 
                    hidden_size=hidden_dim, 
                    num_layers=num_layers, 
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
            )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_feat, x_context):
        B, T, D = x_feat.shape
        branch_outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                x_input = x_feat
            else:
                x_trans = x_feat.permute(0, 2, 1)
                x_pooled = F.avg_pool1d(x_trans, kernel_size=scale, stride=scale, ceil_mode=True)
                x_input = x_pooled.permute(0, 2, 1)
            out, _ = self.grus[i](x_input)
            last_step = out[:, -1, :]
            branch_outputs.append(last_step)
            
        feature_stack = torch.stack(branch_outputs, dim=1)
        branch_weights = self.branch_gate(x_context)
        weighted_features = feature_stack * branch_weights.unsqueeze(-1)
        fused_feature = weighted_features.sum(dim=1)
        out = self.norm(fused_feature)
        return self.head(out)
    
class ContextGatedMultiGRU(nn.Module):
    def __init__(self, 
                 input_dim, 
                 context_dim,
                 hidden_dim=64, 
                 scales=[1, 3, 6], # 不同的采样步长
                 num_layers=1,
                 dropout=0.1,
                 gate_internal_dim=64):
        super().__init__()
        self.scales = scales
        self.feature_gate = ContextAwareAttentionGating(
            context_dim=context_dim,
            feature_dim=input_dim,
            internal_dim=gate_internal_dim,
            dropout=dropout
        )
        
        # B. 多尺度处理分支 (Multi-Scale Branches)
        self.grus = nn.ModuleList()
        self.projections = nn.ModuleList()
        
        for scale in scales:
            # 这里的 input_dim 是经过门控后的特征维度，保持不变
            self.grus.append(
                nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=0 if num_layers==1 else dropout)
            )
            # 投影层，增加非线性能力
            self.projections.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ))
            
        # C. 融合层 (Fusion Layer)
        # 学习如何根据 hidden state 自身的质量来融合不同尺度
        self.fusion_attention = nn.Linear(hidden_dim, 1)
        
        # D. 最终输出头
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_feat, x_context):
        """
        x_feat:    (Batch, Seq_Len, Input_Dim) - 时序特征
        x_context: (Batch, Context_Dim)        - 截面/环境特征
        """
        B, T, D = x_feat.shape
        
        # --- 1. 上下文特征门控 (Context Gating) ---
        # 利用 context 生成特征权重
        # feature_weights: (Batch, Input_Dim)
        feature_weights = self.feature_gate(x_context)
        
        # 将权重应用到时序特征的所有时间步上
        # x_feat: (B, T, D) * weights: (B, 1, D) -> (B, T, D)
        x_gated = x_feat * feature_weights.unsqueeze(1)
        
        # --- 2. 多尺度并行处理 (Multi-Scale Processing) ---
        multi_scale_features = []
        
        for i, scale in enumerate(self.scales):
            # 下采样逻辑
            if scale == 1:
                x_input = x_gated
            else:
                # Permute to (B, D, T) for pooling
                x_trans = x_gated.permute(0, 2, 1)
                # AvgPool1d 可以平滑高频噪音，提取低频趋势
                x_pooled = F.avg_pool1d(x_trans, kernel_size=scale, stride=scale, ceil_mode=True)
                x_input = x_pooled.permute(0, 2, 1)
            
            # GRU 编码
            # 我们只取最后一个时间步的隐状态作为该尺度的 Summary
            # out: (B, T_scaled, H), h_n: (Num_Layers, B, H)
            out, h_n = self.grus[i](x_input)
            
            # 取最后一层的 hidden state
            last_hidden = h_n[-1] 
            
            # 投影
            feat = self.projections[i](last_hidden)
            multi_scale_features.append(feat)

        # --- 3. 注意力融合 (Attention Fusion) ---
        # stack: (B, Num_Scales, Hidden_Dim)
        stack = torch.stack(multi_scale_features, dim=1)
        
        # 计算每个尺度的注意力分数
        # scores: (B, Num_Scales, 1)
        attn_scores = self.fusion_attention(stack)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 加权求和
        # (B, Num_Scales, Hidden) * (B, Num_Scales, 1) -> Sum dim 1
        fused_feat = (stack * attn_weights).sum(dim=1)
        
        # --- 4. 输出 ---
        out = self.norm(fused_feat)
        return self.output_head(out)

class UnifiedModelInference:
    def __init__(self, config_path, model_path, device='cuda'):
        self.config = self._load_config(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # 自动获取数据目录
        self.data_dir = self.config['paths'].get('processed_data_dir', './processed_data')
        
        col_indices_path = 'col_indices.pkl'
        scaler_path = 'scaler.pkl'
        
        # 尝试从当前目录或数据目录加载 Scaler
        if not os.path.exists(col_indices_path):
            col_indices_path = os.path.join(self.data_dir, 'col_indices.pkl')
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(self.data_dir, 'scaler.pkl')
            
        if not os.path.exists(col_indices_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing preprocessing files. Check 'col_indices.pkl' and 'scaler.pkl'")

        self.col_config = joblib.load(col_indices_path)
        self.scaler = joblib.load(scaler_path)
        # 加载门控特征的 Scaler (必须存在)
        if os.path.exists('gate_scaler.pkl'):
            self.marketscaler = joblib.load('gate_scaler.pkl')
        else:
             # 如果找不到，尝试去 data_dir 找
            self.marketscaler = joblib.load(os.path.join(self.data_dir, 'gate_scaler.pkl'))
            
        self.feature_cols = self.col_config['feature_cols']
        self.ap_indices = self.col_config['ap_indices']
        self.bp_indices = self.col_config['bp_indices']
        
        print(f"Loading Unified Model from {model_path}...")
        self.model = self._build_and_load_model(model_path)
        
    def _load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_and_load_model(self, model_path):
        """根据配置构建模型结构并加载权重"""
        input_dim = self.config['model']['input_dim']
        hidden_dim = self.config['model']['hidden_dim']
        num_layers = self.config['model']['num_layers']
        dropout = self.config['model']['dropout']
        num_heads = self.config['model'].get('num_heads', 4)
        
        # 这里的 10 对应 gate_cols 的数量，如果不同请修改
        context_dim = 10
        config = self.config
        if self.config['modeltype']=='sLSTM':
            model = ContextAwareSLSTMPredictor(
                feature_dim=config['model']['input_dim'],
                context_dim=10,
                hidden_dim=config['model']['hidden_dim'],
                num_layers=config['model']['num_layers'],
                num_heads=config['model']['num_heads'],
                dropout=config['model']['dropout']
            )
        elif config['modeltype']=='GRU':
            model = ContextGatedSingleGRU(
                feature_dim=config['model']['input_dim'],
                context_dim=10,
                hidden_dim=config['model']['hidden_dim'],
                num_layers=config['model']['num_layers'],
                dropout=config['model']['dropout']
            )
        elif config['modeltype']=='Multi_GRU':
            model = ContextGatedMultiGRU(
                input_dim=config['model']['input_dim'],
                context_dim=10,
                hidden_dim=config['model']['hidden_dim'],
                num_layers=config['model']['num_layers'],
                dropout=config['model']['dropout'],
                scales = config['model']['scale']
            )
        
        model.to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _predict_single_side(self, feats_tensor, market_tensor, sites_code, target_code, model):
        """
        辅助函数：对单边数据进行推理
        """
        seq_len = self.config['training']['seq_len']
        dilation = self.config['training'].get('dilation', 1)
        batch_size = self.config['training']['batch_size']
        dummy_labels = torch.zeros(len(sites_code))
        
        # 构建 Dataset (会自动根据 target_code 筛选出 AP 或 BP 的有效行)
        dataset = SelectedIndicesDataset(
            features=feats_tensor, 
            indicators=market_tensor, # 传入市场特征
            labels=dummy_labels, 
            sites=sites_code,
            target_code=target_code, 
            seq_len=seq_len, 
            dilation=dilation
        )
        
        if len(dataset) == 0:
            return np.array([]), np.array([])
            
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        preds_list = []
        
        with torch.no_grad():
            for x_feat, x_ind, _ in loader:
                x_feat = x_feat.to(self.device)
                x_ind = x_ind.to(self.device)
                # 使用传入的 model 进行预测
                pred = model(x_feat, x_ind)
                if pred.ndim == 0: pred = pred.unsqueeze(0)
                preds_list.append(pred.detach().cpu().numpy())
                
        all_preds = np.concatenate(preds_list).ravel() if preds_list else np.array([])
        return dataset.valid_anchors, all_preds

    def predict_csv(self, input_path, market_path, output_path=None):
        print(f"Processing {input_path}...")
        
        # 1. 读取基础特征数据
        try:
            if input_path.endswith('.parquet'):
                df = pd.read_parquet(input_path)
            else:
                df = pd.read_csv(input_path)
        except Exception as e:
            print(f"Error reading input file: {e}")
            return None
        create_timestamp_fast(df)

        # 2. 读取并计算市场/门控数据
        try:
            if market_path.endswith('.parquet'):
                market = pd.read_parquet(market_path)
            else:
                market = pd.read_csv(market_path)
        except Exception as e:
            print(f"Error reading market file: {e}")
            return None
            
        # 确保时间戳对齐
        market['hms'] = pd.to_datetime(market['hms'])
        ms_delta = pd.to_timedelta(market['ms'], unit='ms')
        market['timestamp'] = market['hms'] + ms_delta
        
        # 计算技术指标
        market_factor = calculate_gating_indicators(market)
        
        # 定义需要用到的列 (Gate Columns)
        # 注意：这里的列表必须和训练时完全一致，且顺序一致
        gate_cols = ['spread', 'depth', 'rv_5', 'range',
                     'l1_imbalance', 'l5_imbalance', 'order_imbalance',
                     'oi_change', 'trade_intensity', 'rsi', 'dist_avg', 'trend']
        
        # 标准化 Gate 特征
        scaled_vals = self.marketscaler.transform(market_factor[gate_cols].values)
        market_factor[gate_cols] = scaled_vals.astype(np.float32)
        
        # 实际输入到模型的列 (包含 decay 和 impulse)
        indicators = ['depth','rv_5','range','l5_imbalance','oi_change','trade_intensity','trend','spread','gate_session_decay','gate_open_impulse']
        # 补充：确保 decay 和 impulse 存在 (calculate_gating_indicators 已计算)
        
        # 3. 合并数据
        df = pd.merge(
            df, 
            market_factor[['timestamp'] + indicators], # 只merge需要的列
            on='timestamp', 
            how='left'
        )    
        df[indicators] = df[indicators].fillna(0.0)

        if 'featSite' not in df.columns:
            raise ValueError("Input CSV must contain 'featSite' column.")

        # 4. 准备 Tensor
        try:
            feats_raw = df[self.feature_cols].values
        except KeyError as e:
            print(f"Missing columns: {e}")
            return None
            
        feats_scaled = self.scaler.transform(feats_raw).astype(np.float32)
        
        # 准备市场上下文 Tensor
        # 注意这里只取 indicators 列，且顺序必须与训练一致
        # (N, 10)
        market_context_data = df[indicators].values.astype(np.float32) 
        market_tensor = torch.from_numpy(market_context_data)
        
        N = len(df)
        sites_str = df['featSite'].astype(str).str.lower().values
        sites_code = np.zeros(N, dtype=np.int8)
        sites_code[sites_str == 'ap'] = 1
        sites_code[sites_str == 'bp'] = 2
        
        final_preds = np.full(N, np.nan, dtype=np.float32)
        
        # --- AP 推理 (使用 self.model) ---
        feats_ap_pure = feats_scaled[:, self.ap_indices]
        ap_indices, ap_preds = self._predict_single_side(
            torch.from_numpy(feats_ap_pure), market_tensor, sites_code, 1, self.model
        )
        if len(ap_indices) > 0: final_preds[ap_indices] = ap_preds

        # --- BP 推理 (使用 self.model) ---
        feats_bp_pure = feats_scaled[:, self.bp_indices]
        bp_indices, bp_preds = self._predict_single_side(
            torch.from_numpy(feats_bp_pure), market_tensor, sites_code, 2, self.model
        )
        if len(bp_indices) > 0: final_preds[bp_indices] = bp_preds

        df['prediction'] = final_preds
        if 'label' in df.columns:
            valid_mask = ~np.isnan(final_preds)
            if valid_mask.sum() > 0:
                print(f"IC: {df.loc[valid_mask, 'prediction'].corr(df.loc[valid_mask, 'label']):.4f}")
            
        valid_count = (~np.isnan(final_preds)).sum()
        print(f"Done. {valid_count}/{N} predicted.")
        return df

# ==========================================
# 4. 主程序入口
# ==========================================
def build_date_file_map(input_dir, suffix=".parquet"):
    """
    扫描文件夹，生成一个 { '20250812': '/full/path/to/xxx_20250812_xxx.parquet' } 的字典
    """
    # 1. 获取所有指定后缀的文件
    all_files = glob.glob(os.path.join(input_dir, f"*{suffix}"))
    
    date_map = {}
    # 正则：匹配 2025 开头的8位数字
    pattern = re.compile(r'(2025\d{4})')
    
    for f_path in all_files:
        f_name = os.path.basename(f_path)
        match = pattern.search(f_name)
        if match:
            date_str = match.group(1)
            # 如果同一天有多个文件，这里会覆盖，或者你可以加逻辑报错
            date_map[date_str] = f_path
            
    return date_map

def determine_actions(test_data_predict, threshold, use_pred=True):
    # 直接从DataFrame中提取相关列
    label_or_pred_column = 'pred' if use_pred else 'label'

    # 使用布尔索引来处理不同的条件
    condition_ap = (test_data_predict['featSite'] == 'AP')
    condition_bp = (test_data_predict['featSite'] == 'BP')

    # 根据条件计算 Thev
    Thev = pd.Series(index=test_data_predict.index, dtype=float)
    Thev[condition_ap] = test_data_predict['refPrice'] - test_data_predict[label_or_pred_column]
    Thev[condition_bp] = test_data_predict['refPrice'] + test_data_predict[label_or_pred_column]

    # 计算动作
    action_buy = (Thev - test_data_predict['ap1']) > threshold
    action_sale = (test_data_predict['bp1'] - Thev) > threshold

    # 默认行为是 'no_trade'
    actions = ['no_trade'] * len(test_data_predict)
    actions = pd.Series(actions)  # 确保 actions 是 Pandas Series 类型

    # 更新动作
    actions[action_buy] = 'buy'
    actions[action_sale] = 'sale'

    # 将结果添加到DataFrame
    test_data_predict['action'] = actions

    return test_data_predict

if __name__ == "__main__":
    # 1. 定义命令行参数
    parser = argparse.ArgumentParser(description="Unified Context-Aware Model Inference Pipeline")
    
    # 必需参数
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config JSON')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the unified trained model (.pth)')
    parser.add_argument('--modelname', type=str, required=True, help='Name identifier for the output file (e.g., experiment_v1)')
    
    # 可选参数
    parser.add_argument('--input_dir', type=str, default='./merged_features_test_withlabel', 
                        help='Directory containing input feature files')
    parser.add_argument('--market_dir', type=str, default='/home/zyyuan/project1/try/out_market_data', 
                        help='Directory containing market data files')
    parser.add_argument('--device', type=str, default='cuda:0', help='Computing device (default: cuda:0)')
    
    args = parser.parse_args()

    # 2. 打印运行信息
    print("-" * 30)
    print(f"Task: {args.modelname}")
    print(f"Config: {args.config}")
    print(f"Unified Model: {args.model_path}")
    print(f"Input Dir: {args.input_dir}")
    print(f"Market Dir: {args.market_dir}")
    print("-" * 30)
    feature_file_map = build_date_file_map(args.input_dir, suffix="_features.parquet")
    market_file_map = build_date_file_map(args.market_dir, suffix=".csv")
    # 3. 初始化推理引擎
    engine = UnifiedModelInference(args.config, args.model_path, device=args.device)
    
    # 4. 循环处理文件
    target_dates =  ['20250901', '20250902', '20250903', '20250904', '20250908', '20250909', '20250910', '20250911', '20250912', '20250915', '20250916', '20250917', '20250918', '20250919', '20250922', '20250923', '20250924', '20250925', '20250926', '20250929', '20250930', '20251009', '20251010', '20251013', '20251014', '20251015', '20251016', '20251017']
    
    df_all_list = []
    
    for date in target_dates:
        input_file = feature_file_map.get(date)
        market_file = market_file_map.get(date)
        
        # 检查文件是否存在
        if not os.path.exists(input_file):
            print(f"Warning: Input file not found {input_file}, skipping...")
            continue
        if not os.path.exists(market_file):
            print(f"Warning: Market file not found {market_file}, skipping...")
            continue
            
        df_result = engine.predict_csv(input_file, market_file)
        
        if df_result is not None:
            # 提取需要的列
            cols_to_keep = ['label', 'prediction', 'LABEL_CAL_DQ_inst1_30', 'featSite', 'ap1', 'bp1','refPrice']
            existing_cols = [c for c in cols_to_keep if c in df_result.columns]
            df_all_list.append(df_result[existing_cols])

    # 5. 合并、计算指标与可视化 (保持原有逻辑)
    if df_all_list:
        df_all = pd.concat(df_all_list, ignore_index=True)
        output_filename = f"{args.modelname}.pkl"
        df_all.to_pickle(output_filename)
        print(f"\nAll predictions saved to: {os.path.abspath(output_filename)}")
        print(f"Total rows: {len(df_all)}")

        df_all['cost'] = 0.0

        # Cost 计算
        mask_bp = df_all['featSite'].astype(str).str.lower() == 'bp'
        mask_ap = df_all['featSite'].astype(str).str.lower() == 'ap'
        
        results = []
        # 阈值扫描 (使用预测值的绝对分布范围，这里假设模型输出在 0~2.5 之间)
        # 如果模型输出是 logit 或 normalized value，可能需要调整这个范围
        thresholds = np.arange(-0.5, 3.0, 0.01)
        mask_bp = df_all['featSite'].astype(str).str.lower() == 'bp'
        mask_ap = df_all['featSite'].astype(str).str.lower() == 'ap'
        df_all.loc[mask_bp,'theoprice'] = df_all.loc[mask_bp,'refPrice']+df_all.loc[mask_bp,'prediction']
        df_all.loc[mask_ap,'theoprice'] = df_all.loc[mask_ap,'refPrice']-df_all.loc[mask_ap,'prediction']
        df_all.loc[mask_ap,'future_price'] = df_all.loc[mask_ap,'refPrice'] - df_all.loc[mask_ap,'label']
        df_all.loc[mask_bp,'future_price'] = df_all.loc[mask_bp,'refPrice'] + df_all.loc[mask_bp,'label']
        for threshold in thresholds:
            # print(threshold)
            buy_signal = df_all[df_all['theoprice']>(df_all['ap1']+threshold)]
            sell_signal = df_all[df_all['theoprice']<(df_all['bp1']-threshold)]
            buy_profit = buy_signal['future_price'] - buy_signal['ap1'] - (0.5/10000)*buy_signal['ap1']
            sell_profit = sell_signal['bp1'] - sell_signal['future_price'] - (0.5/10000)*sell_signal['bp1']
            DQ_buy = buy_profit.sum()
            DQ_sell = sell_profit.sum()
            DQ = (DQ_buy+DQ_sell)
            All_move = buy_profit.abs().sum()+sell_profit.abs().sum()
            DQ_neg = (All_move - DQ) /2
            DQ_pos = (DQ + DQ_neg)
            DQR = DQ_pos/DQ_neg
            DQ = 15*DQ
            signal_count = len(buy_signal)+len(sell_signal)
            results.append({
                        'Threshold': threshold,
                        'DQ': DQ,
                        'DQR': DQR,
                        'Count': signal_count
                    })

        df_res = pd.DataFrame(results)
        # 过滤掉信号过少的点，避免 DQR 虚高
        df_res = df_res[df_res['Count'] > 50] 
        df_res = df_res[df_res['DQ'] >=0]
        print(df_res[df_res['DQR'] >=1.5].head())
        df_res.to_csv(f"{args.modelname}_predtable.csv")
        # --- 绘图逻辑 (保持不变) ---
        if not df_res.empty:
            sns.set_theme(style="whitegrid")
            fig, ax1 = plt.subplots(figsize=(12, 6))

            color_dq = '#2a9d8f'
            ax1.plot(df_res['Threshold'], df_res['DQ'], color=color_dq, linewidth=2.5, label='DQ (Total Profit)')
            ax1.set_xlabel('Threshold', fontsize=12)
            ax1.set_ylabel('DQ (Profit)', color=color_dq, fontsize=12, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor=color_dq)
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            color_dqr = '#e76f51'
            ax2.plot(df_res['Threshold'], df_res['DQR'], color=color_dqr, linewidth=2.5, linestyle='--', label='DQR')
            ax2.set_ylabel('DQR (Ratio)', color=color_dqr, fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=color_dqr)
            ax2.grid(False)

            # 标注最大值
            if df_res['DQ'].max() > -np.inf:
                max_dq_idx = df_res['DQ'].idxmax()
                max_dq_val = df_res.loc[max_dq_idx, 'DQ']
                max_dq_thresh = df_res.loc[max_dq_idx, 'Threshold']
                ax1.annotate(f'Max DQ: {max_dq_val:.0f}\n@ {max_dq_thresh:.2f}',
                             xy=(max_dq_thresh, max_dq_val), 
                             xytext=(max_dq_thresh, max_dq_val),
                             arrowprops=dict(facecolor=color_dq, shrink=0.05),
                             color=color_dq, fontweight='bold')

            plt.title(f'Threshold Analysis: {args.modelname}', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(f"{args.modelname}.jpg")
            print(f"Plot saved to {args.modelname}.jpg")