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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def load_feature_config(json_path, all_train_columns,side):
    """
    读取分组配置，并将特征名转换为对应的列索引
    """
    # 1. 读取 JSON
    with open(json_path, 'r') as f:
        group_config = json.load(f) # 得到 {'Expert_1': ['f1_bp', ...], ...}
    

    # 2. 建立映射表: 特征名 -> 索引 (例如 'f1_bp' -> 0)
    # 这一步极其重要，Tensor切片需要整数索引
    feat_to_idx = {name: i for i, name in enumerate(all_train_columns)}
    
    # 3. 转换: 将每个Expert下的特征名列表，变成索引列表
    group_indices = []
    
    # 确保按 Expert_1, Expert_2... 顺序读取，防止字典乱序
    sorted_keys = sorted(group_config.keys()) 
    
    for expert_name in sorted_keys:
        feature_names = group_config[expert_name]
        if side=='buy':
            feature_names = [feature+'bp' for feature in feature_names]
        else:
            feature_names = [feature+'ap' for feature in feature_names]
        # 查找对应的索引
        indices = [feat_to_idx[name] for name in feature_names if name in feat_to_idx]
        
        if len(indices) == 0:
            print(f"警告: {expert_name} 没有匹配到任何特征！")
            
        group_indices.append(indices)
        print(f"已加载 {expert_name}: 包含 {len(indices)} 个特征")
        
    return group_indices
def seed_everything(seed: int = 42):
    """
    设置全局随机种子，保证结果可复现
    """
    # 1. Python 原生
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止 hash 随机化

    # 2. Numpy
    np.random.seed(seed)

    # 3. PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多 GPU

    # 4. 保证 CuDNN 确定性 (会对性能有轻微影响)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Global seed set to {seed}")
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
def load_aligned_data_raw(npz_path, gate_dir, gate_cols, cut=False):
    """
    修改版: 返回对齐后的【全量】原始特征，不进行 AP/BP 切分。
    切分逻辑移交到 Dataset 构建阶段。
    """
    try:
        basename = os.path.basename(npz_path)
        # 提取日期
        match = re.search(r'(\d{8})', basename)
        if not match: return None, None, None, None
        date_str = match.group(1)
        
        # 寻找 Gate 文件
        search_pattern = os.path.join(gate_dir, f"*{date_str}*_gate.parquet")
        gate_files = glob.glob(search_pattern)
        if not gate_files:
            logging.warning(f"No gate file found for date {date_str}, skipping.")
            return None, None, None, None   
            
        gate_path = gate_files[0]
        
        # 加载 NPZ
        with np.load(npz_path, allow_pickle=True) as d:
            raw_feats = d['features'] # (N, Total_Feats)
            raw_labels = d['labels']
            sites = d['sites']
            timestamps = d['timestamps'] if 'timestamps' in d else d['hms']
            
        # 加载 Gate
        req_cols = ['timestamp'] + gate_cols
        df_gate = pd.read_parquet(gate_path, columns=req_cols)
        
        # 构建索引对齐
        df_npz_idx = pd.DataFrame({
            'timestamp': timestamps,
            'orig_idx': np.arange(len(timestamps))
        })
        
        df_npz_idx['timestamp'] = pd.to_datetime(df_npz_idx['timestamp'])
        df_gate['timestamp'] = pd.to_datetime(df_gate['timestamp'])
        
        # Inner Join
        merged_df = pd.merge(df_npz_idx, df_gate, on='timestamp', how='inner')
        
        if len(merged_df) == 0:
            logging.warning(f"No overlapping data for {date_str}.")
            return None, None, None, None
            
        valid_indices = merged_df['orig_idx'].values
        
        # 提取数据 (注意：这里返回的是包含所有列的 raw_feats)
        aligned_feats = raw_feats[valid_indices] 
        aligned_labels = raw_labels[valid_indices]
        aligned_sites = sites[valid_indices]
        
        # 提取门控特征
        aligned_indicators = merged_df[gate_cols].values.astype(np.float32)
        if cut:
            aligned_indicators = np.clip(aligned_indicators, -3.0, 3.0)
            
        # 转 Tensor
        feats_tensor = torch.from_numpy(aligned_feats).float()
        labels_tensor = torch.from_numpy(aligned_labels).float()
        indicators_tensor = torch.from_numpy(aligned_indicators).float()
        
        return feats_tensor, labels_tensor, aligned_sites, indicators_tensor

    except Exception as e:
        logging.error(f"Error processing {npz_path}: {e}")
        return None, None, None, None

class SelectedIndicesDataset(torch.utils.data.Dataset):
    def __init__(self, features, indicators, labels, sites, target_code, seq_len, dilation=1):
        """
        新增 indicators 参数: (N, 12)
        """
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

def load_and_process_day(fpath, ap_indices, bp_indices, target_code):
    try:
        with np.load(fpath) as d:
            raw_feats = torch.from_numpy(d['features']) 
            raw_labels = torch.from_numpy(d['labels'])  
            sites = d['sites']
            
            if target_code == 1:
                target_feats = raw_feats[:, ap_indices]
            elif target_code == 2: 
                target_feats = raw_feats[:, bp_indices]
            else:
                raise ValueError(f"Unknown target_code: {target_code}")
            return target_feats, raw_labels, sites
            
    except Exception as e:
        logging.warning(f"Error loading {fpath}: {e}")
        return None, None, None

def setup_logging(save_dir):
    log_file = os.path.join(save_dir, 'run.log')
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_date_from_filename(filepath):
    match = re.search(r'(\d{8})', os.path.basename(filepath))
    return int(match.group(1)) if match else 0

class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(GRUPredictor, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
class MixerBlock(nn.Module):
    def __init__(self, seq_len, num_features, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(num_features)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(num_features)
        self.channel_mlp = nn.Sequential(
            nn.Linear(num_features, num_features * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(num_features * 4, num_features), nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.time_mlp(self.norm1(x).permute(0,2,1)).permute(0,2,1)
        x = x + self.channel_mlp(self.norm2(x))
        return x

class TimeMixerPredictor(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=64, num_layers=2, dropout=0.1, downsample_levels=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.levels = downsample_levels + 1
        self.blocks = nn.ModuleList()
        curr_len = seq_len
        for _ in range(self.levels):
            self.blocks.append(nn.Sequential(*[MixerBlock(curr_len, hidden_dim, dropout) for _ in range(num_layers)]))
            curr_len //= 2
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.head = nn.Linear(hidden_dim, 1)
        self.weights = nn.Parameter(torch.ones(self.levels) / self.levels)
    def forward(self, x):
        x = self.embedding(x)
        outputs = []
        for i in range(self.levels):
            x_out = self.blocks[i](x)
            outputs.append(x_out[:, -1, :])
            if i < self.levels - 1:
                x = self.downsample(x.permute(0, 2, 1)).permute(0, 2, 1)
        final = sum(o * w for o, w in zip(outputs, F.softmax(self.weights, dim=0)))
        return self.head(final)
import torch
import torch.nn as nn
class CCCLoss(nn.Module):
    """
    Lin's Concordance Correlation Coefficient (CCC) Loss.
    Loss = 1 - CCC
    
    CCC 衡量的是预测序列与真实序列在【均值】、【方差】和【相关性】上的综合一致度。
    范围 [-1, 1]，也就是 Loss 范围 [0, 2]。
    
    - 解决了 MSE 导致的"预测值坍缩为0"的问题 (因为方差差异会被惩罚)
    - 解决了 IC Loss 导致的"忽略数值绝对大小"的问题 (因为均值偏移会被惩罚)
    """
    def __init__(self, eps=1e-8):
        super(CCCLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        # 1. 展平向量 (Batch 维度和序列维度合并，计算全局统计量)
        # 假设输入形状可能是 [Batch, Seq_Len] 或 [Batch, 1]
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 2. 计算均值
        mean_pred = torch.mean(pred_flat)
        mean_target = torch.mean(target_flat)
        
        # 3. 计算方差 (使用 unbiased=False 对应公式中的总体方差，训练更稳定)
        var_pred = torch.var(pred_flat, unbiased=False)
        var_target = torch.var(target_flat, unbiased=False)
        
        # 4. 计算协方差
        # Cov = E[(x - ux)(y - uy)]
        covariance = torch.mean((pred_flat - mean_pred) * (target_flat - mean_target))
        
        # 5. 计算 CCC
        # 分子: 2 * Covariance
        numerator = 2 * covariance
        
        # 分母: var_pred + var_target + (mean_diff)^2
        # 这一步体现了 CCC 的精髓：它不仅要求相关(Cov大)，还要求方差一致(var项)，且均值一致(mean差项)
        denominator = var_pred + var_target + (mean_pred - mean_target)**2
        
        ccc = numerator / (denominator + self.eps)
        
        # 6. Loss = 1 - CCC
        # 我们希望 CCC 趋近于 1
        return 1.0 - ccc
class NativeLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        last_step = self.norm(last_step)
        return self.head(last_step)

import torch
import torch.nn as nn
import math

# --- JIT 加速的核心扫描算子 ---
@torch.jit.script
def slstm_scan_jit(
    i_gate: torch.Tensor, 
    f_gate: torch.Tensor, 
    z_gate: torch.Tensor, 
    o_gate: torch.Tensor,
    init_c: torch.Tensor, 
    init_n: torch.Tensor, 
    init_m: torch.Tensor
):
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

# --- sLSTM 单元 ---
class ContextGating(nn.Module):
    def __init__(self, feature_dim, context_dim, hidden_dim=64):
        """
        feature_dim: 原始特征数量 (例如 409)
        context_dim: 市场指标数量 (例如 10)
        hidden_dim: 门控网络的中间层维度
        """
        super().__init__()
        
        # 这是一个简单的 MLP，将市场状态映射为 0-1 之间的特征权重
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),  # 激活函数
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid() # 关键：输出必须在 [0, 1] 之间，充当门控
        )

    def forward(self, context):
        """
        context: (Batch, Context_Dim)
        Output: (Batch, Feature_Dim) -> 每个特征对应的权重
        """
        return self.gate_net(context)

class SoftmaxScaleGating(nn.Module):
    def __init__(self, context_dim, feature_dim, temperature=1.0, hidden_dim=64):
        """
        context_dim: 市场指标维度 (Input)
        feature_dim: 特征维度 (Output, 即 d_output)
        temperature: 温度系数 (beta)，控制分布的陡峭程度
        """
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

# --- 3. 集成 Context Gating 的预测模型 ---
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
        
        # --- 2. 应用权重 (广播) ---
        # x_feat: (B, T, F) * gates: (B, 1, F)
        # 这一步实现了"上下文感知的特征缩放"
        x_gated = x_feat * gates.unsqueeze(1)
        
        # --- 3. sLSTM 前向传播 ---
        x = self.embedding(x_gated)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        
        # 取最后一个时间步进行预测
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

def evaluate_model(model, test_files, config, scaler, col_config, device, threshold, gate_cols):
    """
    修改后的评估函数：同时评估 AP 和 BP
    """
    model.eval()
    feature_cols = col_config['feature_cols']
    ap_indices = col_config['ap_indices']
    bp_indices = col_config['bp_indices']
    gate_dir = config['paths']['gate_dir']
    
    horizon_cols = [
        'LABEL_CAL_DQ_inst1_1', 'LABEL_CAL_DQ_inst1_3', 'LABEL_CAL_DQ_inst1_5', 
        'LABEL_CAL_DQ_inst1_10', 'LABEL_CAL_DQ_inst1_15', 'LABEL_CAL_DQ_inst1_30', 
        'LABEL_CAL_DQ_inst1_60'
    ]
    price_cols = ['ap1', 'bp1'] 
    
    # 收集容器
    all_preds = []
    all_labels = []
    all_costs = [] # 存储成本价
    all_horizons = {h: [] for h in horizon_cols}
    all_is_bp = [] # 标记是否为BP模式 (True=Buy, False=Sell)
    
    for fpath in test_files:
        try:
            # 1. 读取数据
            req_cols = list(set(feature_cols + ['featSite','timestamp'] + price_cols + horizon_cols + ['label']))
            df_raw = pd.read_parquet(fpath)
            # 确保有 timestamp
            if 'timestamp' not in df_raw.columns:
                create_timestamp_fast(df_raw)
            df_raw = df_raw[[c for c in req_cols if c in df_raw.columns]]
            
            # 2. Gate 数据合并
            basename = os.path.basename(fpath)
            match = re.search(r'(\d{8})', basename)
            if not match: continue
            date_str = match.group(1)
            
            gate_search = os.path.join(gate_dir, f"*{date_str}*_gate.parquet")
            gate_files_found = glob.glob(gate_search)
            if not gate_files_found: continue

            df_gate = pd.read_parquet(gate_files_found[0], columns=['timestamp'] + gate_cols)
            df_merged = pd.merge(df_raw, df_gate, on='timestamp', how='inner')
            if len(df_merged) == 0: continue

            # 3. 准备 Tensor
            feats_raw = df_merged[feature_cols].values
            feats_scaled = scaler.transform(feats_raw).astype(np.float32)
            indicators_pure = df_merged[gate_cols].values.astype(np.float32)
            if config.get('cut', False):
                indicators_pure = np.clip(indicators_pure, -3.0, 3.0)
            
            feats_tensor = torch.from_numpy(feats_scaled) # 全量
            indicators_tensor = torch.from_numpy(indicators_pure)
            
            sites_str = df_merged['featSite'].astype(str).str.lower().values
            sites_code = np.zeros(len(df_merged), dtype=np.int8)
            sites_code[sites_str == 'ap'] = 1
            sites_code[sites_str == 'bp'] = 2
            
            dummy_labels = torch.zeros(len(df_merged))
            dilation = config['training'].get('dilation', 1)
            seq_len = config['training']['seq_len']

            # --- 双边循环 (AP & BP) ---
            tasks = [('ap', ap_indices, 1), ('bp', bp_indices, 2)]
            
            for mode, indices, t_code in tasks:
                # 提取对应列
                feats_side = feats_tensor[:, indices]
                
                ds = SelectedIndicesDataset(
                    feats_side, indicators_tensor, dummy_labels, sites_code, 
                    target_code=t_code, seq_len=seq_len, dilation=dilation
                )
                
                if len(ds) == 0: continue
                
                loader = DataLoader(ds, batch_size=config['training']['batch_size']*2, shuffle=False, num_workers=0)
                
                file_preds = []
                with torch.no_grad():
                    for x_feat, x_ind, _ in loader:
                        x_feat, x_ind = x_feat.to(device), x_ind.to(device)
                        pred = model(x_feat, x_ind).squeeze()
                        if pred.ndim == 0: pred = pred.unsqueeze(0)
                        file_preds.append(pred.cpu().numpy())
                
                if not file_preds: continue
                batch_preds = np.concatenate(file_preds)
                
                # 收集数据
                valid_idx = ds.valid_anchors
                curr_slice = df_merged.iloc[valid_idx].reset_index(drop=True)
                
                all_preds.append(batch_preds)
                if 'label' in curr_slice.columns:
                    all_labels.append(curr_slice['label'].values)
                else:
                    all_labels.append(np.zeros_like(batch_preds))
                
                # 记录成本: BP用ap1(买入价), AP用bp1(卖出价)
                if mode == 'bp':
                    all_costs.append(curr_slice['ap1'].values)
                    all_is_bp.append(np.ones(len(batch_preds), dtype=bool))
                else:
                    all_costs.append(curr_slice['bp1'].values)
                    all_is_bp.append(np.zeros(len(batch_preds), dtype=bool))
                    
                for h in horizon_cols:
                    if h in curr_slice.columns:
                        all_horizons[h].append(curr_slice[h].values)
                    else:
                        all_horizons[h].append(np.zeros_like(batch_preds))
                        
        except Exception as e:
            logging.error(f"Error evaluating {fpath}: {e}")

    if not all_preds:
        return 0.0

    # --- 汇总计算 ---
    global_preds = np.concatenate(all_preds)
    global_labels = np.concatenate(all_labels)
    global_costs = np.concatenate(all_costs)
    global_is_bp = np.concatenate(all_is_bp)
    
    # CCC & MSE
    mse = np.mean((global_preds - global_labels)**2)
    # 简化的 CCC 计算用于日志
    mean_p, mean_t = np.mean(global_preds), np.mean(global_labels)
    var_p, var_t = np.var(global_preds), np.var(global_labels)
    cov = np.mean((global_preds - mean_p)*(global_labels - mean_t))
    ccc = (2 * cov) / (var_p + var_t + (mean_p - mean_t)**2 + 1e-8)
    
    # PnL Calculation
    # 使用 90 分位数作为动态阈值 (或者传入的固定阈值)
    if threshold is None:
        threshold = np.quantile(global_preds, 0.9)
        
    signal_mask = global_preds > threshold
    total_signals = np.sum(signal_mask)
    
    log_msg = f"Eval Combined | MSE: {mse:.6f} | CCC: {ccc:.6f} | Signals: {total_signals}"
    
    final_score = ccc
    total_profit_max = -np.inf
    
    if total_signals > 0:
        for h in horizon_cols:
            if not all_horizons[h]: continue
            global_h_prices = np.concatenate(all_horizons[h])
            
            # PnL Logic:
            # BP (Buy): Future - Cost(Ask)
            # AP (Sell): Cost(Bid) - Future
            future_vals = global_h_prices[signal_mask]
            cost_vals = global_costs[signal_mask]
            is_bp_vals = global_is_bp[signal_mask]
            
            pnl_bp = future_vals - cost_vals
            pnl_ap = cost_vals - future_vals
            
            # 合并
            pnl = np.where(is_bp_vals, pnl_bp, pnl_ap)
            total_pnl = np.sum(pnl)
            
            short_name = h.split('_')[-1]
            log_msg += f" | {short_name}: {total_pnl:.2f}"
            if total_pnl > total_profit_max:
                total_profit_max = total_pnl
                
    logging.info(log_msg)
    return total_profit_max

from torch.utils.data import ConcatDataset, DataLoader
def train_pipeline(config, save_dir, device):
    logging.info(f"Starting Joint Training (AP + BP)...")
    data_dir = config['paths']['processed_data_dir']
    gate_dir = config['paths']['gate_dir']
    all_files = sorted(glob.glob(os.path.join(data_dir, "day_*.npz")), key=get_date_from_filename)
    test_file_count = 14
    train_files = all_files[:-test_file_count]
    # train_files = all_files[:6]
    raw_data_dir = config['paths'].get('raw_data_dir', './merged_features_final')
    all_parquet = sorted(glob.glob(os.path.join(raw_data_dir, "*_features.parquet")), key=get_date_from_filename)
    test_files_parquet = all_parquet[-test_file_count:]
    # test_files_parquet = all_parquet[-2:]
    col_indices_path = os.path.join(data_dir, 'col_indices.pkl')
    col_config = joblib.load(col_indices_path)
    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    ap_indices = torch.tensor(col_config['ap_indices'], dtype=torch.long)
    bp_indices = torch.tensor(col_config['bp_indices'], dtype=torch.long)
    gate_cols = ['depth','rv_5','range','l5_imbalance','oi_change','trade_intensity','trend','spread','gate_session_decay','gate_open_impulse']
    if config['modeltype']=='sLSTM':
        model = ContextAwareSLSTMPredictor(
            feature_dim=config['model']['input_dim'],
            context_dim=len(gate_cols),
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout']
        ).to(device)
    elif config['modeltype']=='GRU':
        model = ContextGatedSingleGRU(
            feature_dim=config['model']['input_dim'],
            context_dim=len(gate_cols),
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        ).to(device)
    elif config['modeltype']=='Multi_GRU':
        # model = ContextGatedMultiScaleGRU(
        #     input_dim=config['model']['input_dim'],
        #     context_dim=len(gate_cols),
        #     hidden_dim=config['model']['hidden_dim'],
        #     num_layers=config['model']['num_layers'],
        #     dropout=config['model']['dropout'],
        #     scales = config['model']['scale']
        # ).to(device)
        model = ContextGatedMultiGRU(
            input_dim=config['model']['input_dim'],
            context_dim=len(gate_cols),
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            scales = config['model']['scale']
        ).to(device)
    
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)
    criterion = CCCLoss()
    
    best_model_path = os.path.join(save_dir, f"model_best_joint.pth")
    best_test_score = -np.inf
    # --- Training Loop ---
    logging.info(f"Training for {config['training']['epochs']} epochs...")
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        total_batches = 0
        random.shuffle(train_files)
        
        for f_idx, fpath in enumerate(train_files):
            cut = config.get('cut', False)
            
            # 1. 加载全量数据 (包含所有列)
            data_pack = load_aligned_data_raw(fpath, gate_dir, gate_cols, cut)
            if data_pack[0] is None: continue
            full_feats, labels, sites, indicators = data_pack
            
            dilation = config['training'].get('dilation', 1) 
            seq_len = config['training']['seq_len']
            
            # 2. 构建 AP 数据集 (筛选 AP 列 + target_code=1)
            feats_ap = full_feats[:, ap_indices]
            ds_ap = SelectedIndicesDataset(
                feats_ap, indicators, labels, sites, 
                target_code=1, seq_len=seq_len, dilation=dilation
            )
            
            # 3. 构建 BP 数据集 (筛选 BP 列 + target_code=2)
            feats_bp = full_feats[:, bp_indices]
            ds_bp = SelectedIndicesDataset(
                feats_bp, indicators, labels, sites, 
                target_code=2, seq_len=seq_len, dilation=dilation
            )
            
            # 4. 合并数据集 (Joint Training)
            datasets_to_combine = []
            if len(ds_ap) > 0: datasets_to_combine.append(ds_ap)
            if len(ds_bp) > 0: datasets_to_combine.append(ds_bp)
            
            if not datasets_to_combine: continue
            
            combined_ds = ConcatDataset(datasets_to_combine)
            train_loader = DataLoader(combined_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
            
            day_loss = 0
            for x_feat, x_ind, y in train_loader:
                x_feat, x_ind, y = x_feat.to(device), x_ind.to(device), y.to(device)
                
                optimizer.zero_grad()
                pred = model(x_feat, x_ind).squeeze()
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                day_loss += loss.item()
                total_batches += 1
                
            if (f_idx + 1) % 20 == 0:
                current_avg = day_loss/len(train_loader) if len(train_loader)>0 else 0
                sys.stdout.write(f"\r[Epoch {epoch+1}] Day {f_idx+1}/{len(train_files)} | Loss: {current_avg:.4f}")
                sys.stdout.flush()
                
            epoch_loss += day_loss
        avg_epoch_loss = epoch_loss / total_batches if total_batches > 0 else 0
        print(f"\n>> Epoch {epoch+1} Train Loss: {avg_epoch_loss:.6f}")
        

        
        logging.info(f"Evaluating Epoch {epoch+1}...")
        current_score = evaluate_model(model, test_files_parquet, config, scaler, col_config, device, threshold=None, gate_cols=gate_cols)
        if current_score > best_test_score:
            best_test_score = current_score
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"*** New Best Model Saved (CCC: {best_test_score:.6f}) ***")

    return best_model_path

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    save_dir = os.path.join(config['paths']['output_root'], config['experiment_name'])
    os.makedirs(save_dir, exist_ok=True)
    setup_logging(save_dir)
    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    train_pipeline(config, save_dir, device)