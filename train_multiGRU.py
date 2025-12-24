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
from torch.utils.data import ConcatDataset, DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

class SelectedIndicesDataset(Dataset):
    def __init__(self, features, labels, sites, target_code, seq_len, dilation=1):
        self.features = features
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
        x = self.features[row_end - self.span_offset : row_end + 1 : self.dilation]
        
        y = self.labels[row_end]
        return x, y


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

def load_raw_day_data(fpath):
    """
    加载一天的原始数据，返回完整特征、标签和sites。
    切片逻辑移交到训练循环中处理。
    """
    try:
        with np.load(fpath) as d:
            # 加载所有特征，不进行列筛选
            raw_feats = torch.from_numpy(d['features']).float() # 确保是 float
            raw_labels = torch.from_numpy(d['labels']).float()
            sites = d['sites']
            
            return raw_feats, raw_labels, sites
            
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
    
class MultiRateGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, scales=[1, 3, 10], dropout=0.1):
        super().__init__()
        self.scales = scales
        
        # 为每个尺度创建一个独立的 GRU
        self.grus = nn.ModuleList()
        self.projections = nn.ModuleList()
        
        for scale in scales:
            # 1. GRU 骨干
            self.grus.append(
                nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=1)
            )
            # 2. 投影层 (可选，用于增强特征)
            self.projections.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ))
            
        # 融合层：使用 Attention 机制来决定哪个尺度更重要
        self.fusion_attention = nn.Linear(hidden_dim, 1)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (Batch, Seq_Len, Input_Dim)
        """
        B, T, D = x.shape
        multi_scale_features = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                x_sampled = x
            else:
                x_trans = x.permute(0, 2, 1)
                x_pooled = F.avg_pool1d(x_trans, kernel_size=scale, stride=scale, ceil_mode=True)
                x_sampled = x_pooled.permute(0, 2, 1)

            out, h_n = self.grus[i](x_sampled)
            global_feat = h_n.squeeze(0)
            feat = self.projections[i](global_feat)
            multi_scale_features.append(feat) # List of (B, Hidden)
        stack = torch.stack(multi_scale_features, dim=1)
        attn_scores = F.softmax(self.fusion_attention(stack), dim=1)
        weighted_feat = (stack * attn_scores).sum(dim=1)
        return self.output_head(weighted_feat)

class SAGRUBlock(nn.Module):
    """
    单个尺度的处理单元：GRU -> Self-Attention -> Residual -> Norm
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1, num_heads=4):
        super().__init__()
        
        # 1. GRU 骨干
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 2. Self-Attention
        # 注意：embed_dim 必须能被 num_heads 整除
        if hidden_dim % num_heads != 0:
            num_heads = 1 # 兜底策略
            
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. 归一化与投影
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 4. 最终特征投影 (保持你原有的结构)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (Batch, Seq_Len_Scaled, Input_Dim)
        
        # --- A. GRU ---
        # out: (Batch, Seq_Len, Hidden) -> 包含历史所有时刻的信息
        # h_n: (Layers, Batch, Hidden) -> 包含最后时刻的信息
        out, h_n = self.gru(x)
        
        # --- B. Attention (Query-Key-Value) ---
        # Query: 我们只关心最后一步 (Current State)
        # shape: (Batch, 1, Hidden)
        query = out[:, -1:, :] 
        
        # Key/Value: 所有的历史步 (History Memory)
        # shape: (Batch, Seq_Len, Hidden)
        key = out
        value = out
        
        # attn_out: (Batch, 1, Hidden)
        attn_out, _ = self.attention(query, key, value)
        
        # --- C. Residual & Norm ---
        # 将 Attention 提取的上下文加回到当前状态上
        context = query + self.dropout(attn_out)
        context = self.norm(context)
        
        # 移除时间维度: (Batch, 1, Hidden) -> (Batch, Hidden)
        feat_flat = context.squeeze(1)
        
        # --- D. Projection ---
        return self.projection(feat_flat)


class MultiScaleSAGRU(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim=64, 
                 scales=[1, 3, 10], 
                 dropout=0.1,
                 numlayers = 1,
                 num_heads=4):
        super().__init__()
        self.scales = scales
        
        # 为每个尺度创建一个 SAGRUBlock
        self.branches = nn.ModuleList()
        
        for scale in scales:
            self.branches.append(
                SAGRUBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=numlayers,
                    dropout=dropout,
                    num_heads=num_heads
                )
            )
            
        # 融合层：决定哪个尺度更重要
        self.fusion_attention = nn.Linear(hidden_dim, 1)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (Batch, Seq_Len, Input_Dim)
        """
        B, T, D = x.shape
        multi_scale_features = []
        
        for i, scale in enumerate(self.scales):
            # 1. 降采样 (Downsampling)
            if scale == 1:
                x_input = x
            else:
                x_trans = x.permute(0, 2, 1)
                # ceil_mode=True 防止丢掉最后几个 tick
                x_pooled = F.avg_pool1d(x_trans, kernel_size=scale, stride=scale, ceil_mode=True)
                x_input = x_pooled.permute(0, 2, 1)

            # 2. 输入到对应的 SAGRU 分支
            # out shape: (Batch, Hidden_Dim)
            feat = self.branches[i](x_input)
            multi_scale_features.append(feat)

        # 3. 多尺度融合 (Fusion)
        # stack: (Batch, Num_Scales, Hidden)
        stack = torch.stack(multi_scale_features, dim=1)
        
        # 计算尺度权重 (Softmax Attention)
        attn_scores = self.fusion_attention(stack) # (B, Scales, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 加权求和
        weighted_feat = (stack * attn_weights).sum(dim=1)
        
        # 4. 输出
        return self.output_head(weighted_feat)

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
class TradingOpportunityLoss(nn.Module):
    """
    针对高频交易优化的损失函数。
    逻辑：
    1. Label > 0 (有机会): 
       - 如果预测到了 (Pred > 0): 正常 MSE
       - 如果错过了 (Pred <= 0): 重罚 (w_miss)
    2. Label <= 0 (噪音/成本区):
       - 如果正确空仓 (Pred <= 0): 轻罚/忽略 (w_noise)，允许拟合不精细
       - 如果虚假开仓 (Pred > 0): 重罚 (w_fp)，防止回撤
    """
    def __init__(self, w_miss=10.0, w_noise=0.1, w_fp=5.0,threshold = 0.0, reduction='mean'):
        """
        Args:
            w_miss (float): 错过盈利机会的惩罚权重 (Target > 0, Pred <= 0)
            w_noise (float): 噪音区域的松弛权重 (Target <= 0, Pred <= 0)
            w_fp (float): 虚假信号的惩罚权重 (Target <= 0, Pred > 0)
            reduction (str): 'mean', 'sum' or 'none'
        """
        super(TradingOpportunityLoss, self).__init__()
        self.w_miss = w_miss
        self.w_noise = w_noise
        self.w_fp = w_fp
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, pred, target):
        mse_loss = (pred - target) ** 2
        is_opportunity = (target > self.threshold)
        is_noise = (~is_opportunity)
        
        pred_positive = (pred > self.threshold)
        pred_negative = (~pred_positive)

        # 3. 定义四象限权重
        # 象限 A: 正确捕获 (Hit) -> 权重 1.0 (基准)
        # 象限 B: 错过机会 (Miss) -> 权重 w_miss
        # 象限 C: 虚假信号 (False Positive) -> 权重 w_fp
        # 象限 D: 正确观望 (Correct Hold) -> 权重 w_noise
        
        # 利用广播机制初始化权重矩阵，默认为 1.0
        weights = torch.ones_like(pred)
        
        # 应用权重 (使用 bool索引)
        # Miss: 这是一个很大的遗憾，重罚
        weights[is_opportunity & pred_negative] = self.w_miss
        
        # False Positive: 这是一个亏损交易，重罚
        weights[is_noise & pred_positive] = self.w_fp
        
        # Correct Hold: 这是震荡行情，没必要让模型拼命拟合 -0.5 还是 -0.6
        weights[is_noise & pred_negative] = self.w_noise
        
        # 4. 加权
        weighted_loss = mse_loss * weights

        # 5. Reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss

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
        gates = self.wx(x_conv) # (B, T, 4*H)
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
        return self.dropout(out) + x # Residual connection

# --- sLSTM 完整预测模型 ---
class sLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            sLSTMBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.head(x[:, -1, :])

class CausalConv1d(nn.Module):
    """
    因果卷积层：确保模型只能看到过去，不能看到未来。
    实现方式：通过 Padding 在左侧补齐，保证输出长度与输入一致。
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # 核心：计算左侧需要补多少零
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=self.padding, 
            dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out

class MultiHeadTCNBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads_dilations=[1, 2, 4, 8], dropout=0.1):
        super().__init__()
        self.heads_dilations = heads_dilations
        self.num_heads = len(heads_dilations)
        head_dim = hidden_dim // self.num_heads
        self.heads = nn.ModuleList([
            CausalConv1d(
                in_channels=in_dim,
                out_channels=head_dim,
                kernel_size=3,
                dilation=d
            )
            for d in heads_dilations
        ])
        
        self.total_head_dim = head_dim * self.num_heads
        self.fusion_conv = nn.Conv1d(self.total_head_dim, hidden_dim * 2, kernel_size=1) 
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(self.num_heads, hidden_dim) # GroupNorm 对多头非常友好
        self.res_proj = nn.Conv1d(in_dim, hidden_dim, 1) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x))
        x_concat = torch.cat(head_outputs, dim=1)
        x_fusion = self.fusion_conv(x_concat)
        x_val, x_gate = x_fusion.chunk(2, dim=1)
        x_out = x_val * torch.sigmoid(x_gate)
        x_out = self.dropout(x_out)
        return self.norm(x_out + self.res_proj(x))

class MultiHeadTCNPredictor(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim=64, 
                 num_layers=2, 
                 dropout=0.1,
                 dilations=[1, 2, 4, 8]):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                MultiHeadTCNBlock(hidden_dim, hidden_dim, heads_dilations=dilations, dropout=dropout)
            )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        x_final = self.norm(x)
        return self.head(x_final[:, -1, :])

class DecompositionGRU(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim=64, 
                 num_layers=1, 
                 dropout=0.1,
                 trend_window=5, 
                 trend_stride=5):
        super().__init__()
        self.trend_window = trend_window
        self.trend_stride = trend_stride
        self.gru_trend = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.ln_trend = nn.LayerNorm(hidden_dim)
        self.head_trend = nn.Linear(hidden_dim, 1)
        self.gru_res = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.ln_res = nn.LayerNorm(hidden_dim)
        self.head_res = nn.Linear(hidden_dim, 1)

        self.pad_left = trend_window // 2
        self.pad_right = trend_window // 2

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, Input_Dim)
        Returns:
            pred: (Batch, 1)
        """
        B, T, D = x.shape
        
        # ================= Step 1: 序列分解 =================
        
        # 1.1 计算趋势 (Moving Average)
        # Permute to (B, D, T) for pooling
        x_trans = x.permute(0, 2, 1)
        
        # 使用 replicate padding 避免边界出现 0 值干扰趋势
        # 这里的 padding 逻辑是为了让输出长度保持为 T
        x_padded = F.pad(x_trans, (self.pad_left, self.pad_right), mode='replicate')
        
        trend_trans = F.avg_pool1d(
            x_padded, 
            kernel_size=self.trend_window, 
            stride=1
        )
        
        # 转回 (B, T, D)
        # 注意：由于 pad 的存在，输出可能比 T 稍微大一点或小一点，取决于奇偶
        # 我们截取前 T 个以防万一
        trend = trend_trans[:, :, :T].permute(0, 2, 1)
        
        # 1.2 计算残差
        residual = x - trend
        
        # ================= Step 2: 双路处理 =================
        
        # --- Path A: 趋势项 (Trend) ---
        # 降采样：每 trend_stride 步取一个点
        # 假设 stride=5，则 T=60 变为 T=12
        trend_sampled = trend[:, ::self.trend_stride, :]
        
        # GRU Forward
        out_trend, _ = self.gru_trend(trend_sampled)
        
        # 取最后一个时间步 + 归一化 + 投影
        # out_trend[:, -1, :] -> (B, Hidden)
        feat_trend = self.ln_trend(out_trend[:, -1, :])
        pred_trend = self.head_trend(feat_trend)
        
        # --- Path B: 残差项 (Residual) ---
        # 保持全分辨率，捕捉每一个 Tick 的跳动
        out_res, _ = self.gru_res(residual)
        
        # 取最后一个时间步 + 归一化 + 投影
        feat_res = self.ln_res(out_res[:, -1, :])
        pred_res = self.head_res(feat_res)
        
        # ================= Step 3: 结果融合 =================
        
        # 最终预测 = 趋势预测 + 波动预测
        # 这种加法结构让模型具有很好的解释性
        return pred_trend + pred_res

def evaluate_model(model, test_files, config, scaler, col_config, device, threshold):
    model.eval()
    
    # 获取特征索引
    ap_indices = col_config['ap_indices']
    bp_indices = col_config['bp_indices']
    feature_cols = col_config['feature_cols']
    
    horizon_cols = [
        'LABEL_CAL_DQ_inst1_1', 
        'LABEL_CAL_DQ_inst1_3', 
        'LABEL_CAL_DQ_inst1_5', 
        'LABEL_CAL_DQ_inst1_10', 
        'LABEL_CAL_DQ_inst1_15', 
        'LABEL_CAL_DQ_inst1_30', 
        'LABEL_CAL_DQ_inst1_60'
    ]
    price_cols = ['ap1', 'bp1'] 
    target_label_col = 'label'

    # --- 1. 数据收集容器 (Collectors) ---
    # 使用列表存储每个批次/文件的数据，最后统一 concat
    all_data = {
        'preds': [],        # 预测值
        'labels': [],       # 真实 Label (用于计算分位数和 MSE)
        'ap1': [],          # 用于计算 BP 模式的成本
        'bp1': [],          # 用于计算 AP 模式的成本
        'is_buy': [],       # bool 标记: True代表BP模式(做多), False代表AP模式(做空)
    }
    # 为每个 horizon 准备容器
    all_horizon_prices = {h: [] for h in horizon_cols}

    # 统计计数器 (用于 logging)
    processed_files_count = 0

    for fpath in test_files:
        try:
            # --- 加载与预处理 ---
            req_cols = list(set(feature_cols + ['featSite'] + price_cols + horizon_cols + [target_label_col]))
            df = pd.read_parquet(fpath, columns=req_cols)
            
            # 特征缩放
            feats_raw = df[feature_cols].values
            feats_scaled = scaler.transform(feats_raw).astype(np.float32)
            
            # Sites 掩码
            sites_str = df['featSite'].astype(str).str.lower().values
            sites_code = np.zeros(len(df), dtype=np.int8)
            sites_code[sites_str == 'ap'] = 1
            sites_code[sites_str == 'bp'] = 2
            
            dummy_labels = torch.zeros(len(df)) 
            dilation = config['training'].get('dilation', 1)

            # --- 定义双边任务 (AP & BP) ---
            tasks = [('ap', ap_indices, 1), ('bp', bp_indices, 2)]

            for mode, indices, t_code in tasks:
                # 1. 构建 Dataset & DataLoader
                feats_pure = feats_scaled[:, indices]
                ds = SelectedIndicesDataset(
                    features=torch.from_numpy(feats_pure), 
                    labels=dummy_labels, 
                    sites=sites_code, 
                    target_code=t_code, 
                    seq_len=config['training']['seq_len'],
                    dilation=dilation
                )
                
                if len(ds) == 0: continue
                
                # 2. 推理 (Inference)
                loader = DataLoader(ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
                file_preds = []
                with torch.no_grad():
                    for x, _ in loader:
                        x = x.to(device)
                        pred = model(x).squeeze()
                        if pred.ndim == 0: pred = pred.unsqueeze(0)
                        file_preds.append(pred.cpu().numpy())
                
                if not file_preds: continue
                batch_preds = np.concatenate(file_preds)

                # 3. 提取对应的原始数据
                valid_indices = ds.valid_anchors
                curr_slice = df.iloc[valid_indices].reset_index(drop=True)
                
                # 4. 存入收集器
                all_data['preds'].append(batch_preds)
                
                # 存 Label
                if target_label_col in curr_slice.columns:
                    all_data['labels'].append(curr_slice[target_label_col].values)
                else:
                    # 如果没有label，填0 (防止concat出错，虽然eval通常都有label)
                    all_data['labels'].append(np.zeros_like(batch_preds))

                # 存价格信息
                all_data['ap1'].append(curr_slice['ap1'].values)
                all_data['bp1'].append(curr_slice['bp1'].values)
                
                # 存模式标记 (BP=True, AP=False)
                is_buy_mask = np.full(len(batch_preds), (mode == 'bp'), dtype=bool)
                all_data['is_buy'].append(is_buy_mask)
                
                # 存 Horizon 价格
                for h in horizon_cols:
                    if h in curr_slice.columns:
                        all_horizon_prices[h].append(curr_slice[h].values)
                    else:
                        all_horizon_prices[h].append(np.zeros_like(batch_preds))
            
            processed_files_count += 1

        except Exception as e:
            logging.error(f"Error evaluating {fpath}: {e}")

    if not all_data['preds']:
        logging.warning("No predictions generated.")
        return 0.0

    # --- 2. 全量拼接 (Concatenate) ---
    logging.info(f"Concatenating data from {processed_files_count} files...")
    
    global_preds = np.concatenate(all_data['preds'])
    global_labels = np.concatenate(all_data['labels'])
    global_ap1 = np.concatenate(all_data['ap1'])
    global_bp1 = np.concatenate(all_data['bp1'])
    global_is_buy = np.concatenate(all_data['is_buy']) # Bool array
    
    global_horizons = {}
    for h in horizon_cols:
        if all_data['preds']: # 只要有数据
            global_horizons[h] = np.concatenate(all_horizon_prices[h])
        else:
            global_horizons[h] = np.array([])

    # --- 3. 计算指标 (MSE / CCC) ---
    # MSE & R2
    mse = np.mean((global_preds - global_labels) ** 2)
    var_label = np.var(global_labels)
    r2 = 1 - (mse / var_label) if var_label > 1e-9 else 0.0
    
    # CCC
    mean_pred = np.mean(global_preds)
    mean_target = np.mean(global_labels)
    var_pred = np.var(global_preds)
    cov = np.mean((global_preds - mean_pred) * (global_labels - mean_target))
    numerator = 2 * cov
    denominator = var_pred + var_label + (mean_pred - mean_target)**2
    ccc_score = numerator / denominator if denominator > 1e-9 else 0.0

    # --- 4. 离线 PnL 计算 (Vectorized) ---
    
    # A. 确定阈值：Label 的 90 分位数
    # 注意：这里使用的是【全量 Labels】的分布
    label_90_pct = np.percentile(global_preds, 90)
    calculated_threshold = label_90_pct
    
    # B. 生成信号 Mask
    # 只有预测值 > Label 90分位数值 时才交易
    signal_mask = global_preds > calculated_threshold
    total_signals = np.sum(signal_mask)

    # 准备日志
    log_msg = (f"Global Eval | Samples: {len(global_preds)} | Signals: {total_signals} "
               f"(Thresh: {calculated_threshold:.4f}) | "
               f"MSE: {mse:.6f} | R2: {r2:.6f} | CCC: {ccc_score:.6f}")

    total_profit_max = -np.inf

    # C. 循环 Horizon 计算 PnL
    if total_signals > 0:
        # 预先筛选出有信号的子集，减少计算量
        # (可选优化，这里直接用 mask 索引也很快)
        
        # 提取有信号的行对应的价格和模式
        sig_ap1 = global_ap1[signal_mask]
        sig_bp1 = global_bp1[signal_mask]
        sig_is_buy = global_is_buy[signal_mask] # True for BP, False for AP

        sorted_horizons = sorted(horizon_cols, key=lambda x: int(x.split('_')[-1]))
        
        for h_col in sorted_horizons:
            sig_future = global_horizons[h_col][signal_mask]
            
            # 向量化计算 PnL
            # 逻辑：
            # PnL = (Future - Ask1) * is_buy + (Bid1 - Future) * (~is_buy)
            # 也就是：
            # BP模式 (Buy): Future - Ask1
            # AP模式 (Sell): Bid1 - Future
            
            pnl_bp = sig_future - sig_ap1
            pnl_ap = sig_bp1 - sig_future
            
            # 使用 np.where 根据模式选择对应的 PnL
            pnl_combined = np.where(sig_is_buy, pnl_bp, pnl_ap)
            
            total_pnl = np.sum(pnl_combined)
            avg_pnl = total_pnl / total_signals
            
            short_name = h_col.split('_')[-1] + "s"
            log_msg += f" | {short_name}: {avg_pnl:.5f}"
            
            if total_pnl > total_profit_max:
                total_profit_max = total_pnl
    else:
        log_msg += " | No Signals Triggered"
        total_profit_max = 0.0

    logging.info(log_msg)
    
    # --- 5. 打印预测值分布 (可选) ---
    p_dist = np.percentile(global_preds, [0, 25, 50, 75, 90, 100])
    logging.info(f"Pred Dist: Min={p_dist[0]:.3f}, 50%={p_dist[2]:.3f}, "
                 f"90%={p_dist[4]:.3f}, Max={p_dist[5]:.3f}")

    return total_profit_max


def train_pipeline(config, save_dir, device):
    side = config['training'].get('side', 'buy')
    mode = 'bp' if side == 'buy' else 'ap'
    target_code = 1 if mode == 'ap' else 2
    
    logging.info(f"Starting Joint Training (AP + BP)...")
    data_dir = config['paths']['processed_data_dir']
    all_files = sorted(glob.glob(os.path.join(data_dir, "day_*.npz")), key=get_date_from_filename)
    test_file_count = 14
    train_files = all_files[:-test_file_count]
    # train_files = all_files[:6]
    # print(train_files)
    raw_data_dir = config['paths'].get('raw_data_dir', './merged_features_final')
    all_parquet = sorted(glob.glob(os.path.join(raw_data_dir, "*_features.parquet")), key=get_date_from_filename)
    test_files_parquet = all_parquet[-test_file_count:] 
    # test_files_parquet = all_parquet[-2:]
    col_indices_path = os.path.join(data_dir, 'col_indices.pkl')
    col_config = joblib.load(col_indices_path)
    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    logging.info(f"Starting Joint Training (AP + BP)...")
    ap_indices = torch.tensor(col_config['ap_indices'], dtype=torch.long)
    bp_indices = torch.tensor(col_config['bp_indices'], dtype=torch.long)
    
    # 检查维度一致性 (可选，防止 AP/BP 特征数量不一致报错)
    assert len(ap_indices) == len(bp_indices), "AP and BP feature counts must match for joint training!"
    if config['modeltype']=='GRU':
        model = GRUPredictor(

            input_dim=config['model']['input_dim'],

            hidden_dim=config['model']['hidden_dim'],

            num_layers=config['model']['num_layers'],

            dropout=config['model']['dropout']

        ).to(device)

    elif config['modeltype']=='tsmixer':

        seq_len = config['training']['seq_len']

        levels = config['model'].get('downsample_levels', 2)

        model = TimeMixerPredictor(

            input_dim=config['model']['input_dim'],

            seq_len=seq_len,

            hidden_dim=config['model']['hidden_dim'],

            num_layers=config['model']['num_layers'],

            dropout=config['model']['dropout'],

            downsample_levels=levels

        ).to(device)

    elif config['modeltype']=='LSTM':

        model = NativeLSTMPredictor(

            input_dim=config['model']['input_dim'],

            hidden_dim=config['model']['hidden_dim'],

            num_layers=config['model']['num_layers'],

            dropout=config['model']['dropout']

        ).to(device)

    elif config['modeltype']=='sLSTM':

        model = sLSTMPredictor(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout']
        ).to(device)
    elif config['modeltype']=='Multi_GRU':
        model = MultiRateGRU(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            scales=config['model']['scale'],
            dropout=config['model']['dropout']
        ).to(device)
    elif config['modeltype']=='BiGRU':
        model = DecompositionGRU(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            trend_window=5,
            trend_stride=5
        ).to(device)
    elif config['modeltype']=='Multi_SAGRU':
        num_layers = config['model'].get("num_layers",1)
        model = MultiScaleSAGRU(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            scales=config['model']['scale'],
            dropout=config['model']['dropout'],
            numlayers=num_layers,
            num_heads=config['model']['num_heads']
        ).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)
    # criterion = nn.MSELoss()
    # loss_args = config['training'].get('loss_params', {})
    # print(loss_args)
    # criterion = TradingOpportunityLoss(
    #     w_miss=loss_args.get('w_miss', 10.0),
    #     w_noise=loss_args.get('w_noise', 1.0),
    #     w_fp=loss_args.get('w_fp', 5.0),
    #     threshold = loss_args.get('threshold', 0.5)
    # )
    criterion = CCCLoss()
    best_model_path = os.path.join(save_dir, f"model_best_joint.pth")
    best_test_profit = -np.inf

    # --- Training Loop ---
    logging.info(f"Training for {config['training']['epochs']} epochs...")
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        total_batches = 0
        random.shuffle(train_files)
        
        for f_idx, fpath in enumerate(train_files):
            raw_feats, labels, sites = load_raw_day_data(fpath)
            
            if raw_feats is None or len(raw_feats) == 0: continue
            
            dilation = config['training'].get('dilation', 1)
            seq_len = config['training']['seq_len']
            
            # 2. 构建 AP 数据集 (target_code=1)
            # 选取 ap_indices 对应的列
            feats_ap = raw_feats[:, ap_indices]
            ds_ap = SelectedIndicesDataset(
                feats_ap, labels, sites, 
                target_code=1, 
                seq_len=seq_len, 
                dilation=dilation
            )
            
            # 3. 构建 BP 数据集 (target_code=2)
            # 选取 bp_indices 对应的列
            feats_bp = raw_feats[:, bp_indices]
            ds_bp = SelectedIndicesDataset(
                feats_bp, labels, sites, 
                target_code=2, 
                seq_len=seq_len, 
                dilation=dilation
            )
            
            # 4. 合并 AP 和 BP 数据集
            # 如果某天某一边的样本数为0，ConcatDataset 也能正常处理
            datasets_to_combine = []
            if len(ds_ap) > 0: datasets_to_combine.append(ds_ap)
            if len(ds_bp) > 0: datasets_to_combine.append(ds_bp)
            
            if not datasets_to_combine:
                continue
                
            combined_dataset = ConcatDataset(datasets_to_combine)
            
            # 5. 放入 DataLoader (shuffle=True 会将 AP 和 BP 样本彻底打乱)
            train_loader = DataLoader(
                combined_dataset, 
                batch_size=config['training']['batch_size'], 
                shuffle=True, 
                num_workers=0
            )
            
            day_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_x).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                # print(loss)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                day_loss += loss.item()
                total_batches += 1
            if (f_idx + 1) % 20 == 0:
                sys.stdout.write(f"\r[Epoch {epoch+1}] Day {f_idx+1}/{len(train_files)} | Loss: {day_loss/len(train_loader):.4f}")
                sys.stdout.flush()
            epoch_loss+=day_loss
        avg_epoch_loss = epoch_loss / total_batches if total_batches > 0 else 0
        print(f"\n>> Epoch {epoch+1} Train Loss: {avg_epoch_loss:.6f}")
        

        # --- Evaluation ---
        logging.info(f"Evaluating Epoch {epoch+1}...")
        current_profit = evaluate_model(model, test_files_parquet, config, scaler, col_config, device,threshold=config['training']['loss_params']['threshold'])
        
        logging.info(f">> Epoch {epoch+1} Test Avg Profit: {current_profit:.10f}")
        
        if current_profit > best_test_profit:
            best_test_profit = current_profit
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"*** New Best Model Saved ***")

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