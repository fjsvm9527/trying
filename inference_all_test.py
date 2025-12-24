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

class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(GRUPredictor, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class MixerBlock(nn.Module):
    def __init__(self, seq_len, num_features, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(num_features)
        self.time_mlp = nn.Sequential(nn.Linear(seq_len, seq_len), nn.GELU(), nn.Dropout(dropout), nn.Linear(seq_len, seq_len), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(num_features)
        self.channel_mlp = nn.Sequential(nn.Linear(num_features, num_features * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(num_features * 4, num_features), nn.Dropout(dropout))
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

class NativeLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        last_step = self.norm(last_step)
        return self.head(last_step)

# --- sLSTM 相关 ---
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

class sLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([sLSTMBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x[:, -1, :])
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
    
class SAGRU(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 hidden_dim=64, 
                 num_layers=2, 
                 num_heads=4, 
                 dropout=0.1):
        """
        Args:
            feature_dim (int): 输入特征维度
            hidden_dim (int): GRU 隐层维度
            num_layers (int): GRU 层数
            num_heads (int): Attention 头数
            dropout (float): Dropout 比率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. GRU Backbone
        # batch_first=True: 输入形状为 (Batch, Seq_Len, Feature)
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 2. Self-Attention Layer
        # 使用 PyTorch 原生的 MultiheadAttention，效率更高
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True # 关键参数，确保输入是 (Batch, Seq, Feature)
        )
        
        # 3. LayerNorm & Residual
        # 用于稳定 Attention 的输出
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 4. Output Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        query = gru_out[:, -1:, :] 
        key = gru_out
        value = gru_out
        attn_output, attn_weights = self.attention(query, key, value)
        context_vector = query + attn_output
        context_vector = self.norm(context_vector)
        feature_final = context_vector.squeeze(1)
        prediction = self.head(feature_final)      
        return prediction

class InceptionBlock(nn.Module):
    """
    并行使用多个不同尺寸的卷积核，提取多尺度局部特征
    """
    def __init__(self, in_dim, out_dim, kernels=[1, 3, 5]):
        super().__init__()
        self.branches = nn.ModuleList()
        
        # 确保输出总维度等于 out_dim，分配给每个分支
        branch_dim = out_dim // len(kernels)
        self.branch_dim = branch_dim
        self.project_dim = branch_dim * len(kernels) # 实际总输出维度
        
        for k in kernels:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_dim, branch_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(branch_dim),
                nn.ReLU()
            ))
            
    def forward(self, x):
        # x: (Batch, In_Dim, Seq_Len)
        outs = [branch(x) for branch in self.branches]
        # 在通道维度拼接: (Batch, Total_Out_Dim, Seq_Len)
        return torch.cat(outs, dim=1)

class InceptionGRU(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim=64, 
                 num_layers=2, 
                 dropout=0.1):
        super().__init__()
        
        # 1. Inception CNN 提取多尺度局部特征
        # 它会自动创建 kernel=1, 3, 5 的三个分支
        self.inception = InceptionBlock(input_dim, hidden_dim, kernels=[1, 3, 5])
        
        # Inception 输出的实际维度 (可能因为取整略小于 hidden_dim)
        inception_out_dim = self.inception.project_dim
        
        # 2. GRU 建模全局依赖
        self.gru = nn.GRU(
            input_size=inception_out_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. 简单的 Attention 融合 (可选，增强最后一步的表现)
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        x_cnn = x.permute(0, 2, 1)
        
        # --- A. Inception 多尺度特征提取 ---
        # features: (B, Hidden, T)
        features = self.inception(x_cnn)
        
        # permute back: (B, T, Hidden)
        gru_input = features.permute(0, 2, 1)
        
        # --- B. GRU ---
        # gru_out: (B, T, Hidden)
        gru_out, _ = self.gru(gru_input)
        
        # --- C. Attention Pooling (或是直接取最后一步) ---
        # 这里演示 Attention Pooling：加权平均所有时间步
        # weights: (B, T, 1)
        weights = F.softmax(self.attention(gru_out), dim=1)
        # context: (B, Hidden)
        context = torch.sum(gru_out * weights, dim=1)
        
        # --- D. 预测 ---
        return self.head(context)

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
                    num_layers=1,
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

class UnifiedModelInference:
    def __init__(self, config_path, model_path, device='cuda'):
        self.config = self._load_config(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # 自动获取数据目录
        self.data_dir = self.config['paths'].get('processed_data_dir', './processed_data')
        
        col_indices_path = os.path.join(self.data_dir, 'col_indices.pkl')
        scaler_path = os.path.join(self.data_dir, 'scaler.pkl')
        
        if not os.path.exists(col_indices_path) or not os.path.exists(scaler_path):
            # 尝试当前目录查找，作为 fallback
            col_indices_path = 'col_indices.pkl'
            scaler_path = 'scaler.pkl'
            if not os.path.exists(col_indices_path):
                raise FileNotFoundError(f"Missing preprocessing files. Check 'col_indices.pkl' and 'scaler.pkl'")

        self.col_config = joblib.load(col_indices_path)
        self.scaler = joblib.load(scaler_path)
        
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
        model_type = self.config['modeltype']
        input_dim = self.config['model']['input_dim']
        hidden_dim = self.config['model']['hidden_dim']
        num_layers = self.config['model']['num_layers']
        dropout = self.config['model']['dropout']
        
        # 根据类型实例化模型
        if model_type == 'GRU':
            model = GRUPredictor(input_dim, hidden_dim, num_layers, dropout)
        elif model_type == 'tsmixer':
            seq_len = self.config['training']['seq_len']
            levels = self.config['model'].get('downsample_levels', 2)
            model = TimeMixerPredictor(input_dim, seq_len, hidden_dim, num_layers, dropout, levels)
        elif model_type == 'LSTM':
            model = NativeLSTMPredictor(input_dim, hidden_dim, num_layers, dropout)
        elif model_type == 'sLSTM':
            num_heads = self.config['model'].get('num_heads', 4)
            model = sLSTMPredictor(input_dim, hidden_dim, num_layers, num_heads, dropout)
        elif model_type=='Multi_GRU':
            model = MultiRateGRU(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                scales=self.config['model']['scale'],
                dropout=dropout
            )
        elif model_type=='SAGRU':
            num_heads = self.config['model'].get('num_heads', 4)
            model = SAGRU(
                feature_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout
            )
        elif model_type=='Inception_GRU':
            model = InceptionGRU(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif model_type=='Multi_SAGRU':
            model = MultiScaleSAGRU(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                scales=self.config['model']['scale'],
                dropout=self.config['model']['dropout'],
                num_heads=self.config['model']['num_heads']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.to(self.device)
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _predict_single_side(self, feats_tensor, sites_code, target_code):
        """
        辅助函数：对单边数据进行推理
        注意：这里直接使用 self.model，因为现在只有一个模型
        """
        seq_len = self.config['training']['seq_len']
        dilation = self.config['training'].get('dilation', 1)
        batch_size = self.config['training']['batch_size']
        dummy_labels = torch.zeros(len(sites_code))
        
        # 构建 Dataset (会自动根据 target_code 筛选出 AP 或 BP 的有效行)
        dataset = SelectedIndicesDataset(
            features=feats_tensor, labels=dummy_labels, sites=sites_code,
            target_code=target_code, seq_len=seq_len, dilation=dilation
        )
        
        if len(dataset) == 0:
            return np.array([]), np.array([])
            
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        preds_list = []
        
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                out = self.model(x).squeeze()
                if out.ndim == 0: out = out.unsqueeze(0)
                preds_list.append(out.cpu().numpy())
                
        all_preds = np.concatenate(preds_list) if preds_list else np.array([])
        # 返回：原始数据的行索引 (anchors) 和 对应的预测值
        return dataset.valid_anchors, all_preds

    def predict_csv(self, csv_path):
        print(f"Processing {csv_path}...")
        try:
            if csv_path.endswith('.parquet'):
                df = pd.read_parquet(csv_path)
            else:
                df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

        if 'featSite' not in df.columns:
            raise ValueError("Input CSV must contain 'featSite' column.")

        # 1. 准备特征矩阵
        try:
            feats_raw = df[self.feature_cols].values
        except KeyError as e:
            print(f"Missing columns: {e}")
            return None
            
        # 全量缩放
        feats_scaled = self.scaler.transform(feats_raw).astype(np.float32)
        N = len(df)
        
        # 2. 准备 Site 编码
        sites_str = df['featSite'].astype(str).str.lower().values
        sites_code = np.zeros(N, dtype=np.int8)
        sites_code[sites_str == 'ap'] = 1
        sites_code[sites_str == 'bp'] = 2
        
        # 初始化结果数组 (NaN 填充)
        final_preds = np.full(N, np.nan, dtype=np.float32)
        
        # --- AP 推理 (使用 self.model) ---
        # 提取 AP 对应的特征列
        feats_ap_pure = feats_scaled[:, self.ap_indices]
        # target_code=1 代表 AP，_predict_single_side 会筛选出 sites_code==1 的行
        ap_row_indices, ap_preds = self._predict_single_side(
            torch.from_numpy(feats_ap_pure), sites_code, target_code=1
        )
        if len(ap_row_indices) > 0:
            final_preds[ap_row_indices] = ap_preds

        # --- BP 推理 (使用同一个 self.model) ---
        # 提取 BP 对应的特征列
        feats_bp_pure = feats_scaled[:, self.bp_indices]
        # target_code=2 代表 BP
        bp_row_indices, bp_preds = self._predict_single_side(
            torch.from_numpy(feats_bp_pure), sites_code, target_code=2
        )
        if len(bp_row_indices) > 0:
            final_preds[bp_row_indices] = bp_preds

        # 3. 结果写回
        df['prediction'] = final_preds
        
        if 'label' in df.columns:
            valid_mask = ~np.isnan(final_preds)
            if valid_mask.sum() > 0:
                ic = df.loc[valid_mask, 'prediction'].corr(df.loc[valid_mask, 'label'])
                print(f"IC: {ic:.4f}")
            
        valid_count = (~np.isnan(final_preds)).sum()
        print(f"Done. {valid_count}/{N} predicted.")
        return df

# ==========================================
# 3. 主程序入口
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
if __name__ == "__main__":
    # 1. 定义命令行参数
    parser = argparse.ArgumentParser(description="Unified Model Inference Pipeline")
    
    # 必需参数
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config JSON')
    # 修改点：不再需要分别的 ap/bp model，只需要一个 model_path
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained unified model (.pth)')
    parser.add_argument('--modelname', type=str, required=True, help='Name identifier for the output file')
    
    # 可选参数
    parser.add_argument('--input_dir', type=str, default='./merged_features_test_withlabel', 
                        help='Directory containing input feature files')
    parser.add_argument('--device', type=str, default='cuda:0', help='Computing device')
    
    args = parser.parse_args()
    feature_file_map = build_date_file_map(args.input_dir, suffix="_features.parquet")
    # market_file_map = build_date_file_map(args.market_dir, suffix=".csv")
    # 2. 打印运行信息
    print("-" * 30)
    print(f"Task: {args.modelname}")
    print(f"Config: {args.config}")
    print(f"Unified Model: {args.model_path}")
    print(f"Input Dir: {args.input_dir}")
    print("-" * 30)

    # 3. 初始化推理引擎 (使用 UnifiedModelInference)
    engine = UnifiedModelInference(args.config, args.model_path, device=args.device)
    
    # 4. 循环处理文件
    target_dates = ['20250901', '20250902', '20250903', '20250904', '20250908', '20250909', '20250910', '20250911', '20250912', '20250915', '20250916', '20250917', '20250918', '20250919', '20250922', '20250923', '20250924', '20250925', '20250926', '20250929', '20250930', '20251009', '20251010', '20251013', '20251014', '20251015', '20251016', '20251017']
    # target_dates = ['20250828', '20250829']
    # target_dates = ['20250812','20250813','20250814','20250815','20250818','20250819','20250820','20250821','20250822','20250825','20250826','20250827','20250828','20250829']
    df_all_list = []
    
    for date in target_dates:
        # 支持 parquet 或 csv
        input_file = feature_file_map.get(date)
        
        if not os.path.exists(input_file):
            print(f"Warning: File not found {input_file}, skipping...")
            continue
            
        df_result = engine.predict_csv(input_file)
        
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
        thresholds = np.arange(0.0, 3.0, 0.01)
        mask_bp = df_all['featSite'].astype(str).str.lower() == 'bp'
        mask_ap = df_all['featSite'].astype(str).str.lower() == 'ap'
        df_all.loc[mask_bp,'theoprice'] = df_all.loc[mask_bp,'refPrice']+df_all.loc[mask_bp,'prediction']
        df_all.loc[mask_ap,'theoprice'] = df_all.loc[mask_ap,'refPrice']-df_all.loc[mask_ap,'prediction']
        df_all.loc[mask_ap,'future_price'] = df_all.loc[mask_ap,'refPrice'] - df_all.loc[mask_ap,'label']
        df_all.loc[mask_bp,'future_price'] = df_all.loc[mask_bp,'refPrice'] + df_all.loc[mask_bp,'label']
        results = []
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