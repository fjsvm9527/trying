import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal, kl_divergence
from tqdm import tqdm
import os
import random
from sklearn.preprocessing import StandardScaler
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import joblib
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def prepare_tensor_with_context(df, feature_cols, context_cols, label_col, seq_len=30):
    """
    生成张量数据：
    - X (Feature): 3D 张量 [Batch, Seq_Len, Feat_Dim] (时序滑动窗口)
    - C (Context): 2D 张量 [Batch, Context_Dim] (仅取窗口末端的静态状态)
    """
    required_cols = ['trade_day', 'timestamp'] + feature_cols + context_cols + [label_col]
    if not all(c in df.columns for c in required_cols):
        raise ValueError("Missing columns in DataFrame")

    print(">>> [Data] Sorting and Grouping...")
    df = df.sort_values(['trade_day', 'timestamp'])
    groups = df.groupby('trade_day')
    
    X_list, C_list, y_list = [], [], []
    
    print(f">>> [Data] Generating Tensors (Seq_Len={seq_len})...")
    for date, group in tqdm(groups, desc="Processing Days"):
        feat_vals = group[feature_cols].values.astype(np.float32) # (N, F)
        ctx_vals = group[context_cols].fillna(0.0).values.astype(np.float32)  # (N, C)
        label_vals = group[label_col].values.astype(np.float32)   # (N,)
        pad_width_feat = ((seq_len - 1, 0), (0, 0))
        feat_padded = np.pad(feat_vals, pad_width_feat, mode='edge')
        
        feat_windows_raw = np.lib.stride_tricks.sliding_window_view(feat_padded, window_shape=seq_len, axis=0)
        feat_windows = feat_windows_raw.transpose(0, 2, 1) # 最终形状: (N, Seq_Len, Feat_Dim)
        context_2d = ctx_vals 
        
        # 4. 收集
        X_list.append(feat_windows)
        C_list.append(context_2d)
        y_list.append(label_vals)
        
    # 5. 合并
    if not X_list:
        raise ValueError("No data processed!")
        
    X_all = np.concatenate(X_list, axis=0)
    C_all = np.concatenate(C_list, axis=0)
    
    print(f"Dataset Shape | X: {X_all.shape} (3D), Context: {C_all.shape} (2D)")
    
    # 返回 Tensor
    return torch.from_numpy(X_all), torch.from_numpy(C_all)

# =========================================================================
#  2. 模型架构: Self-Supervised VAE (Unsupervised)
# =========================================================================
# ==========================================
class LSTMBackbone(nn.Module):
    def __init__(self, num_factors, hidden_dim, num_layers=1, dropout=0.0):
        super(LSTMBackbone, self).__init__()
        self.lstm = nn.LSTM(num_factors, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
    def forward(self, x):
        # x: [Batch, Seq, Feat]
        _, (h_n, _) = self.lstm(x)
        h_n_last_layer = h_n[-1] # [Batch, Hidden]
        return h_n_last_layer

# ==========================================
# 2. GRU Backbone
# ==========================================
class GRUBackbone(nn.Module):
    def __init__(self, num_factors, hidden_dim, num_layers=1, dropout=0.0):
        super(GRUBackbone, self).__init__()
        self.gru = nn.GRU(num_factors, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
    def forward(self, x):
        _, h_n = self.gru(x)
        h_n_last_layer = h_n[-1] # [Batch, Hidden]
        return h_n_last_layer

# ==========================================
# 3. TSMixer Backbone
# ==========================================
class TSMixerBlock(nn.Module):
    def __init__(self, num_factors, seq_len, ff_dim):
        super(TSMixerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(num_factors)
        self.time_mlp = nn.Sequential(nn.Linear(seq_len, ff_dim), nn.ReLU(), nn.Linear(ff_dim, seq_len))
        self.norm2 = nn.LayerNorm(num_factors)
        self.feature_mlp = nn.Sequential(nn.Linear(num_factors, ff_dim), nn.ReLU(), nn.Linear(ff_dim, num_factors))
        
    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.time_mlp(x)
        x = x.transpose(1, 2)
        x = x + res
        
        res = x
        x = self.norm2(x)
        x = self.feature_mlp(x)
        x = x + res
        return x

class TSMixerBackbone(nn.Module):
    def __init__(self, num_factors, seq_len, hidden_dim, ff_dim=64, num_blocks=2):
        super(TSMixerBackbone, self).__init__()
        self.blocks = nn.ModuleList([TSMixerBlock(num_factors, seq_len, ff_dim) for _ in range(num_blocks)])
        # 最后的映射层，确保输出维度为 hidden_dim
        self.final_fc = nn.Linear(num_factors, hidden_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x_pooled = x.mean(dim=1) # Global Average Pooling
        h = F.relu(self.final_fc(x_pooled))
        return h

# ==========================================
# 4. DLinear Backbone
# ==========================================
class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        
    def forward(self, x):
        # Padding on the left
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1, 1)
        x = torch.cat([front, x], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class DLinearBackbone(nn.Module):
    def __init__(self, seq_len, num_factors, hidden_dim, kernel_size=25):
        super(DLinearBackbone, self).__init__()
        # 修正：kernel_size 不能大于 seq_len
        if kernel_size > seq_len:
            kernel_size = seq_len // 2 + 1
            
        self.moving_avg = MovingAvg(kernel_size)
        
        input_dim = seq_len * num_factors
        # 这里的中间层维度可以写死或配置，这里沿用你参考代码的逻辑
        self.seasonal_fc = nn.Linear(input_dim, 128)
        self.trend_fc = nn.Linear(input_dim, 128)
        
        # 最终输出层，映射到统一的 hidden_dim
        self.combined_fc = nn.Linear(256, hidden_dim)

    def forward(self, x):
        trend_part = self.moving_avg(x)
        seasonal_part = x - trend_part
        
        seasonal_flat = seasonal_part.reshape(seasonal_part.size(0), -1)
        trend_flat = trend_part.reshape(trend_part.size(0), -1)
        
        h_seasonal = F.relu(self.seasonal_fc(seasonal_flat))
        h_trend = F.relu(self.trend_fc(trend_flat))
        
        h_combined = torch.cat((h_seasonal, h_trend), dim=1)
        h = F.relu(self.combined_fc(h_combined))
        return h

class ContextGatedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, context_size, gating_mode=(True, True, True)):
        super(ContextGatedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gating_mode = gating_mode
        
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.num_controlled_gates = sum(gating_mode)
        
        if self.num_controlled_gates > 0:
            self.context_gate = nn.Sequential(
                nn.Linear(context_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_controlled_gates * hidden_size), 
                nn.Sigmoid() 
            )
        else:
            self.context_gate = None
        
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        with torch.no_grad():
            self.weight_ih.bias.data[self.hidden_size : 2 * self.hidden_size].fill_(1.0)
            self.weight_hh.bias.data[self.hidden_size : 2 * self.hidden_size].fill_(1.0)
        if self.context_gate is not None:
            self.context_gate[-2].bias.data.fill_(1.0)

    def forward(self, x, context, h_c):
        """
        x:       [Batch, Input_Dim]
        context: [Batch, Context_Dim]  <-- 这是一个 2D 张量
        h_c:     tuple([Batch, Hidden], [Batch, Hidden])
        """
        h_prev, c_prev = h_c
        
        # --- 标准 LSTM 计算 ---
        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
        
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        
        if self.num_controlled_gates > 0:
            ctx_out = self.context_gate(context) 
            
            ctx_chunks = ctx_out.chunk(self.num_controlled_gates, dim=1)
            chunk_idx = 0
            if self.gating_mode[0]:
                i_gate = i_gate * ctx_chunks[chunk_idx]
                chunk_idx += 1
            
            if self.gating_mode[1]:
                f_gate = f_gate * ctx_chunks[chunk_idx]
                chunk_idx += 1
                
            if self.gating_mode[2]:
                o_gate = o_gate * ctx_chunks[chunk_idx]
                chunk_idx += 1
        c_next = (f_gate * c_prev) + (i_gate * g_gate)
        h_next = o_gate * torch.tanh(c_next)
        
        return h_next, c_next

class ContextLSTMBackbone(nn.Module):
    """
    Backbone 封装：循环处理时间序列，但 Context 是静态的
    """
    def __init__(self, num_factors, hidden_dim, context_dim, num_layers=1, dropout=0.0, 
                 gating_mode=(True, True, True)):
        super(ContextLSTMBackbone, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 这里仅演示单层，多层需要 ModuleList
        self.cell = ContextGatedLSTMCell(num_factors, hidden_dim, context_dim, gating_mode)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context):
        """
        x:       [Batch, Seq_Len, Input_Dim]
        context: [Batch, Context_Dim]  <-- 静态 2D Context，无需时间步维度
        """
        batch_size, seq_len, _ = x.size()
        
        # 初始化状态
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            h, c = self.cell(x_t, context, (h, c))
            
        return h

class SelfSupervisedVAE(nn.Module):
    def __init__(self, encoder_config, num_factors, seq_len, hidden_dim, latent_dim=1, context_dim=0):
        super(SelfSupervisedVAE, self).__init__()
        
        model_type = encoder_config.get('type', 'context_lstm').lower()
        print(f"[Model] Initializing VAE with Encoder: {model_type.upper()} | Context Dim: {context_dim}")
        gating_mode = encoder_config.get('mode', (False,False,False))
        # 动态实例化 Encoder
        if model_type == 'context_lstm':
            if context_dim == 0:
                raise ValueError("Using ContextLSTM but context_dim is 0!")
            self.backbone = ContextLSTMBackbone(
                num_factors=num_factors,
                hidden_dim=hidden_dim,
                context_dim=context_dim,
                num_layers=encoder_config.get('num_layers', 1),
                dropout=encoder_config.get('dropout', 0.0),
                gating_mode=gating_mode
            )
        elif model_type=='lstm':
            raise ValueError("此脚本专用于演示 ContextLSTM，请在 Config 中设置 type='context_lstm'")

        # VAE 投影层
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_factors)
        )

    def reparameterize(self, mu, log_var):
        # if self.training:
        #     std = torch.exp(0.5 * log_var)
        #     eps = torch.randn_like(std)
        #     return mu + eps * std
        # else:
        #     return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, context=None):
        # 1. Encoder 提取特征
        if isinstance(self.backbone, ContextLSTMBackbone):
            if context is None:
                raise ValueError("ContextLSTM requires context input!")
            h = self.backbone(x, context)
        else:
            h = self.backbone(x)
        
        # 2. 映射分布
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        # 3. 采样
        z = self.reparameterize(mu, log_var)
        
        # 4. 解码重构
        recon_x = self.decoder_net(z)
        
        return recon_x, mu, log_var

    def run_model(self, x, context, gamma=0.01):
        recon_x, mu, log_var = self.forward(x, context)
        
        # Target: 重构序列的最后一个时间步 (Last Step Reconstruction)
        target = x[:, -1, :] 
        
        recon_loss = F.mse_loss(recon_x, target)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / x.size(0)
        
        total_loss = recon_loss + gamma * kl_loss
        return total_loss, recon_loss, kl_loss

    def extract_latent(self, x, context):
        with torch.no_grad():
            h = self.backbone(x, context)
            mu = self.fc_mu(h)
        return mu

# =========================================================================
#  3. 辅助函数
# =========================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f: return json.load(f)
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 63,
    'max_depth': 6,
    'feature_fraction': 0.3,
    'bagging_fraction': 0.8, 
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'n_jobs': 4,
    'verbose': -1,
    'seed': 42
}
def mixup_dataframe(df, feature_cols, label_col, alpha=0.2, augment_ratio=1.0, seed=42):
    """
    对 DataFrame 进行 Mixup 增强
    Args:
        alpha: Beta分布参数 (推荐 0.1~0.4)
        augment_ratio: 增强数据量比例 (1.0 表示增加一倍数据)
    """
    if alpha <= 0 or augment_ratio <= 0:
        return df

    print(f">>> 执行 Mixup 增强 (alpha={alpha}, ratio={augment_ratio})...")
    np.random.seed(seed)
    
    # 提取数据矩阵
    X = df[feature_cols].values
    y = df[label_col].values
    n_samples = len(df)
    n_augment = int(n_samples * augment_ratio)
    
    # 随机索引
    idx_i = np.random.choice(n_samples, n_augment, replace=True) # 样本 A
    idx_j = np.random.permutation(idx_i)                         # 样本 B
    
    # 生成 Lambda (Beta 分布)
    lam = np.random.beta(alpha, alpha, size=n_augment)
    lam_x = lam.reshape(-1, 1) # 广播用于特征矩阵
    print(lam_x)
    # 线性插值
    X_new = lam_x * X[idx_i] + (1 - lam_x) * X[idx_j]
    y_new = lam * y[idx_i] + (1 - lam) * y[idx_j]
    
    # 构建新 DataFrame
    df_new = pd.DataFrame(X_new, columns=feature_cols)
    df_new[label_col] = y_new
    
    # 拼接 (Original + Mixup)
    df_aug = pd.concat([df[feature_cols + [label_col]], df_new], axis=0).reset_index(drop=True)
    
    print(f"    原始样本: {n_samples} -> 增强后: {len(df_aug)}")
    return df_aug

def retrain_with_selected_features(train_df, valid_df, selected_features, params=LGB_PARAMS, label_col='prj2_1_label', mixup_alpha=0.2,name = "context"):
    """
    仅使用筛选后的 top_factors 进行重训并评估 (集成了 Mixup)
    """
    print(f"========================================")
    print(f"开始重训 (Refitting) - 使用特征数: {len(selected_features)}")
    print(f"========================================")
    
    # --- [修改点]：在此处对训练集进行 Mixup 增强 ---
    # 仅增强 Train，绝不增强 Valid
    if mixup_alpha > 0:
        train_df_final = mixup_dataframe(
            train_df, 
            selected_features, 
            label_col, 
            alpha=mixup_alpha, 
            augment_ratio=0.0 # 默认增加一倍数据
        )
    else:
        train_df_final = train_df

    # 1. 构建精简版 Dataset
    # 注意：这里使用 train_df_final (可能是增强过的)
    train_data = lgb.Dataset(
        train_df_final[selected_features], 
        label=train_df_final[label_col], 
        feature_name=selected_features
    )
    
    # 验证集保持原样 valid_df
    valid_data = lgb.Dataset(
        valid_df[selected_features], 
        label=valid_df[label_col], 
        feature_name=selected_features,
        reference=train_data 
    )
    
    # 2. 训练模型 (参数保持不变)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=20)
        ]
    )
    
    # 3. 评估效果 (保持原代码不变)
    print("\n>>> 正在评估重训模型性能...")
    
    # 预测验证集
    valid_preds = model.predict(valid_df[selected_features], num_iteration=model.best_iteration)
    # valid_preds = model.predict(valid_df[selected_features])
    valid_df['pred'] = valid_preds
    thresholds = np.arange(0.0, 10.0, 0.01)
    max_dq = -np.inf
    results = []
    
    for threshold in thresholds:
            buy_mask = valid_df['pred'] > threshold
            sell_mask = valid_df['pred'] <  -1*threshold
            pnl_buy_raw = valid_df.loc[buy_mask,'prj2_1_label'] - (0.5/10000)*valid_df.loc[buy_mask,'LABEL_CAL_DQ_inst1_60']
            pnl_sell_raw = -1*valid_df.loc[sell_mask,'prj2_1_label'] - (0.5/10000)*valid_df.loc[sell_mask,'LABEL_CAL_DQ_inst1_60']
            DQ_buy = np.sum(pnl_buy_raw)
            DQ_sell = np.sum(pnl_sell_raw)
            DQ = DQ_buy + DQ_sell
            count = len(pnl_buy_raw) + len(pnl_sell_raw)
            if count == 0:
                continue     
            final_DQ = 15 * DQ 
            if final_DQ > max_dq:
                max_dq = final_DQ
            abs_move = np.abs(pnl_buy_raw).sum() + np.abs(pnl_sell_raw).sum()
            dq_neg = (abs_move - DQ) / 2.0
            dq_pos = DQ + dq_neg
            dqr = dq_pos / dq_neg if dq_neg != 0 else 0
            results.append({
                'Threshold': threshold, 
                'DQ': final_DQ, 
                'DQR': dqr, 
                'Count': len(pnl_buy_raw) + len(pnl_sell_raw)
            })
    df_res = pd.DataFrame(results)
    df_res = df_res[(df_res['Count']>50)&(df_res['DQ']>0)]
    df_res['AVG_DQ'] = df_res['DQ'] / df_res['Count']
    if not df_res.empty:
            sns.set_theme(style="whitegrid")
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(df_res['Threshold'], df_res['DQ'], color='#2a9d8f', linewidth=2, label='DQ')
            ax2 = ax1.twinx()
            ax2.plot(df_res['Threshold'], df_res['DQR'], color='#e76f51', linestyle='--', linewidth=2, label='DQR')
            plt.title(f"Max DQ Analysis (Mixup Alpha={mixup_alpha}):")
            plt.savefig(f"/home/zyyuan/project2/pictures/{name}.jpg")
            
    # 计算 IC
    ic, p_value = pearsonr(valid_preds, valid_df[label_col].values)
    
    # 计算 RMSE
    mse = np.mean((valid_preds - valid_df[label_col].values) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"----------------------------------------")
    print(f"Refit Results (Mixup Alpha={mixup_alpha}):")
    print(f"Feature Count : {len(selected_features)}")
    print(f"Validation IC : {ic:.6f}")
    print(f"Validation RMSE: {rmse:.6f}")
    print(f"----------------------------------------")
    
    return model, {'ic': ic, 'rmse': rmse},df_res

# =========================================================================
#  4. 主程序
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    # 强制覆盖 latent_dim 为 1
    parser.add_argument('--latent_dim', type=int, default=64, help='压缩到几维特征')
    args_cmd = parser.parse_args()
    
    cfg = load_config(args_cmd.config)
    
    # 路径 & 参数
    train_path = cfg['paths']['train_data']
    valid_path = cfg['paths']['valid_data']
    label_col = cfg['data']['label_col']
    seq_len = cfg['data']['seq_len']
    batch_size = cfg['training']['batch_size']
    epochs = cfg['training']['epochs']
    lr = cfg['training']['learning_rate']
    gamma = 0.005 # 自监督 VAE 的 KL 权重通常很小
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}, Latent Dim: {cfg['model']['latent_dim']}")

    # 1. 加载数据
    print(">>> Loading Data...")
    train_df = pd.read_pickle(train_path)
    valid_df = pd.read_pickle(valid_path)
    if abs(train_df.iloc[:, 0].mean()) > 1.0:
        print("Warning: Data seems not standardized. Training VAE might be unstable.")
    
    # 自动特征选择
    context_cols = cfg['context_cols']
    exclude = ['trade_day', 'timestamp', label_col, 'ExchActionDay', 'ExchUpdateTime']+context_cols
    imp_df = pd.read_csv("/home/zyyuan/project2/feature_importance.csv")
    feature_cols = imp_df.head(300)['feature'].tolist()
    print(f"Selected Top 300 features for VAE input.")
    # feature_cols = ['adtm_30m','coppock_10_15_30m','rsj_30','TrendStrenth_30','ar_30','macd_long','rsi_long','cmo_30','mfi_30m','upper_bb','variance_diff_30m','pv_corr_10','OI_MA_600','netflow_30min','PSY_60','cci_30','amivest_lr_30','effective_depth_10min','skew_overall_10m','regression_factor_10','oi_vol_corr_30','rp_momentum_600','lower_band','net_inflow_min']+imp_df.head(300)['feature'].tolist()
    # 生成张量 (注意：这里我们不需要 y 进行训练，但需要 y 来保持行数对齐)
    X_train, C_train = prepare_tensor_with_context(train_df, feature_cols,context_cols, label_col, seq_len)
    X_valid, C_valid = prepare_tensor_with_context(valid_df, feature_cols,context_cols, label_col, seq_len)
    
    train_loader = DataLoader(TensorDataset(X_train, C_train), batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(TensorDataset(X_valid, C_valid), batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. 初始化模型
    model = SelfSupervisedVAE(
        encoder_config=cfg['model'],
        num_factors=len(feature_cols),
        seq_len=seq_len,
        hidden_dim=cfg['model']['hidden_dim'],
        latent_dim=cfg['model']['latent_dim'],
        context_dim=len(context_cols)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. 训练循环
    print("\n>>> Start Self-Supervised Training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_recon = 0
        train_kl = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x_batch, ctx_batch in pbar:
            x_batch = x_batch.to(device)
            ctx_batch = ctx_batch.to(device)
            
            optimizer.zero_grad()
            total_loss, recon, kl = model.run_model(x_batch, context=ctx_batch, gamma=gamma)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_recon += recon.item()
            train_kl += kl.item()
            pbar.set_postfix(Recon=f"{recon.item():.4f}", KL=f"{kl.item():.4f}")
            
        # 验证
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x_batch, ctx_batch in valid_loader:
                x_batch = x_batch.to(device)
                ctx_batch = ctx_batch.to(device)
                l, _, _ = model.run_model(x_batch, context=ctx_batch, gamma=gamma)
                valid_loss += l.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        print(f"Epoch {epoch+1} | Train Recon: {train_recon/len(train_loader):.4f} | Valid Loss: {avg_valid_loss:.4f},train_kl:{train_kl/len(train_loader):.4f}")
        
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), "best_ss_vae.pth")

    # 4. 特征提取与保存
    print("\n>>> Extracting Latent Features...")
    model.load_state_dict(torch.load("best_ss_vae.pth"))
    model.eval()
    
    # 定义提取函数
    def extract_and_save(dataset, df, name):
        # 这里的 dataset 是 (X, C) 的 TensorDataset
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_zs = []
        
        with torch.no_grad():
            for x_batch, ctx_batch in tqdm(loader, desc=f"Extracting {name}"):
                x_batch = x_batch.to(device)
                ctx_batch = ctx_batch.to(device)
                mu = model.extract_latent(x_batch, context=ctx_batch)
                all_zs.append(mu.cpu().numpy())
                
        z_array = np.concatenate(all_zs, axis=0)
        
        # 排序并赋值
        df_sorted = df.sort_values(['trade_day', 'timestamp']).reset_index(drop=True)
        for i in range(cfg['model']['latent_dim']):
            df_sorted[f'vae_latent_{i}'] = z_array[:, i]
            
        return df_sorted

    new_train_df = extract_and_save(TensorDataset(X_train, C_train), train_df, "Train")
    new_valid_df = extract_and_save(TensorDataset(X_valid, C_valid), valid_df, "Valid")
    new_train_df.to_pickle("train_with_vae_context.pkl")
    new_valid_df.to_pickle("valid_with_vae_context.pkl")
    final_model, metrics,df_res = retrain_with_selected_features(
            new_train_df, 
            new_valid_df, 
            [f'vae_latent_{i}' for i in range(cfg['model']['latent_dim'])],
            params=LGB_PARAMS,
            mixup_alpha=0.2,
            name = cfg["experiment_name"]
        )
    df_res['Avg_pnl'] = (df_res['DQ'] / df_res['Count']) / 15
    df_res.to_csv(f"/home/zyyuan/project2/pictures/{cfg["experiment_name"]}.csv")
    final_model.save_model(f'/home/zyyuan/project2/pictures/lgbm_context_model_{cfg["experiment_name"]}.txt')
    

if __name__ == "__main__":
    main()