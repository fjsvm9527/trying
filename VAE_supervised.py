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
import joblib

# =========================================================================
#  1. 数据预处理 (保持不变)
# =========================================================================
def prepare_tensor_data(df, feature_cols, label_col, seq_len=30):
    """
    将 DataFrame 转换为 3D 时序张量
    目标形状: (Batch, Seq_Len, Features) -> (样本数, 时间步, 特征数)
    """
    # 1. 检查并排序
    required_cols = ['trade_day', 'timestamp'] + feature_cols + [label_col]
    if not all(col in df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        raise ValueError(f"DataFrame 缺少列: {missing}")

    print(">>> [Data] 正在按日期和时间排序...")
    df = df.sort_values(['trade_day', 'timestamp'])

    X_daily_list = []
    y_daily_list = []

    # 2. 按交易日分组处理
    groups = df.groupby('trade_day')
    
    print(f">>> [Data] 正在生成时序张量 (Seq_Len={seq_len})...")
    for date, group in tqdm(groups, desc="Processing Days"):
        # 提取当前特征矩阵 (Time, Features)
        feature_vals = group[feature_cols].values.astype(np.float32)
        label_vals = group[label_col].values.astype(np.float32)
        
        # --- 核心逻辑: 头部填充 (Padding) ---
        # 目的：让当天的第 1 行数据也能作为预测目标
        # padding 后形状: (Time + Seq - 1, Features)
        pad_width = ((seq_len - 1, 0), (0, 0)) 
        padded_features = np.pad(feature_vals, pad_width, mode='edge')
        
        # --- 核心逻辑: 向量化滑动窗口 ---
        # 1. sliding_window_view 默认将窗口维度放在最后
        #    原始输出形状: (Batch, Features, Seq_Len)
        windows_raw = np.lib.stride_tricks.sliding_window_view(padded_features, window_shape=seq_len, axis=0)
        
        # 2. 【关键修改】交换维度
        #    由 (Batch, Features, Seq_Len) -> (Batch, Seq_Len, Features)
        #    0: Batch (不变)
        #    1: Features -> 变成 2
        #    2: Seq_Len  -> 变成 1
        windows = windows_raw.transpose(0, 2, 1)
        
        X_daily_list.append(windows)
        y_daily_list.append(label_vals)

    # 3. 拼接所有天的数据
    if not X_daily_list:
        raise ValueError("数据为空或处理失败")

    X_all = np.concatenate(X_daily_list, axis=0) # [Total_Samples, Seq_Len, Features]
    y_all = np.concatenate(y_daily_list, axis=0) # [Total_Samples]

    print(f">>> [Data] 生成完成! X: {X_all.shape} (Batch, Time, Feat), y: {y_all.shape}")
    
    return torch.from_numpy(X_all),torch.from_numpy(y_all)

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
    
# class SelfSupervisedVAE(nn.Module):
#     def __init__(self, num_factors, seq_len, hidden_dim, latent_dim=1):
#         super(SelfSupervisedVAE, self).__init__()
#         self.seq_len = seq_len
#         self.num_factors = num_factors
        
#         # [Encoder]: GRU 提取时序特征 -> 压缩为 mu, log_var
#         self.gru = nn.GRU(
#             input_size=num_factors,
#             hidden_size=hidden_dim,
#             num_layers=1,
#             batch_first=True,
#             dropout=0.1
#         )
        
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
#         self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
#         # [Decoder]: Latent z -> 重构输入的特征 X
#         # 这里我们尝试重构序列的**最后一个时间步** (Last Step Reconstruction)
#         # 也可以重构整个序列，但重构Last Step对于提取"当前状态"更有效
#         self.decoder_net = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_factors) # 输出维度 = 原始特征数
#         )

#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         # x: [Batch, Seq, Feat]
        
#         # 1. Encoder
#         output, _ = self.gru(x)
#         last_step_h = output[:, -1, :] # 取最后时刻的隐状态
        
#         mu = self.fc_mu(last_step_h)
#         log_var = self.fc_log_var(last_step_h)
        
#         # 2. Sampling
#         z = self.reparameterize(mu, log_var)
        
#         # 3. Decoder (重构)
#         recon_x = self.decoder_net(z)
        
#         return recon_x, mu, log_var

#     def run_model(self, x, gamma=0.01):
#         """
#         训练步骤
#         x: 输入特征序列
#         gamma: KL 散度权重
#         """
#         # 前向传播
#         recon_x_last_step, mu, log_var = self.forward(x)
        
#         # 目标: 重构输入序列的最后一个时间步
#         target = x[:, -1, :] 
        
#         # A. 重构损失 (MSE)
#         recon_loss = F.mse_loss(recon_x_last_step, target)
        
#         # B. KL 散度 (Posterior vs Standard Normal)
#         # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
#         kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#         kl_loss = kl_loss / x.size(0) # Batch Mean
        
#         total_loss = recon_loss + gamma * kl_loss
        
#         return total_loss, recon_loss, kl_loss

#     def extract_latent(self, x):
#         """推理步骤：仅提取 mu 作为特征"""
#         with torch.no_grad():
#             output, _ = self.gru(x)
#             last_step_h = output[:, -1, :]
#             mu = self.fc_mu(last_step_h)
#         return mu # [Batch, Latent_Dim]

class SupervisedVAE(nn.Module):
    def __init__(self, encoder_config, num_factors, seq_len, hidden_dim, latent_dim=1):
        super(SupervisedVAE, self).__init__()
        
        # --- 1. Encoder Backbone (保持不变) ---
        model_type = encoder_config.get('type', 'gru').lower()
        print(f"[Model] Initializing Supervised VAE with: {model_type.upper()}")
        
        if model_type == 'gru':
            self.backbone = GRUBackbone(num_factors, hidden_dim, encoder_config.get('num_layers', 1), encoder_config.get('dropout', 0.))
        elif model_type == 'lstm':
            self.backbone = LSTMBackbone(num_factors, hidden_dim, encoder_config.get('num_layers', 1), encoder_config.get('dropout', 0.))
        elif model_type == 'tsmixer':
            self.backbone = TSMixerBackbone(num_factors, seq_len, hidden_dim, encoder_config.get('ff_dim', 64), encoder_config.get('num_blocks', 2))
        elif model_type == 'dlinear':
            self.backbone = DLinearBackbone(seq_len, num_factors, hidden_dim, encoder_config.get('kernel_size', 25))
        else:
            raise ValueError(f"Unknown type: {model_type}")

        # --- 2. VAE Projection (保持不变) ---
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # --- 3. Decoder (重构 X) ---
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_factors)
        )
        
        # --- [新增] 4. Predictor Head (预测 y) ---
        # 建立 z -> y 的监督关系
        self.predictor_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1) # 回归预测
        )

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu # 预测时直接用均值

    def forward(self, x):
        # 1. Encode
        h = self.backbone(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        # 2. Sample
        z = self.reparameterize(mu, log_var)
        
        # 3. Decode (Reconstruction)
        recon_x = self.decoder_net(z)
        
        # 4. Predict (Supervision) [新增]
        pred_y = self.predictor_net(z)
        
        return recon_x, pred_y, mu, log_var

    def run_model(self, x, y, gamma=0.01, alpha=1.0):
        """
        Args:
            gamma: KL 权重
            alpha: 预测损失权重 (控制监督信号的强弱)
        """
        recon_x, pred_y, mu, log_var = self.forward(x)
        
        # Target X (Last Step)
        target_x = x[:, -1, :] 
        
        # 1. Reconstruction Loss
        recon_loss = F.mse_loss(recon_x, target_x)
        
        # 2. Prediction Loss (Supervised)
        # 确保 y 的形状匹配 [Batch, 1]
        pred_loss = F.mse_loss(pred_y.squeeze(), y)
        
        # 3. KL Divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / x.size(0)
        
        # Total Loss
        total_loss = recon_loss + gamma * kl_loss + alpha * pred_loss
        
        return total_loss, recon_loss, kl_loss, pred_loss

    def extract_latent(self, x):
        with torch.no_grad():
            h = self.backbone(x)
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
    label_scaler = StandardScaler()
    train_df[label_col] = label_scaler.fit_transform(train_df[[label_col]].values)
    
    # B. Transform on Valid (使用训练集的均值方差)
    valid_df[label_col] = label_scaler.transform(valid_df[[label_col]].values)
    
    # C. 保存 Scaler (重要！后续如果想还原真实预测值需要用到)
    scaler_save_path = os.path.join(os.path.dirname(train_path), 'vae_label_scaler.pkl')
    joblib.dump(label_scaler, scaler_save_path)
    # 标准化特征 (这对 VAE 重构至关重要)
    # 假设之前的步骤已经标准化过了，如果没有，这里需要再次标准化
    # 这里我们加载之前保存的 scaler 或者假设数据已经 scale
    # 简单检查一下均值
    if abs(train_df.iloc[:, 0].mean()) > 1.0:
        print("Warning: Data seems not standardized. Training VAE might be unstable.")
    
    # 自动特征选择
    exclude = ['trade_day', 'timestamp', label_col, 'ExchActionDay', 'ExchUpdateTime']
    # 尝试读取 feature importance 文件筛选特征，或者使用全量特征
    try:
        imp_df = pd.read_csv("/home/zyyuan/project2/feature_importance.csv")
        feature_cols = imp_df.head(300)['feature'].tolist()
        print(f"Selected Top 300 features for VAE input.")
    except:
        feature_cols = [c for c in train_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])]
        print(f"Using all {len(feature_cols)} features.")

    # 生成张量 (注意：这里我们不需要 y 进行训练，但需要 y 来保持行数对齐)
    alpha = 0.5 # 监督损失的权重，建议 0.1 ~ 1.0 之间
    
    # ... (数据加载部分) ...
    # 修改：prepare_tensor_data 现在返回两个张量
    X_train, y_train = prepare_tensor_data(train_df, feature_cols, label_col, seq_len)
    X_valid, y_valid = prepare_tensor_data(valid_df, feature_cols, label_col, seq_len)
    
    # 修改：TensorDataset 包含 y
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型 (换成 SupervisedVAE)
    model = SupervisedVAE(
        encoder_config=cfg['model'],
        num_factors=len(feature_cols),
        seq_len=seq_len,
        hidden_dim=cfg['model']['hidden_dim'],
        latent_dim=cfg['model']['latent_dim']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n>>> Start Supervised-VAE Training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_recon = 0
        train_pred = 0
        train_kl = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        # 修改：解包 x 和 y
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            # 修改：传入 y 和 alpha
            total_loss, recon, kl, pred = model.run_model(x_batch, y_batch, gamma=gamma, alpha=alpha)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_recon += recon.item()
            train_kl += kl.item()
            train_pred += pred.item()
            
            pbar.set_postfix(
                Recon=f"{recon.item():.4f}", 
                Pred=f"{pred.item():.4f}", # 显示预测损失
                KL=f"{kl.item():.4f}"
            )
            
        # 验证循环
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                l, _, _, _ = model.run_model(x_batch, y_batch, gamma=gamma, alpha=alpha)
                valid_loss += l.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        avg_train_pred = train_pred / len(train_loader)
        
        print(f"Epoch {epoch+1} | Valid Loss: {avg_valid_loss:.4f} | Train Pred MSE: {avg_train_pred:.4f}")
        
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), "best_sup_vae.pth")

    # ... (后续提取特征 extract_and_save 部分逻辑不需要变，因为 extract_latent 接口没变) ...
    # extract_and_save 只用 Loader 里的 x_batch 即可
    def extract_and_save(loader, df, name_suffix):
        all_zs = []
        with torch.no_grad():
            # Loader 现在 yield (x, y)，我们只需要 x
            for x_batch, _ in tqdm(loader, desc=f"Extracting {name_suffix}"):
                x_batch = x_batch.to(device)
                z = model.extract_latent(x_batch)
                all_zs.append(z.cpu().numpy())
        z_array = np.concatenate(all_zs, axis=0) # [N, 1]
        
        # 将提取出的特征加入 DataFrame
        # 因为我们之前 sort_values 过，且 prepare_tensor_data 保持了顺序，所以可以直接赋值
        # 注意：prepare_tensor_data 是按 trade_day, timestamp 排序的，df 也必须是一样的排序
        df_sorted = df.sort_values(['trade_day', 'timestamp']).reset_index(drop=True)
        
        # 检查长度是否一致
        if len(df_sorted) != len(z_array):
            print(f"Error: Length mismatch! DF: {len(df_sorted)}, Z: {len(z_array)}")
            # 这里的 mismatch 通常是因为 prepare_tensor_data 里可能有 drop 操作或者 df 变动
            # 在目前的逻辑里，长度应该是一致的
            return None
        
        # 命名新特征
        for i in range(args_cmd.latent_dim):
            df_sorted[f'vae_latent_{i}'] = z_array[:, i]
            
        return df_sorted

    # 重新构建不打乱的 Loader 用于顺序提取
    # 注意：prepare_tensor_data 内部已经排过序了，所以生成的 tensor 是有序的
    full_train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False, num_workers=4)
    
    new_train_df = extract_and_save(full_train_loader, train_df, "Train")
    new_valid_df = extract_and_save(valid_loader, valid_df, "Valid")
    
    # 5. 保存带有新特征的数据，供 LightGBM 使用
    if new_train_df is not None and new_valid_df is not None:
        save_dir = os.path.dirname(train_path)
        new_train_path = os.path.join(save_dir, "train_with_vae.pkl")
        new_valid_path = os.path.join(save_dir, "valid_with_vae.pkl")
        
        new_train_df.to_pickle(new_train_path)
        new_valid_df.to_pickle(new_valid_path)
        
        print(f"Saved augmented data to:")
        print(f"  {new_train_path}")
        print(f"  {new_valid_path}")
        
        # 打印新特征的相关性 (与 Label)
        print("\nChecking Correlation of new feature with Label:")
        corr_matrix = pd.DataFrame()
        train_corr = new_train_df[[f'vae_latent_{i}' for i in range(args_cmd.latent_dim)]].corrwith(new_train_df[label_col])
        valid_corr = new_valid_df[[f'vae_latent_{i}' for i in range(args_cmd.latent_dim)]].corrwith(new_valid_df[label_col])
        corr_matrix['train_corr'] = train_corr
        corr_matrix['valid_corr'] = valid_corr
        corr_matrix.to_csv("vae_test.csv")

if __name__ == "__main__":
    main()