import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import random
import json
import joblib

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
    
    return torch.from_numpy(X_all)

# =========================================================================
#  2. 模型架构: Denoising Autoencoder (DAE)
# =========================================================================

# --- Backbones (保持不变) ---
class LSTMBackbone(nn.Module):
    def __init__(self, num_factors, hidden_dim, num_layers=1, dropout=0.0):
        super(LSTMBackbone, self).__init__()
        self.lstm = nn.LSTM(num_factors, hidden_dim, num_layers, batch_first=True, dropout=dropout)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

class GRUBackbone(nn.Module):
    def __init__(self, num_factors, hidden_dim, num_layers=1, dropout=0.0):
        super(GRUBackbone, self).__init__()
        self.gru = nn.GRU(num_factors, hidden_dim, num_layers, batch_first=True, dropout=dropout)
    def forward(self, x):
        _, h_n = self.gru(x)
        return h_n[-1]

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
        self.final_fc = nn.Linear(num_factors, hidden_dim)
    def forward(self, x):
        for block in self.blocks: x = block(x)
        x_pooled = x.mean(dim=1)
        h = F.relu(self.final_fc(x_pooled))
        return h

class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1, 1)
        x = torch.cat([front, x], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class DLinearBackbone(nn.Module):
    def __init__(self, seq_len, num_factors, hidden_dim, kernel_size=25):
        super(DLinearBackbone, self).__init__()
        if kernel_size > seq_len: kernel_size = seq_len // 2 + 1
        self.moving_avg = MovingAvg(kernel_size)
        input_dim = seq_len * num_factors
        self.seasonal_fc = nn.Linear(input_dim, 128)
        self.trend_fc = nn.Linear(input_dim, 128)
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

# --- [修改] Denoising Autoencoder 类 ---
class DenoisingAutoencoder(nn.Module):
    def __init__(self, encoder_config, num_factors, seq_len, hidden_dim, latent_dim=64, noise_level=0.1):
        super(DenoisingAutoencoder, self).__init__()
        
        self.noise_level = noise_level # 噪声强度 (假设数据已标准化，通常 0.05-0.2)
        
        model_type = encoder_config.get('type', 'gru').lower()
        print(f"[Model] Initializing Denoising Autoencoder with: {model_type.upper()}")

        # 1. Encoder Backbone
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

        # 2. Bottleneck (压缩层)
        # DAE 不需要 log_var，只需要映射到 latent_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            # nn.ReLU() # 可选：如果希望 latent 是非负稀疏的
            # nn.LayerNorm(latent_dim) # 可选：标准化隐变量
        )
        
        # 3. Decoder (重构层)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_factors) # 输出维度 = 原始特征数
        )

    def add_noise(self, x):
        """给输入添加高斯噪声"""
        if self.training:
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        return x

    def forward(self, x):
        # 1. 噪声注入 (Input Corruption)
        x_noisy = self.add_noise(x)
        
        # 2. 编码 (Encoder)
        h = self.backbone(x_noisy)
        z = self.bottleneck(h) # Latent Representation
        
        # 3. 解码 (Decoder)
        recon_x = self.decoder(z)
        
        return recon_x, z

    def run_model(self, x):
        """
        训练步骤
        x: 原始干净的输入
        """
        # Forward (内部会自动加噪)
        recon_x, _ = self.forward(x)
        
        # Target: 干净的原始输入 (这里我们重构最后一个时间步)
        target = x[:, -1, :] 
        
        # Loss: 仅计算重构误差 (MSE)
        recon_loss = F.mse_loss(recon_x, target)
        
        return recon_loss

    def extract_latent(self, x):
        """推理步骤：提取干净输入的隐特征"""
        with torch.no_grad():
            # 推理时不加噪
            h = self.backbone(x)
            z = self.bottleneck(h)
        return z

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
    parser.add_argument('--latent_dim', type=int, default=64, help='特征压缩维度')
    parser.add_argument('--noise', type=float, default=0.1, help='输入噪声强度 (std)')
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
    latent_dim = cfg['model']['latent_dim']
    noise = cfg['training']['noise']
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}, Latent Dim: {latent_dim}, Noise Level: {noise}")

    # 1. 加载数据
    print(">>> Loading Data...")
    train_df = pd.read_pickle(train_path)
    valid_df = pd.read_pickle(valid_path)
    
    # 简单检查数据是否归一化 (DAE 对数据尺度敏感)
    if abs(train_df.iloc[:, 0].mean()) > 5.0:
        print("Warning: Data seems NOT standardized. DAE performance might be poor.")

    # 自动特征选择
    exclude = ['trade_day', 'timestamp', label_col, 'ExchActionDay', 'ExchUpdateTime']
    try:
        imp_df = pd.read_csv("/home/zyyuan/project2/feature_importance.csv")
        feature_cols = imp_df.head(300)['feature'].tolist()
        print(f"Selected Top 300 features.")
    except:
        feature_cols = [c for c in train_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])]
        print(f"Using all {len(feature_cols)} features.")

    # 生成张量
    X_train_tensor = prepare_tensor_data(train_df, feature_cols, label_col, seq_len)
    X_valid_tensor = prepare_tensor_data(valid_df, feature_cols, label_col, seq_len)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(TensorDataset(X_valid_tensor), batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. 初始化模型 [修改为 DenoisingAutoencoder]
    model = DenoisingAutoencoder(
        encoder_config=cfg['model'],
        num_factors=len(feature_cols),
        seq_len=seq_len,
        hidden_dim=cfg['model']['hidden_dim'],
        latent_dim=latent_dim,
        noise_level=noise # 传入噪声参数
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. 训练循环
    print("\n>>> Start DAE Training (Reconstruction with Noise)...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for (x_batch,) in pbar:
            x_batch = x_batch.to(device)
            
            optimizer.zero_grad()
            
            # Loss 只包含 MSE
            loss = model.run_model(x_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
        # 验证
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for (x_batch,) in valid_loader:
                x_batch = x_batch.to(device)
                # 验证集计算 Loss 也不加噪（或者可以加噪测试鲁棒性，通常不加）
                # 这里我们复用 run_model，注意 run_model 内部 forward 在 eval 模式下是不加噪的
                l = model.run_model(x_batch)
                valid_loss += l.item()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")
        
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), "best_dae.pth")

    # 4. 特征提取与保存
    print("\n>>> Extracting Latent Features...")
    model.load_state_dict(torch.load("best_dae.pth"))
    model.eval()
    
    def extract_and_save(loader, df, name_suffix):
        all_zs = []
        with torch.no_grad():
            for (x_batch,) in tqdm(loader, desc=f"Extracting {name_suffix}"):
                x_batch = x_batch.to(device)
                z = model.extract_latent(x_batch) # 提取隐变量
                all_zs.append(z.cpu().numpy())
        
        z_array = np.concatenate(all_zs, axis=0)
        
        df_sorted = df.sort_values(['trade_day', 'timestamp']).reset_index(drop=True)
        
        if len(df_sorted) != len(z_array):
            print(f"Error: Length mismatch! DF: {len(df_sorted)}, Z: {len(z_array)}")
            return None
        
        # 命名新特征 (dae_latent_0, ...)
        for i in range(latent_dim):
            df_sorted[f'dae_latent_{i}'] = z_array[:, i]
            
        return df_sorted

    full_train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=False)
    
    new_train_df = extract_and_save(full_train_loader, train_df, "Train")
    new_valid_df = extract_and_save(valid_loader, valid_df, "Valid")
    
    # 5. 保存
    if new_train_df is not None and new_valid_df is not None:
        save_dir = os.path.dirname(train_path)
        new_train_path = os.path.join(save_dir, "train_with_dae.pkl")
        new_valid_path = os.path.join(save_dir, "valid_with_dae.pkl")
        
        new_train_df.to_pickle(new_train_path)
        new_valid_df.to_pickle(new_valid_path)
        
        print(f"Saved augmented data to:")
        print(f"  {new_train_path}")
        print(f"  {new_valid_path}")
        
        # 简单检查相关性
        print("\nChecking Correlation of DAE feature 0 with Label:")
        corr = new_train_df[f'dae_latent_0'].corr(new_train_df[label_col])
        print(f"Correlation: {corr:.6f}")

if __name__ == "__main__":
    main()