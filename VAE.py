import argparse
import numpy as np
import torch
import pandas as pd
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import torch.nn.functional as F

# =========================================================================
#  1. 基礎設置和因子列表
# =========================================================================
def set_seed(seed: int):
    """設置隨機種子以確保實驗的可複現性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"全局隨機種子已設置為: {seed}")

factors= [
      "pvol_30min",
 "spread_volatility",
 "pvol_5min",
 "atr",
 "vol",
 "realized_vol_10",
 "OI_std_300",
 "OI_std_120",
 "realized_vol_60",
 "OI_std_600",
 "volume_oi_ratio",
 "spread",
 "ES",
 "effective_spread",
 "relative_spread",
 "OI_MA_600",
 "rp_momentum_600",
 "volume_cluster",
 "rp_momentum_20",
 "vol_std_600",
 "rv_corr_120",
 "upper_band",
 "lower_band",
 "oi_change",
 "rsi_short",
 "kdj",
 "upper_bb",
 "lower_bb",
 "rsi_long",
 "oi_support",
 "macd_long",
 "channel_position",
 "macd"
 ,"add_neg_add_realized_vol_60_add_realized_vol_60_abs_pvol_30min_rsi_short_",
 "sub_sub_channel_position_realized_vol_10_abs_neg_macd_",
 "sub_mul_pvol_30min_pvol_30min_rsi_long_",
 "rsi",
 "SOIR5",
 "delta2_10s_rolling_5min",
 "delta2_tick_rolling_1min",
 "ts_mean_ts_mean_ret_5min_",
 "vol_mid_corr_120",
 "neg_ts_mean_upper_bb_",
 "div_abs_ts_mean_ts_mean_volume_cluster_ts_std_rp_momentum_20_",
 "vol_std_120",
 "SOIR4",
 "pv_corr_120",
 "OFI4",
 "tick_ret",
 "buy_volume",
 "ewm_vol_10",
 "sellvol_1min",
 "netflow_10min",
 "bigorder_vol_pct_1min",
 "amivest_lr_30",
 "rp_momentum_600_kdj_corr",
 "rv_corr_120_OFI4_corr",
 "slope",
 "elascity_min",
 "cvar_10",
 "volret_ratio_10",
 "CV_10",
 "skew_upside_10m_rank",
 "OI_std_600_rank",
 "mfi_30m_mean",
 "sellvolpct_10min",
 "AB_vol",
 "PSY_60",
 "sq_down_return",
 "dtm",
 "skew_overall_10m_mean",
 "cci_30_mean",
 "effective_depth_30min_mean_30",
 "PSY_60_mean_30",
 "amihud_ratio_60_mean_30",
 "big_buy_volume_rank",
 "rsj_30_rank",
 "elascity_min_CV_10_corr"
 ,"netflow_10min_oi_change_corr"
]

# =========================================================================
#  2. 時序數據生成函數
# =========================================================================
def create_3d_tensors(df: pd.DataFrame, factors: list, label_col: str, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    df.sort_values(by=['date', 'time'], inplace=True); X_list, y_list = [], []; unique_dates = df['date'].unique()
    print(f"正在為 {len(unique_dates)} 個交易日生成 3D 時序數據 (序列長度={seq_len})...");
    for date in tqdm(unique_dates):
        daily_df = df[df['date'] == date].copy(); daily_x = daily_df[factors].values; daily_y = daily_df[label_col].values; num_rows_today = len(daily_df)
        for i in range(num_rows_today):
            start_idx = i - seq_len + 1
            if start_idx >= 0: sequence = daily_x[start_idx : i + 1, :]
            else:
                available_data = daily_x[0 : i + 1, :]; padding_data = daily_x[0, :]; padding_size = seq_len - len(available_data)
                padding = np.array([padding_data] * padding_size); sequence = np.vstack([padding, available_data])
            X_list.append(sequence); y_list.append(daily_y[i])
    X_3d_np = np.array(X_list, dtype=np.float32); y_np = np.array(y_list, dtype=np.float32); return X_3d_np, y_np

# =========================================================================
#  3. 模型定義庫
# =========================================================================

# --- Encoder: LSTM (基準模型) ---
class LSTM_Encoder(nn.Module):
    def __init__(self, num_factors, hidden_dim, latent_dim, num_layers=1):
        super(LSTM_Encoder, self).__init__(); self.lstm = nn.LSTM(num_factors, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim); self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x); h_n_last_layer = h_n[-1]; mu = self.fc_mu(h_n_last_layer); log_var = self.fc_log_var(h_n_last_layer)
        return mu, log_var

# --- Encoder: GRU ---
class GRU_Encoder(nn.Module):
    def __init__(self, num_factors, hidden_dim, latent_dim, num_layers=1):
        super(GRU_Encoder, self).__init__(); self.gru = nn.GRU(num_factors, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim); self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        _, h_n = self.gru(x); h_n_last_layer = h_n[-1]; mu = self.fc_mu(h_n_last_layer); log_var = self.fc_log_var(h_n_last_layer)
        return mu, log_var

# --- Encoder: TSMixer ---
class TSMixerBlock(nn.Module):
    def __init__(self, num_factors, seq_len, ff_dim):
        super(TSMixerBlock, self).__init__(); self.norm1 = nn.LayerNorm(num_factors)
        self.time_mlp = nn.Sequential(nn.Linear(seq_len, ff_dim), nn.ReLU(), nn.Linear(ff_dim, seq_len)); self.norm2 = nn.LayerNorm(num_factors)
        self.feature_mlp = nn.Sequential(nn.Linear(num_factors, ff_dim), nn.ReLU(), nn.Linear(ff_dim, num_factors))
    def forward(self, x):
        res = x; x = self.norm1(x); x = x.transpose(1, 2); x = self.time_mlp(x); x = x.transpose(1, 2); x = x + res
        res = x; x = self.norm2(x); x = self.feature_mlp(x); x = x + res
        return x

class TSMixer_Encoder(nn.Module):
    def __init__(self, num_factors, seq_len, latent_dim, ff_dim=64, num_blocks=1):
        super(TSMixer_Encoder, self).__init__(); self.blocks = nn.ModuleList([TSMixerBlock(num_factors, seq_len, ff_dim) for _ in range(num_blocks)])
        self.final_fc = nn.Linear(num_factors, 128); self.fc_mu = nn.Linear(128, latent_dim); self.fc_log_var = nn.Linear(128, latent_dim)
    def forward(self, x):
        for block in self.blocks: x = block(x)
        x_pooled = x.mean(dim=1); h = F.relu(self.final_fc(x_pooled)); mu = self.fc_mu(h); log_var = self.fc_log_var(h)
        return mu, log_var

# --- Encoder: DLinear ---
class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super(MovingAvg, self).__init__(); self.kernel_size = kernel_size; self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1, 1); x = torch.cat([front, x], dim=1); x = x.permute(0, 2, 1); x = self.avg(x); x = x.permute(0, 2, 1)
        return x
        
class DLinear_Encoder(nn.Module):
    def __init__(self, seq_len, num_factors, latent_dim, kernel_size=25):
        super(DLinear_Encoder, self).__init__(); self.moving_avg = MovingAvg(kernel_size); input_dim = seq_len * num_factors
        self.seasonal_fc = nn.Linear(input_dim, 128); self.trend_fc = nn.Linear(input_dim, 128)
        self.combined_fc = nn.Linear(256, 128); self.fc_mu = nn.Linear(128, latent_dim); self.fc_log_var = nn.Linear(128, latent_dim)
    def forward(self, x):
        trend_part = self.moving_avg(x); seasonal_part = x - trend_part
        seasonal_flat = seasonal_part.reshape(seasonal_part.size(0), -1); trend_flat = trend_part.reshape(trend_part.size(0), -1)
        h_seasonal = F.relu(self.seasonal_fc(seasonal_flat)); h_trend = F.relu(self.trend_fc(trend_flat))
        h_combined = torch.cat((h_seasonal, h_trend), dim=1); h = F.relu(self.combined_fc(h_combined))
        mu = self.fc_mu(h); log_var = self.fc_log_var(h)
        return mu, log_var

# --- Decoder (共用) ---
class LSTM_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_factors, seq_len, num_layers=1):
        super(LSTM_Decoder, self).__init__(); self.seq_len = seq_len; self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True); self.fc2 = nn.Linear(hidden_dim, num_factors)
    def forward(self, z):
        h = F.relu(self.fc1(z)); h_repeated = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(h_repeated); x_recon = self.fc2(lstm_out)
        return x_recon

# --- VAE (主模型，可切換 Encoder) ---
class VAE(nn.Module):
    def __init__(self, num_factors, hidden_dim, latent_dim, seq_len, encoder_type='lstm'):
        super(VAE, self).__init__()
        print(f"--- 正在初始化 VAE，使用 {encoder_type.upper()} 編碼器 ---")
        if encoder_type == 'lstm': self.encoder = LSTM_Encoder(num_factors, hidden_dim, latent_dim)
        elif encoder_type == 'gru': self.encoder = GRU_Encoder(num_factors, hidden_dim, latent_dim)
        elif encoder_type == 'tsmixer': self.encoder = TSMixer_Encoder(num_factors, seq_len, latent_dim)
        elif encoder_type == 'dlinear': self.encoder = DLinear_Encoder(seq_len, num_factors, latent_dim)
        else: raise ValueError("不支持的 Encoder 類型!")
        self.decoder = LSTM_Decoder(latent_dim, hidden_dim, num_factors, seq_len)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var); eps = torch.randn_like(std); return mu + eps * std
    def forward(self, x):
        mu, log_var = self.encoder(x); z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__(); self.fc1 = nn.Linear(latent_dim, 128); self.fc2 = nn.Linear(128, 64); self.fc3 = nn.Linear(64, 1)
    def forward(self, z):
        h = F.relu(self.fc1(z)); h = F.relu(self.fc2(h)); return self.fc3(h)

# =========================================================================
#  4. 輔助函數
# =========================================================================
def vae_loss_function(recon_x, x, mu, log_var):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum'); kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss, kl_loss

def extract_latent_features(vae_model, full_data_3d, batch_size, device):
    print(f"正在分批提取 {len(full_data_3d)} 個樣本的潛在特徵..."); vae_model.eval()
    dataset = TensorDataset(torch.tensor(full_data_3d, dtype=torch.float32)); loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_latent_mu = []
    with torch.no_grad():
        for (x_batch,) in tqdm(loader, desc="Extracting Features"):
            x_batch = x_batch.to(device); mu, _ = vae_model.encoder(x_batch); all_latent_mu.append(mu.cpu().numpy())
    return np.concatenate(all_latent_mu, axis=0)
def generate_positions_classification(df: pd.DataFrame, hold_time: int) -> pd.DataFrame:
    # print("开始根据分类结果生成仓位信号...")
    df_with_pos = df.copy()
    all_positions = []
    if not isinstance(df_with_pos.index, pd.MultiIndex):
        df_with_pos.set_index(['date','time'],inplace = True)
    df_with_pos.sort_values(by='time', inplace=True)
    active_signal = 0; signal_hold_counter = 0
    for date, daily_data in df_with_pos.groupby(level='date'):
        positions_today = pd.Series(index=daily_data.index, dtype=int)
        for i in range(len(daily_data)):
            current_pred_class = daily_data['pred'].iloc[i]
            if current_pred_class == 1: active_signal = 1; signal_hold_counter = hold_time
            elif current_pred_class == 2: active_signal = -1; signal_hold_counter = hold_time
            if active_signal != 0: signal_hold_counter -= 1
            current_position = 0
            if active_signal != 0 and signal_hold_counter >= 0: current_position = active_signal
            else: active_signal = 0
            if i == len(daily_data) - 1: current_position = 0
            positions_today.iloc[i] = current_position
        all_positions.append(positions_today)
    df_with_pos['position'] = pd.concat(all_positions)
    # print("仓位信号生成完毕。")
    return df_with_pos

def calculate_ticksize(df_with_pos: pd.DataFrame, commission: float, initial_capital: float = 10000000):
    # print("开始根据仓位计算收益...")
    capital = 0.0; net_values = [0.0]; all_daily_pnl = []; all_daily_ret = []
    unique_dates = df_with_pos.index.get_level_values('date').unique(); date_open = []
    for date in unique_dates:
        daily_data = df_with_pos.loc[date]; entry_price = 0; daily_pnl = 0; open_times = 0
        position_changes = daily_data['position'].diff().fillna(daily_data['position'].iloc[0])
        for i in range(len(daily_data)):
            if position_changes.iloc[i] != 0:
                prev_position = daily_data['position'].iloc[i] - position_changes.iloc[i]; target_position = daily_data['position'].iloc[i]
                if prev_position != 0:
                    if prev_position == 1: exit_price = daily_data['askp1_trade'].iloc[i]; pnl = (exit_price - entry_price) / 0.02; daily_pnl += pnl
                    elif prev_position == -1: exit_price = daily_data['bidp1_trade'].iloc[i]; pnl = (entry_price - exit_price) / 0.02; daily_pnl += pnl
                    entry_price = 0
                if target_position != 0:
                    open_times += 1
                    if target_position == 1: entry_price = daily_data['bidp1_trade'].iloc[i]
                    elif target_position == -1: entry_price = daily_data['askp1_trade'].iloc[i]
                    daily_pnl -= 1
        capital += daily_pnl; daily_ret = (daily_pnl*0.02) / ((daily_data['bidp1_trade'].iloc[0]+daily_data['askp1_trade'].iloc[0])/2); all_daily_ret.append(daily_ret); net_values.append(capital); all_daily_pnl.append(daily_pnl); date_open.append(open_times)
    results_df = pd.DataFrame({'date': unique_dates, 'net_value': net_values[1:], 'daily_pnl': all_daily_pnl, "opentimes":date_open, 'daily_return':all_daily_ret})
    results_df['Net_Value'] = (1+results_df['daily_return']).cumprod()
    if results_df['opentimes'].sum() == 0: return 0, 0, 1.0, 0
    return results_df['daily_pnl'].sum() / results_df['opentimes'].sum(), results_df['opentimes'].mean(),results_df['Net_Value'].iloc[-1],(results_df['Net_Value']/results_df['Net_Value'].cummax()).min() - 1
# =========================================================================
#  5. 主程序
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="使用多種可選Encoder的Sequential VAE框架")
    parser.add_argument('-encoder_type', type=str, default='tsmixer', choices=['lstm', 'gru', 'tsmixer', 'dlinear'], help="選擇Encoder的架構")
    parser.add_argument('-seed', type=int, default=42, help="隨機種子")
    parser.add_argument('-epochs', type=int, default=5, help="訓練輪次")
    parser.add_argument('-batch_size', type=int, default=1024, help="批次大小")
    parser.add_argument('-lr', type=float, default=5e-5, help="學習率")
    parser.add_argument('-seq_len', type=int, default=6, help="輸入序列的長度")
    parser.add_argument('-latent_dim', type=int, default=12, help="潛在特徵的維度")
    parser.add_argument('-hidden_dim', type=int, default=128, help="RNN隱藏層維度")
    parser.add_argument('-gamma', type=float, default=20.0, help="TC損失的權重 (解耦強度)")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("--- 正在加載和預處理數據 ---")
    try:
        train_df = pd.read_pickle("Train_final.pkl")
        test_df = pd.read_pickle("Test_final.pkl")
    except FileNotFoundError as e: print(f"錯誤: 數據文件未找到 - {e}"); return
    num_factors = len(factors); print(f"找到 {num_factors} 個因子。")
    scaler = StandardScaler(); train_df[factors] = scaler.fit_transform(train_df[factors]); test_df[factors] = scaler.transform(test_df[factors])
    X_train_3d, y_train = create_3d_tensors(train_df, factors, 'ret_600', args.seq_len)
    X_test_3d, y_test = create_3d_tensors(test_df, factors, 'ret_600', args.seq_len)
    train_dataset = TensorDataset(torch.tensor(X_train_3d, dtype=torch.float32)); train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print("--- 正在初始化模型和優化器 ---")
    vae = VAE(num_factors, args.hidden_dim, args.latent_dim, args.seq_len, args.encoder_type).to(device)
    discriminator = Discriminator(args.latent_dim).to(device)
    optimizer_vae = optim.Adam(vae.parameters(), lr=args.lr); optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr)
    d_loss_criterion = nn.BCEWithLogitsLoss()

    print(f"\n--- 開始訓練 (Encoder: {args.encoder_type.upper()}, Gamma={args.gamma}) ---")
    for epoch in range(args.epochs):
        vae.train(); discriminator.train(); total_recon_loss, total_kl_loss, total_tc_loss, total_d_loss = 0,0,0,0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for (x,) in progress_bar:
            x = x.to(device); batch_size = x.size(0)
            recon_x, mu, log_var = vae(x); recon_loss, kl_loss = vae_loss_function(recon_x, x, mu, log_var)
            z_sampled = vae.reparameterize(mu, log_var); d_z_logits = discriminator(z_sampled)
            zeros_for_vae = torch.zeros(batch_size, 1, device=device); tc_loss = d_loss_criterion(d_z_logits, zeros_for_vae)
            vae_loss = recon_loss + kl_loss + args.gamma * tc_loss
            optimizer_vae.zero_grad(); vae_loss.backward(); optimizer_vae.step()
            z_real = vae.reparameterize(mu, log_var).detach(); z_shuffled = z_real.clone()
            for i in range(args.latent_dim): z_shuffled[:, i] = z_shuffled[torch.randperm(z_shuffled.size(0)), i]
            d_real_logits = discriminator(z_real); d_shuffled_logits = discriminator(z_shuffled)
            ones_for_d = torch.ones(batch_size, 1, device=device); zeros_for_d = torch.zeros(batch_size, 1, device=device)
            d_loss = 0.5 * (d_loss_criterion(d_real_logits, ones_for_d) + d_loss_criterion(d_shuffled_logits, zeros_for_d))
            optimizer_d.zero_grad(); d_loss.backward(); optimizer_d.step()
            total_recon_loss += recon_loss.item(); total_kl_loss += kl_loss.item(); total_tc_loss += tc_loss.item(); total_d_loss += d_loss.item()
        avg_recon=total_recon_loss/len(train_loader.dataset); avg_kl=total_kl_loss/len(train_loader.dataset); avg_tc=total_tc_loss/len(train_loader); avg_d=total_d_loss/len(train_loader)
        progress_bar.set_postfix(Recon=f"{avg_recon:.2f}", KL=f"{avg_kl:.2f}", TC=f"{avg_tc:.2f}", D_Loss=f"{avg_d:.2f}")

    # print("\n--- 開始測試潛在時序特徵的預測能力 ---")
    # X_train_latent = extract_latent_features(vae, X_train_3d, args.batch_size, device)
    # X_test_latent = extract_latent_features(vae, X_test_3d, args.batch_size, device)
    # print(f"已提取潛在時序特徵，維度: {X_train_latent.shape[1]}")
    # ridge_latent = Ridge(alpha=1.0).fit(X_train_latent, y_train); preds_latent = ridge_latent.predict(X_test_latent)
    # corr_latent, _ = pearsonr(preds_latent, y_test)
    # print(f"【潛在特徵 - {args.encoder_type.upper()}】的預測 IC: {corr_latent:.6f}")
    # X_train_flat = X_train_3d.reshape(X_train_3d.shape[0], -1); X_test_flat = X_test_3d.reshape(X_test_3d.shape[0], -1)
    # ridge_original = Ridge(alpha=1.0).fit(X_train_flat, y_train); preds_original = ridge_original.predict(X_test_flat)
    # corr_original, _ = pearsonr(preds_original, y_test)
    # print(f"【原始扁平化特徵】的預測 IC: {corr_original:.6f}")
    vae.eval()
    # 步驟 1: 分批提取訓練集和測試集的潛在特徵 (z)
    X_train_latent = extract_latent_features(vae, X_train_3d, args.batch_size, device)
    X_test_latent = extract_latent_features(vae, X_test_3d, args.batch_size, device)
    print(f"已提取潛在時序特徵，維度: {X_train_latent.shape[1]}")

    # 步驟 2: 準備原始特徵的扁平化版本 (作為基準和組合的基礎)
    X_train_flat = X_train_3d[:,-1,:]
    X_test_flat = X_test_3d[:,-1,:]
    print(f"原始特徵扁平化後維度: {X_train_flat.shape[1]}")

    # 步驟 3: 創建組合特徵集 (拼接)
    X_train_combined = np.concatenate([X_train_flat, X_train_latent], axis=1)
    X_test_combined = np.concatenate([X_test_flat, X_test_latent], axis=1)
    print(f"組合特徵集維度: {X_train_combined.shape[1]}")


    # --- 場景 B: 增強模型 (使用原始特徵 + 潛在特徵) ---
    print("\n正在訓練增強模型 (原始特徵 + 潛在特徵)...")
    ridge_combined = Ridge(alpha=1.0)
    ridge_combined.fit(X_train_combined, y_train)
    preds_combined = ridge_combined.predict(X_test_combined)
    temp_test_data = test_df.copy()
    temp_test_data['pred'] = 0
    temp_test_data['pred'] = 0
    temp_test_data.loc[preds_combined>np.quantile(preds_combined,0.95),'pred'] = 1
    temp_test_data.loc[preds_combined<np.quantile(preds_combined,0.05),'pred'] = 2
 
 
    df_with_pos = generate_positions_classification(temp_test_data, hold_time=5)
    ave_ret, num, nv, _ = calculate_ticksize(df_with_pos, commission=0.02)
    print( ave_ret, num, nv)
    corr_combined, _ = pearsonr(preds_combined, y_test)
    print(f"【原始特徵 + 潛在特徵】的預測 IC: {corr_combined:.6f}")
    
    # --- (可選) 場景 C: 只使用潛在特徵的模型 ---
    print("\n正在訓練純潛在特徵模型...")
    ridge_latent = Ridge(alpha=1.0)
    ridge_latent.fit(X_train_latent, y_train)
    preds_latent = ridge_latent.predict(X_test_latent)
    
    corr_latent, _ = pearsonr(preds_latent, y_test)
    print(f"【僅潛在特徵】的預測 IC: {corr_latent:.6f}")


if __name__ == '__main__':
    main()
