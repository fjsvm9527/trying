import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
import joblib
import json
import os
import glob
import re
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numba
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Numba 加速函数 (来自第二段代码，必须放在全局)
# ==============================================================================
@numba.jit(nopython=True, nogil=True)
def _calculate_cvar_window(window_returns: np.ndarray, alpha: float) -> float:
    valid_returns = window_returns[~np.isnan(window_returns)]
    if len(valid_returns) == 0:
        return np.nan
    var_threshold = np.percentile(valid_returns, alpha * 100)
    tail_losses = valid_returns[valid_returns < var_threshold]
    if len(tail_losses) == 0:
        return var_threshold
    cvar = np.mean(tail_losses)
    return cvar

# ==============================================================================
# 2. Market Feature 计算逻辑 (封装第二段代码)
# ==============================================================================
def preprocess_for_calc(df_raw):
    """市场特征计算的前置处理"""
    df = df_raw.copy()
    # 计算中间价
    df['mid_price'] = (df['bp1'] + df['ap1']) / 2
    df.loc[df['bp1'] == 0, 'mid_price'] = df.loc[df['bp1'] == 0, 'ap1']
    df.loc[df['ap1'] == 0, 'mid_price'] = df.loc[df['ap1'] == 0, 'bp1']
    
    # 计算衍生基础量
    df['delta_volume'] = df['volume'].diff()
    df.iloc[0, df.columns.get_loc('delta_volume')] = df.iloc[0]['volume'] # 填补第一个NaN
    
    df['vol'] = df['volume'] - df['volume'].shift(120) # 过去120 tick的成交量
    df['high'] = df['mid_price'].rolling(120).max()
    df['low'] = df['mid_price'].rolling(120).min()
    df['open'] = df['mid_price'].shift(119)
    
    df['delta_turnover'] = df['turnover'].diff()
    df['tick_ret'] = df['mid_price']/df['mid_price'].shift(1) - 1
    
    # 大单逻辑
    df['bigorder_volume'] = 0
    df.loc[df['delta_volume']>=10, 'bigorder_volume'] = df['delta_volume']
    df['is_bigorder'] = 0
    df.loc[df['bigorder_volume']>0, 'is_bigorder'] = 1
    df['bigorder_pct'] = df['is_bigorder'].rolling(1200).mean()
    
    # 防止除0
    for col in ['open', 'high', 'low', 'mid_price']:
        df[col] = df[col].clip(lower=1e-9)
        
    return df

def calculate_market_features_full(df_input):
    """
    计算所有需要的 Market Features。
    注意：输入必须是连续的 DataFrame (未采样)，否则 Rolling 计算会出错。
    """
    df = preprocess_for_calc(df_input)
    
    # 基础指标
    df['ret_min'] = df['mid_price'] / df['mid_price'].shift(1) - 1
    df['is_up'] = np.where(df['ret_min'] > 0, 1, 0)
    dollar_volume = df['mid_price'] * df['vol']
    dollar_volume = dollar_volume.replace(0, np.nan)
    illiq_ratio = df['ret_min'].abs() / dollar_volume * 1e6
    
    # --- 核心特征计算 ---
    df['PSY_60'] = df['is_up'].rolling(window=60).mean()
    df['ret_kurt_120'] = df['ret_min'].rolling(window=120, min_periods=10).kurt()
    df['amihud_ratio_60'] = illiq_ratio.rolling(window=60, min_periods=10).mean()
    df['ret_5min'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    
    # AR_10
    ar_num = (df['high'] - df['open']).rolling(10, min_periods=5).sum()
    ar_den = (df['open'] - df['low']).rolling(10, min_periods=5).sum()
    df['ar_10'] = ar_num / (ar_den + 1e-9)
    
    # Lower BB
    ma20 = df['mid_price'].rolling(window=20).mean()
    std20 = df['mid_price'].rolling(window=20).std()
    df['lower_bb'] = (ma20 - 2 * std20 - df['mid_price']) / df['mid_price']
    
    # CV_30
    df['CV_30'] = df['ret_min'].rolling(30, min_periods=5).var() / np.abs(df['ret_min'].rolling(30, min_periods=5).mean() + 1e-9)
    
    # Momentum
    df['rp_momentum_120'] = (df['tick_ret'] * df['delta_turnover']).rolling(120).sum()
    
    # CVaR (使用 Numba)
    # values.flatten() 确保传入的是一维数组
    df['cvar_10'] = df['ret_min'].rolling(window=10).apply(_calculate_cvar_window, raw=True, args=(0.05,))
    
    target_cols = [
        'PSY_60', 'ret_kurt_120', 'amihud_ratio_60', 'ret_5min', 
        'ar_10', 'lower_bb', 'CV_30', 'rp_momentum_120', 
        'bigorder_pct', 'cvar_10'
    ]
    
    # 返回仅包含这些特征的 DataFrame，索引与原 df 一致
    return df[target_cols]

# ==============================================================================
# 3. VAE 模型定义 (保持不变)
# ==============================================================================
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
        for block in self.blocks:
            x = block(x)
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

class SelfSupervisedVAE(nn.Module):
    def __init__(self, encoder_config, num_factors, seq_len, hidden_dim, latent_dim=1):
        super(SelfSupervisedVAE, self).__init__()
        model_type = encoder_config.get('type', 'gru').lower()
        if model_type == 'gru':
            self.backbone = GRUBackbone(num_factors, hidden_dim, encoder_config.get('num_layers', 1), encoder_config.get('dropout', 0.0))
        elif model_type == 'lstm':
            self.backbone = LSTMBackbone(num_factors, hidden_dim, encoder_config.get('num_layers', 1), encoder_config.get('dropout', 0.0))
        elif model_type == 'tsmixer':
            self.backbone = TSMixerBackbone(num_factors, seq_len, hidden_dim, encoder_config.get('ff_dim', 64), encoder_config.get('num_blocks', 2))
        elif model_type == 'dlinear':
            self.backbone = DLinearBackbone(seq_len, num_factors, hidden_dim, encoder_config.get('kernel_size', 25))
        else:
            raise ValueError(f"Unknown encoder type: {model_type}")
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder_net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_factors))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.backbone(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder_net(z)
        return recon_x, mu, log_var

    def extract_latent(self, x):
        with torch.no_grad():
            h = self.backbone(x)
            mu = self.fc_mu(h)
        return mu

# ==============================================================================
# 4. 推理引擎 (集成 Market Feature Calculation)
# ==============================================================================
class ModelInference:
    def __init__(self, model_dir, device='cuda'):
        self.device = torch.device(device)
        print(f">>> Loading artifacts from {model_dir}...")
        
        # 1. Scaler
        self.scaler = joblib.load('/home/zyyuan/project2/processed_data/scaler.pkl')
        self.scale_cols = joblib.load('/home/zyyuan/project2/processed_data/feature_cols.pkl')
        
        # 2. VAE
        with open('/home/zyyuan/project2/config/20251229_VAE_lstm.json', 'r') as f:
            self.vae_cfg = json.load(f)
        imp_df = pd.read_csv("/home/zyyuan/project2/feature_importance.csv")
        self.vae_input_cols = imp_df.head(300)['feature'].tolist()
        
        self.vae_model = SelfSupervisedVAE(
            encoder_config=self.vae_cfg['model'],
            num_factors=len(self.vae_input_cols),
            seq_len=self.vae_cfg['data']['seq_len'],
            hidden_dim=self.vae_cfg['model']['hidden_dim'],
            latent_dim=self.vae_cfg['model']['latent_dim']
        ).to(self.device)
        
        state_dict = torch.load('/home/zyyuan/project2/best_ss_vae.pth', map_location=device)
        self.vae_model.load_state_dict(state_dict)
        self.vae_model.eval()
        
        # 3. LightGBM
        self.lgbm_model = lgb.Booster(model_file='/home/zyyuan/project2/lgbm_mixup_top300_model.txt')
        
        self.context_factors = ['is_pm','is_am','is_night','gate_session_decay','gate_open_impulse']
        self.vae_factors = [f'vae_latent_{i}' for i in range(64)]
        self.market_factors = [
            'PSY_60', 'ret_kurt_120', 'amihud_ratio_60', 'ret_5min', 
            'ar_10', 'lower_bb', 'CV_30', 'rp_momentum_120', 
            'bigorder_pct', 'cvar_10'
        ]
        
        # ⚠️ 重要: 请确认 LGBM 训练时的 feature_name 顺序
        # 假设训练时是: Market + Time + VAE
        # 如果训练时没有用 Market Features，这里加进去会报错。
        # 如果训练时用了，必须加上。
        self.lgbm_features = self.market_factors + self.context_factors + self.vae_factors
        # 如果你确定训练时只有 Time + VAE，请注释掉上面一行，使用下面这行：
        # self.lgbm_features = self.context_factors + self.vae_factors

    def preprocess_vae_input(self, df):
        """仅对 VAE 需要的输入进行标准化"""
        df_scaled = df.copy()
        df_scaled[self.scale_cols] = df_scaled[self.scale_cols].fillna(0)
        df_scaled[self.scale_cols] = self.scaler.transform(df_scaled[self.scale_cols].values).astype(np.float32)
        return df_scaled

    def extract_vae_features_batch(self, df_sampled, full_feature_matrix, indices, seq_len):
        """
        高效提取 VAE 特征。
        df_sampled: 已经采样好的 DataFrame (用于拼接结果)
        full_feature_matrix: 全量连续的标准化特征矩阵 (numpy array)
        indices: 采样点的索引 (对应 full_feature_matrix 的行号)
        """
        # 构造 Batch Input Tensor
        # 我们需要从 full_matrix 中根据 indices 提取过去 seq_len 的窗口
        # Shape: [Batch, Seq_Len, Feat]
        
        num_samples = len(indices)
        num_feats = full_feature_matrix.shape[1]
        
        batch_input = np.zeros((num_samples, seq_len, num_feats), dtype=np.float32)
        
        # 简单的循环提取 (如果 Batch 很大可以进一步优化，但这里通常很快)
        # 注意处理边界: 如果 index < seq_len，需要 padding (这里假设 indices >= seq_len-1)
        for i, idx in enumerate(indices):
            start_idx = idx - seq_len + 1
            if start_idx < 0:
                # 头部 Padding 逻辑
                valid_len = idx + 1
                padding_len = seq_len - valid_len
                # 取有效数据
                data_slice = full_feature_matrix[0 : idx+1]
                # Pad 头部
                pad_width = ((padding_len, 0), (0, 0))
                batch_input[i] = np.pad(data_slice, pad_width, mode='edge')
            else:
                batch_input[i] = full_feature_matrix[start_idx : idx+1]
        
        # 转 Tensor
        tensor_data = torch.from_numpy(batch_input).float().to(self.device)
        
        # 推理
        latents_list = []
        batch_size = 2048
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = tensor_data[i : i+batch_size]
                mu = self.vae_model.extract_latent(batch)
                latents_list.append(mu.cpu().numpy())
        
        latents = np.concatenate(latents_list, axis=0)
        
        # 拼接到 Sampled DataFrame
        for i in range(self.vae_cfg['model']['latent_dim']):
            df_sampled[f'vae_latent_{i}'] = latents[:, i]
            
        return df_sampled

    def predict_mixed(self, df_sampled):
        """最终预测"""
        # 确保列存在且顺序一致
        # 填充可能的 NaN (Market features 在开头可能有 NaN)
        df_sampled[self.market_factors] = df_sampled[self.market_factors].fillna(0)
        
        X = df_sampled[self.lgbm_features]
        preds = self.lgbm_model.predict(X)
        return preds

# ==============================================================================
# 5. 主流程
# ==============================================================================
def create_timestamp_fast(df):
    try:
        if not np.issubdtype(df['ExchActionDay'].dtype, np.datetime64):
            date_part = pd.to_datetime(df['ExchActionDay'].astype(str), format='%Y%m%d')
        else:
            date_part = df['ExchActionDay']
        def parse_time(t_str):
            try:
                parts = t_str.split(':')
                return int(parts[0])*3600 + int(parts[1])*60 + (int(parts[2]) if len(parts)>2 else 0)
            except:
                return 0
        time_secs = df['ExchUpdateTime'].apply(parse_time)
        total_secs = time_secs + df['ExchUpdateMillisec'] / 1000.0
        df['timestamp'] = date_part + pd.to_timedelta(total_secs, unit='s')
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
    except Exception as e:
        print(f"时间戳生成失败: {e}")
    return df

def get_feature_columns(df, config_cols):
    cols = [c for c in df.columns if c not in config_cols and c != 'timestamp']
    numeric_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return sorted(numeric_cols)

def get_filtered_files(data_dir, start_date_str):
    all_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    target_files = []
    start_date = int(start_date_str)
    for f in all_files:
        filename = os.path.basename(f)
        match = re.search(r'(20\d{6})', filename)
        if match:
            file_date = int(match.group(1))
            if file_date >= start_date:
                target_files.append((file_date, f))
    target_files.sort(key=lambda x: x[0])
    return [x[1] for x in target_files]

CONFIG_EXCLUDE = [
    'ExchActionDay', 'ExchUpdateTime', 'ExchUpdateMillisec', 'nano', 'mid',
    'last', 'vwap', 'bv1', 'bp1', 'ap1', 'av1', 'triggerInst',
    'triggerInst.volume', 'prj2_2_label', 'prj2_1_label',
    'LABEL_CAL_DQ_inst1_60', 'LABEL_CAL_DQ_inst1_900', 'featSite','is_pm','is_am','is_night','gate_session_decay','gate_open_impulse','trade_day'
]

if __name__ == "__main__":
    DATA_DIR = "/home/zyyuan/project2/feature_generate_all_np_ag_v5_intern_parquet"
    MODEL_DIR = "./processed_data" 
    START_DATE = "20250808"
    LABEL_COL = 'prj2_1_label'
    COST_COL = 'LABEL_CAL_DQ_inst1_60'
    WINDOW_SIZE = 120 # 你的窗口大小
    
    # 1. 初始化模型
    pipeline = ModelInference(MODEL_DIR, device='cuda')
    
    # 2. 获取文件
    files = get_filtered_files(DATA_DIR, START_DATE)
    
    all_results_list = []
    
    for f in tqdm(files, desc="Inference"):
        try:
            # A. 读取全量数据
            df = pd.read_parquet(f)
            df = create_timestamp_fast(df)
            
            # B. 计算时间特征 (全量)
            time_objs = df['timestamp'].dt.time
            seconds = np.array([t.hour * 3600 + t.minute * 60 + t.second for t in time_objs])
            df['is_night'] = 0.0; df['is_am'] = 0.0; df['is_pm'] = 0.0
            decay_minutes = np.full(len(df), 9999.0)
            impulse_minutes = np.full(len(df), 9999.0)
            
            mask_night_1 = (seconds >= 75600)
            if np.any(mask_night_1):
                dt = seconds[mask_night_1] - 75600
                decay_minutes[mask_night_1] = dt / 60.0
                impulse_minutes[mask_night_1] = dt / 60.0
            mask_night_2 = (seconds < 28800)
            if np.any(mask_night_2):
                dt = seconds[mask_night_2] + (24 * 3600 - 75600)
                decay_minutes[mask_night_2] = dt / 60.0
                impulse_minutes[mask_night_2] = dt / 60.0
            df.loc[mask_night_1 | mask_night_2, 'is_night'] = 1.0
            
            mask_am = (seconds >= 32400) & (seconds < 41400)
            if np.any(mask_am):
                curr_secs = seconds[mask_am]
                decay_minutes[mask_am] = (curr_secs - 32400) / 60.0
                is_after_break = (curr_secs >= 37800)
                imp_mins = (curr_secs - 32400) / 60.0
                imp_mins[is_after_break] = (curr_secs[is_after_break] - 37800) / 60.0
                impulse_minutes[mask_am] = imp_mins
            df.loc[mask_am, 'is_am'] = 1.0
            
            mask_pm = (seconds >= 48600) & (seconds <= 15*3600)
            if np.any(mask_pm):
                dt = seconds[mask_pm] - 48600
                decay_minutes[mask_pm] = dt / 60.0
                impulse_minutes[mask_pm] = dt / 60.0
            df.loc[mask_pm, 'is_pm'] = 1.0
            
            df['gate_session_decay'] = 1.0 / np.sqrt(decay_minutes + 1.0)
            df['gate_open_impulse'] = np.exp(- (impulse_minutes ** 2) / (2 * 10.0 ** 2))
            
            # C. 计算 Market Features (全量连续计算) - [新增步骤]
            # 计算得到的 df_market 包含 PSY_60 等指标，行数与 df 一致
            df_market = calculate_market_features_full(df)
            
            # 将 Market Features 合并回主 DataFrame
            # (这里直接 concat 因为索引是一样的)
            df = pd.concat([df, df_market], axis=1)
            
            # D. VAE 数据准备 (全量标准化)
            # 仅标准化 VAE 需要的那些列
            raw_feature_cols = get_feature_columns(df, CONFIG_EXCLUDE + pipeline.market_factors) 
            # 也可以直接用 pipeline.vae_input_cols (这是更严谨的做法，保证与训练一致)
            df_scaled = df.copy()
            df_scaled[pipeline.scale_cols] = pipeline.scaler.transform(df_scaled[pipeline.scale_cols].fillna(0).values).astype(np.float32)
            
            # 获取全量特征矩阵 (numpy array)，供后续 Batch 提取使用
            full_vae_input_matrix = df_scaled[pipeline.vae_input_cols].values
            
            # E. 采样 (Sampling)
            indices = np.arange(WINDOW_SIZE - 1, len(df), 60)
            if len(indices) == 0: continue
            
            # 提取采样点的元数据 + Market Features + Time Features
            # 注意: feat_mean (均值特征) 如果你需要，也要在这里计算并提取
            # 这里假设你训练模型时用了 'raw_feature_cols' 的 rolling mean
            # 为了简单，这里省略 feat_mean 的计算，直接用当前值，或者你需要加回 rolling mean 逻辑
            # 根据你之前的代码逻辑，这里似乎不需要 raw feature mean，只需要 Market Features?
            # 修正: 如果 LGBM 需要 raw_feature_mean，请取消下面注释
            # feat_mean = df[raw_feature_cols].rolling(window=WINDOW_SIZE).mean()
            # chunk_mean = feat_mean.iloc[indices].copy().add_suffix('_mean')
            
            # 提取采样后的 DataFrame
            meta_cols = [LABEL_COL, 'timestamp', COST_COL] + pipeline.context_factors + pipeline.market_factors
            chunk = df[meta_cols].iloc[indices].copy()
            
            # F. VAE 特征提取 (仅针对采样点)
            chunk = pipeline.extract_vae_features_batch(chunk, full_vae_input_matrix, indices, pipeline.vae_cfg['data']['seq_len'])
            
            # G. 预测
            # chunk 此时包含了: Market Features, Time Features, VAE Latents
            preds = pipeline.predict_mixed(chunk)
            
            # 收集结果
            chunk['factor_pred'] = preds
            res_df = chunk[['timestamp', LABEL_COL, COST_COL, 'factor_pred']].dropna()
            all_results_list.append(res_df)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            import traceback
            traceback.print_exc()

    # 合并与回测 (同之前逻辑)
    if all_results_list:
        full_pred_df = pd.concat(all_results_list, axis=0, ignore_index=True)
        # calculate_dq_metrics...