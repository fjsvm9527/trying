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

class SelfSupervisedVAE(nn.Module):
    def __init__(self, encoder_config, num_factors, seq_len, hidden_dim, latent_dim=1):
        super(SelfSupervisedVAE, self).__init__()
        model_type = encoder_config.get('type', 'gru').lower()
        print(f"[Model] Initializing VAE with Encoder Backbone: {model_type.upper()}")

        # 1. 动态实例化 Encoder Backbone
        if model_type == 'gru':
            self.backbone = GRUBackbone(
                num_factors=num_factors, 
                hidden_dim=hidden_dim, 
                num_layers=encoder_config.get('num_layers', 1),
                dropout=encoder_config.get('dropout', 0.0)
            )
        elif model_type == 'lstm':
            self.backbone = LSTMBackbone(
                num_factors=num_factors, 
                hidden_dim=hidden_dim, 
                num_layers=encoder_config.get('num_layers', 1),
                dropout=encoder_config.get('dropout', 0.0)
            )
        elif model_type == 'tsmixer':
            self.backbone = TSMixerBackbone(
                num_factors=num_factors, 
                seq_len=seq_len, 
                hidden_dim=hidden_dim,
                ff_dim=encoder_config.get('ff_dim', 64),
                num_blocks=encoder_config.get('num_blocks', 2)
            )
        elif model_type == 'dlinear':
            self.backbone = DLinearBackbone(
                seq_len=seq_len,
                num_factors=num_factors,
                hidden_dim=hidden_dim,
                kernel_size=encoder_config.get('kernel_size', 25)
            )
        else:
            raise ValueError(f"Unknown encoder type: {model_type}")

        # 2. VAE 投影层 (统一接口)
        # 无论 Backbone 是什么，输出都是 [Batch, hidden_dim]
        # 然后在这里统一映射到 mu 和 log_var
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # 3. Decoder (重构层)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_factors) # 重构最后一刻的特征
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 
        # 1. 骨干网络提取特征 h
        h = self.backbone(x)
        
        # 2. 映射分布参数
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        # 3. 采样
        z = self.reparameterize(mu, log_var)
        
        # 4. 重构
        recon_x = self.decoder_net(z)
        
        return recon_x, mu, log_var

    def run_model(self, x, gamma=0.01):
        recon_x, mu, log_var = self.forward(x)
        target = x[:, -1, :] # Target: Last Step
        
        recon_loss = F.mse_loss(recon_x, target)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / x.size(0)
        
        total_loss = recon_loss + gamma * kl_loss
        return total_loss, recon_loss, kl_loss

    def extract_latent(self, x):
        with torch.no_grad():
            h = self.backbone(x)
            mu = self.fc_mu(h)
        return mu

# ==============================================================================
# 2. 推理引擎类 (封装所有模型加载与预测逻辑)
# ==============================================================================
class ModelInference:
    def __init__(self, model_dir, device='cuda'):
        self.device = torch.device(device)
        print(f">>> Loading artifacts from {model_dir}...")
        
        # 1. 加载标准化器
        self.scaler = joblib.load('/home/zyyuan/project2/processed_data/scaler.pkl')
        self.scale_cols = joblib.load('/home/zyyuan/project2/processed_data/feature_cols.pkl')
        
        # 2. 加载 VAE
        with open(os.path.join(model_dir, '/home/zyyuan/project2/config/20251229_VAE_lstm.json'), 'r') as f:
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
        
        # 3. 加载 LightGBM
        self.lgbm_model = lgb.Booster(model_file=os.path.join(model_dir, '/home/zyyuan/project2/lgbm_mixup_top300_model.txt'))
        self.context_factors = ['is_pm','is_am','is_night','gate_session_decay','gate_open_impulse']
        self.vae_factors = [f'vae_latent_{i}' for i in range(64)]
        self.lgbm_features = self.context_factors+self.vae_factors

    def preprocess(self, df):
        """标准化处理"""
        # 确保包含所有需要的列，缺失填0
        df = df.copy()
        # 这里假设输入的 df 已经有了 time-based features (is_night, etc.)
        # 如果没有，你需要在这里调用 calculate_time_features(df)
        
        # Z-Score 标准化
        df[self.scale_cols] = df[self.scale_cols].fillna(0)
        df[self.scale_cols] = self.scaler.transform(df[self.scale_cols].values).astype(np.float32)
        return df

    def extract_vae_features(self, df):
        """VAE 特征提取"""
        seq_len = self.vae_cfg['data']['seq_len']
        feature_matrix = df[self.vae_input_cols].values
        
        # 使用 Edge Padding 解决冷启动，保证输出长度与 df 一致
        pad_width = ((seq_len - 1, 0), (0, 0))
        padded_features = np.pad(feature_matrix, pad_width, mode='edge')
        
        # 滑动窗口
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded_features, window_shape=seq_len, axis=0)
        windows = windows.transpose(0, 2, 1) # (N, Seq, Feat)
        
        # 推理
        tensor_data = torch.from_numpy(windows).float().to(self.device)
        batch_size = 2048
        latents_list = []
        
        with torch.no_grad():
            for i in range(0, len(tensor_data), batch_size):
                batch = tensor_data[i : i+batch_size]
                mu = self.vae_model.extract_latent(batch)
                latents_list.append(mu.cpu().numpy())
        
        latents = np.concatenate(latents_list, axis=0)
        
        # 拼接到 DataFrame
        for i in range(self.vae_cfg['model']['latent_dim']):
            df[f'vae_latent_{i}'] = latents[:, i]
            
        return df

    def predict(self, df):
        """全流程预测"""
        df = self.preprocess(df)
        df = self.extract_vae_features(df)
        
        # 这里的列顺序必须与 LGBM 训练时严格一致
        X = df[self.lgbm_features]
        preds = self.lgbm_model.predict(X)
        return preds

# ==============================================================================
# 3. 核心流程：文件筛选、读取与预测
# ==============================================================================

def get_filtered_files(data_dir, start_date_str):
    """
    获取目录下大于等于 start_date 的所有 parquet 文件
    文件名格式假设包含: 20250808 这样的日期
    """
    all_files = glob.glob(os.path.join(data_dir, "*.parquet")) # 或 *_features.parquet
    target_files = []
    
    start_date = int(start_date_str)
    
    for f in all_files:
        filename = os.path.basename(f)
        # 正则提取8位日期
        match = re.search(r'(20\d{6})', filename)
        if match:
            file_date = int(match.group(1))
            if file_date >= start_date:
                target_files.append((file_date, f))
    
    # 按日期排序，保证回测顺序
    target_files.sort(key=lambda x: x[0])
    return [x[1] for x in target_files]

def calculate_dq_metrics(df, pred_col='factor_pred', label_col='prj2_1_label', cost_col='LABEL_CAL_DQ_inst1_60'):
    """
    统一回测 DQ 计算逻辑 (复用你之前的代码)
    """
    print(f"\n>>> 开始计算回测指标 (Total Samples: {len(df)})...")
    
    thresholds = np.arange(0.0, 10.0, 0.05) # 步长稍微放大一点加快速度
    max_dq = -np.inf
    best_res = {}
    results = []

    for threshold in thresholds:
        buy_mask = df[pred_col] > threshold
        sell_mask = df[pred_col] < -1 * threshold
        
        # 计算 PnL (扣除成本)
        # 注意：这里假设 cost_col 已经是正数，且公式为你提供的 (0.5/10000)*cost
        cost_buy = (0.5/10000) * df.loc[buy_mask, cost_col]
        cost_sell = (0.5/10000) * df.loc[sell_mask, cost_col]
        
        pnl_buy = df.loc[buy_mask, label_col] - cost_buy
        pnl_sell = -1 * df.loc[sell_mask, label_col] - cost_sell
        
        dq_buy = np.sum(pnl_buy)
        dq_sell = np.sum(pnl_sell)
        dq_total = dq_buy + dq_sell
        
        count = len(pnl_buy) + len(pnl_sell)
        if count == 0: continue
            
        final_dq = 15 * dq_total # 你的系数
        
        # 计算 DQR
        abs_move = np.abs(pnl_buy).sum() + np.abs(pnl_sell).sum()
        dq_neg = (abs_move - dq_total) / 2.0
        dqr = (dq_total + dq_neg) / dq_neg if dq_neg != 0 else 0
        
        res_item = {
            'Threshold': threshold,
            'DQ': final_dq,
            'DQR': dqr,
            'Count': count,
            'Avg_PnL': dq_total / count if count > 0 else 0
        }
        results.append(res_item)
        
        if final_dq > max_dq:
            max_dq = final_dq
            best_res = res_item

    res_df = pd.DataFrame(results)
    print("\n========= Backtest Best Result =========")
    print(pd.Series(best_res))
    print("========================================")
    
    return res_df

def create_timestamp_fast(df):
    """生成时间戳 (保持不变)"""
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

CONFIG = {
    'data_dir': '/home/zyyuan/project2/feature_generate_all_np_ag_v5_intern_parquet', 
    'save_dir': './processed_data',
    'split_date': '20250808',       
    'window_size': 120,                
    'label_col': 'prj2_1_label',      
    
    # -----------------------------------------------------------
    # [新增] 指定需要计算"振幅"和"趋势"的高级特征列表
    # 如果列表为空，则不对任何特征计算高级指标，仅计算均值
    # 请在这里填入你认为最重要的因子列名
    # -----------------------------------------------------------
    'enhanced_features': ['f_35_7', 'f_4_117', 'f_84_37', 'f_31_28', 'f_84_38', 'f_7_32', 'f_7_30', 'f_92_383', 'f_74_362', 'f_33_28', 'f_35_5', 'f_75_239', 'f_8_29', 'f_4_120', 'f_92_322', 'f_35_6', 'f_87_45', 'f_93_234', 'f_5_27', 'f_92_326', 'f_85_40', 'f_63_158', 'f_6_104', 'f_31_24', 'f_93_241', 'f_36_7', 'f_93_224', 'f_85_37', 'f_7_29', 'f_6_112', 'f_75_203', 'f_5_28', 'f_6_85', 'f_77_7', 'f_75_238', 'f_5_26', 'f_85_39', 'f_79_35', 'f_17_35', 'f_44_7', 'f_86_46', 'f_63_6', 'f_64_159', 'f_87_43', 'f_34_7', 'f_17_41', 'f_84_33', 'f_79_31', 'f_32_28', 'f_71_77', 'f_79_32', 'f_19_31', 'f_4_119', 'f_18_30', 'f_52_34', 'f_43_7', 'f_90_44', 'f_55_42', 'f_47_6', 'f_17_24', 'f_73_83', 'f_53_12', 'f_34_5', 'f_20_47', 'f_43_6', 'f_91_44', 'f_64_158', 'f_8_25', 'f_74_361', 'f_18_37', 'f_43_5', 'f_16_46', 'f_20_38', 'f_64_98', 'f_89_24', 'f_52_8', 'f_32_5', 'f_53_28', 'f_55_40', 'f_53_34', 'f_86_47', 'f_80_18', 'f_73_78', 'f_73_77', 'f_61_21', 'f_32_8', 'f_63_134', 'f_56_28', 'f_74_384', 'f_18_25', 'f_40_7', 'f_72_76', 'f_72_82', 'f_55_30', 'f_91_48', 'f_19_38', 'f_20_42', 'f_16_45', 'f_3_39', 'f_1_42', 'f_89_23', 'f_8_32', 'f_78_7', 'f_19_41', 'f_3_32', 'f_48_1', 'f_51_34', 'f_3_37', 'f_91_47', 'f_87_42', 'f_89_20', 'f_61_8', 'f_54_37', 'f_34_4', 'f_27_5', 'f_33_5', 'f_88_23', 'f_2_41', 'f_56_39', 'f_90_43', 'f_37_6', 'f_52_41', 'f_71_78', 'f_80_21', 'f_56_31', 'f_33_9', 'f_61_12', 'f_45_7', 'f_39_1', 'f_50_15', 'f_71_84', 'f_14_39', 'f_45_6', 'f_11_7', 'f_80_16', 'f_14_24', 'f_48_34', 'f_51_37', 'f_90_48', 'f_13_9', 'f_38_7', 'f_62_4', 'f_51_10', 'f_47_5', 'f_72_16', 'f_62_2', 'f_60_2', 'f_24_6', 'f_27_2', 'f_2_37', 'f_15_44', 'f_65_41', 'f_60_4', 'f_12_37', 'f_14_21', 'f_31_22', 'f_12_39', 'f_59_5', 'f_67_41', 'f_27_7', 'f_62_19', 'f_13_8', 'f_30_6', 'f_47_2', 'f_12_5', 'f_28_2', 'f_60_9', 'f_68_1', 'f_42_7', 'f_39_7', 'f_67_42', 'f_59_2', 'f_86_48', 'f_70_1', 'f_16_14', 'f_54_30', 'f_26_2', 'f_49_37', 'f_70_4', 'f_81_3', 'f_38_5', 'f_37_3', 'f_49_34', 'f_42_2', 'f_2_35', 'f_59_17', 'f_49_4', 'f_37_4', 'f_70_6', 'f_15_47', 'f_81_12', 'f_10_3', 'f_29_6', 'f_69_3', 'f_67_16', 'f_50_2', 'f_1_30', 'f_88_13', 'f_54_39', 'f_15_20', 'f_48_23', 'f_25_1', 'f_77_5', 'f_13_14', 'f_88_11', 'f_44_4', 'f_65_35', 'f_76_5', 'f_78_3', 'f_50_6'],

    'exclude_cols': [
        'ExchActionDay', 'ExchUpdateTime', 'ExchUpdateMillisec', 'nano', 'mid',
        'last', 'vwap', 'bv1', 'bp1', 'ap1', 'av1', 'triggerInst',
        'triggerInst.volume', 'prj2_2_label', 'prj2_1_label',
        'LABEL_CAL_DQ_inst1_60', 'LABEL_CAL_DQ_inst1_900', 'featSite','is_pm','is_am','is_night','gate_session_decay','gate_open_impulse','trade_day'
    ]
}
def get_feature_columns(df):
    """获取纯数值特征列"""
    cols = [c for c in df.columns if c not in CONFIG['exclude_cols'] and c != 'timestamp']
    numeric_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return sorted(numeric_cols)
if __name__ == "__main__":
    # --- 配置 ---
    DATA_DIR = "/home/zyyuan/project2/feature_generate_all_np_ag_v5_intern_parquet"
    MODEL_DIR = "./processed_data" # 存放 scaler.pkl, best_ss_vae.pth, lgbm.txt 的目录
    START_DATE = "20250808"        # 测试开始日期
    LABEL_COL = 'prj2_1_label'
    COST_COL = 'LABEL_CAL_DQ_inst1_60'
    
    # 1. 初始化模型
    pipeline = ModelInference(MODEL_DIR, device='cuda')
    
    # 2. 获取文件列表
    files = get_filtered_files(DATA_DIR, START_DATE)
    print(f"筛选出 {len(files)} 个待测试文件 (Start Date: {START_DATE})")
    
    # 3. 循环推理并收集结果
    all_results_list = []
    
    # 需要保留的列 (用于最终计算 DQ)
    keep_cols = ['timestamp',LABEL_COL, COST_COL]
    window = CONFIG['window_size']
    for f in tqdm(files, desc="Inference"):
        try:
            df = pd.read_parquet(f)
            df = create_timestamp_fast(df)
            raw_feature_cols = get_feature_columns(df)
            time_objs = df['timestamp'].dt.time
            seconds = np.array([t.hour * 3600 + t.minute * 60 + t.second for t in time_objs])
            df['is_night'] = 0.0
            df['is_am'] = 0.0
            df['is_pm'] = 0.0
            decay_minutes = np.full(len(df), 9999.0)
            impulse_minutes = np.full(len(df), 9999.0)
            mask_night_1 = (seconds >= 75600)
            if np.any(mask_night_1):
                dt = seconds[mask_night_1] - 75600
                decay_minutes[mask_night_1] = dt / 60.0
                impulse_minutes[mask_night_1] = dt / 60.0
            mask_night_2 = (seconds < 28800) # < 08:00
            if np.any(mask_night_2):
                dt = seconds[mask_night_2] + (24 * 3600 - 75600)
                decay_minutes[mask_night_2] = dt / 60.0
                impulse_minutes[mask_night_2] = dt / 60.0
            df.loc[mask_night_1,'is_night'] = 1.0
            df.loc[mask_night_2,'is_night'] = 1.0
            mask_am = (seconds >= 32400) & (seconds < 41400) # 09:00 - 11:30
            if np.any(mask_am):
                curr_secs = seconds[mask_am]
                decay_minutes[mask_am] = (curr_secs - 32400) / 60.0
                is_after_break = (curr_secs >= 37800)
                imp_mins = (curr_secs - 32400) / 60.0
                imp_mins[is_after_break] = (curr_secs[is_after_break] - 37800) / 60.0
                impulse_minutes[mask_am] = imp_mins
            df.loc[mask_am,'is_am'] = 1.0
            mask_pm = (seconds >= 48600) & (seconds <= 15*3600)
            if np.any(mask_pm):
                dt = seconds[mask_pm] - 48600
                decay_minutes[mask_pm] = dt / 60.0
                impulse_minutes[mask_pm] = dt / 60.0
            
            df.loc[mask_pm,'is_pm'] = 1.0
            sigma = 10.0
            df['gate_session_decay'] = 1.0 / np.sqrt(decay_minutes + 1.0)
            df['gate_open_impulse'] = np.exp(- (impulse_minutes ** 2) / (2 * sigma ** 2)) 
            
            df_all_feats = df[raw_feature_cols]
            feat_mean = df_all_feats.rolling(window=window).mean()
            
            # 2. 高级特征 (振幅 & 趋势): 仅对 target_enhanced_cols 计算
            # 如果列表为空，跳过计算节省时间
            # feat_amp = None
            # feat_trend = None
            # if len(target_enhanced_cols) > 0:
            #     df_target = df[target_enhanced_cols]
                
            #     # 振幅 (耗时操作)
            #     q95 = df_target.rolling(window=window).quantile(0.95)
            #     q05 = df_target.rolling(window=window).quantile(0.05)
            #     feat_amp = q95 - q05
                
            #     # 趋势
            #     ma_head_tail = df_target.rolling(window=trend_k).mean()
            #     feat_trend = ma_head_tail - ma_head_tail.shift(window - trend_k)
            indices = np.arange(window - 1, len(df), 60)
            if len(indices) == 0: continue

            concat_list = []

            # A. 添加均值 (所有特征)
            s_mean = feat_mean.iloc[indices].copy()
            s_mean.columns = [f"{c}_mean" for c in raw_feature_cols]
            concat_list.append(s_mean)
            
            # D. Metadata
            meta_cols = [CONFIG['label_col'], 'timestamp','LABEL_CAL_DQ_inst1_60','is_pm','is_am','is_night','gate_session_decay','gate_open_impulse']
            s_meta = df[meta_cols].iloc[indices].copy()
            concat_list.append(s_meta)
            
            # 合并
            chunk = pd.concat(concat_list, axis=1)
            chunk.dropna(inplace=True)
            preds = pipeline.predict(chunk)
            print(preds)
            # 收集结果
            res_df = chunk[keep_cols].copy()
            res_df['factor_pred'] = preds
            
            # 简单过滤：去掉无法计算 label 的行 (例如尾部数据)
            res_df = res_df.dropna(subset=[LABEL_COL, COST_COL])
            
            all_results_list.append(res_df)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # 4. 合并所有预测结果
    if len(all_results_list) > 0:
        full_pred_df = pd.concat(all_results_list, axis=0, ignore_index=True)
        print(f"总预测样本数: {len(full_pred_df)}")
        
        # 5. 统一回测
        dq_results = calculate_dq_metrics(full_pred_df, 
                                          pred_col='factor_pred', 
                                          label_col=LABEL_COL, 
                                          cost_col=COST_COL)
        
        # 保存回测详情
        dq_results.to_csv("final_backtest_metrics.csv", index=False)
        full_pred_df.to_pickle("final_predictions.pkl") # 保存原始预测结果以便后续分析
        
    else:
        print("未生成任何预测结果，请检查文件路径或日期设置。")