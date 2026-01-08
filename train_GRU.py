import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import StandardScaler
import json

# =========================================================================
#  1. 核心工具：数据预处理 (保持不变)
# =========================================================================
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
    
def prepare_tensor_data(df, feature_cols, label_col, seq_len=30):
    """
    将 DataFrame 转换为 3D 时序张量 (Batch, Seq, Feat)
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
        feature_vals = group[feature_cols].values.astype(np.float32)
        label_vals = group[label_col].values.astype(np.float32)
        
        # 头部填充 (Padding)
        pad_width = ((seq_len - 1, 0), (0, 0)) 
        padded_features = np.pad(feature_vals, pad_width, mode='edge')
        
        # 向量化滑动窗口
        windows_raw = np.lib.stride_tricks.sliding_window_view(padded_features, window_shape=seq_len, axis=0)
        
        # 交换维度 (Batch, Features, Seq_Len) -> (Batch, Seq_Len, Features)
        windows = windows_raw.transpose(0, 2, 1)
        
        X_daily_list.append(windows)
        y_daily_list.append(label_vals)

    if not X_daily_list:
        raise ValueError("数据为空或处理失败")

    X_all = np.concatenate(X_daily_list, axis=0) 
    y_all = np.concatenate(y_daily_list, axis=0) 

    print(f">>> [Data] 生成完成! X: {X_all.shape}, y: {y_all.shape}")
    return torch.from_numpy(X_all), torch.from_numpy(y_all)

# =========================================================================
#  2. 模型架构: GRU Predictor (去除 VAE，改为直接回归)
# =========================================================================

class GRUPredictor(nn.Module):
    def __init__(self, num_factors, seq_len, hidden_dim, num_layers=2, dropout=0.1):
        super(GRUPredictor, self).__init__()
        
        # [Image of GRU architecture]
        # 1. GRU 特征提取层 (Encoder)
        self.gru = nn.GRU(
            input_size=num_factors,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 2. 特征变换层 (可选，增加非线性)
        self.feature_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. 回归预测头 (Regressor Head)
        # 直接将 Hidden State 映射到 Label
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1) # 输出单标量
        )
        
    def forward(self, x):
        # x: [Batch, Seq_Len, Features]
        
        # GRU Forward
        # output: [Batch, Seq_Len, Hidden]
        # h_n:    [Num_Layers, Batch, Hidden]
        output, _ = self.gru(x)
        
        # 取最后一个时间步的输出 (Many-to-One)
        last_step_feature = output[:, -1, :] 
        
        # 特征变换
        h = self.feature_fc(last_step_feature)
        
        # 回归预测
        pred = self.regressor(h)
        
        # Squeeze transform [Batch, 1] -> [Batch]
        return pred.squeeze(-1)

# =========================================================================
#  3. 辅助函数
# =========================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(model, loader, device):
    """评估 IC"""
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            # 直接预测
            y_pred = model(x_batch)
            
            preds.append(y_pred.cpu().numpy())
            trues.append(y_batch.numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    # 计算 IC
    pearson_ic, _ = pearsonr(preds, trues)
    spearman_ic, _ = spearmanr(preds, trues)
    
    return pearson_ic, spearman_ic, preds, trues

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# =========================================================================
#  4. 主程序
# =========================================================================

def main():
    # --- 配置 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args_cmd = parser.parse_args()
    
    cfg = load_config(args_cmd.config)
    train_path = cfg['paths']['train_data']
    valid_path = cfg['paths']['valid_data']
    
    # 数据参数
    label_col = cfg['data']['label_col']
    seq_len = cfg['data']['seq_len']
    
    # 训练参数
    batch_size = cfg['training']['batch_size']
    epochs = cfg['training']['epochs']
    lr = cfg['training']['learning_rate']
    
    # 设备与种子
    set_seed(cfg['training']['seed'])
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    # --- 数据加载 ---
    print(">>> 加载 Pickle 数据...")
    train_df = pd.read_pickle(train_path)
    valid_df = pd.read_pickle(valid_path)
    
    # Label 标准化
    label_scaler = StandardScaler()
    print("Standardizing Label...")
    train_df[label_col] = label_scaler.fit_transform(train_df[[label_col]].values)
    valid_df[label_col] = label_scaler.transform(valid_df[[label_col]].values)
    
    # 特征选择 (读取重要性文件)
    # 假设路径正确，如果不正确请修改
    try:
        imp_path = "/home/zyyuan/project2/feature_importance.csv"
        imp_df = pd.read_csv(imp_path)
        top_n = 300
        feature_cols = imp_df.head(top_n)['feature'].tolist()
        print(f"使用 Feature Importance 前 {top_n} 个特征")
    except Exception as e:
        print(f"无法读取特征重要性文件 ({e})，尝试自动识别特征...")
        exclude = ['trade_day', 'timestamp', label_col, 'ExchActionDay', 'ExchUpdateTime']
        
    print(f"特征数量: {len(feature_cols)}")
    
    # 生成时序张量
    X_train, y_train = prepare_tensor_data(train_df, feature_cols, label_col, seq_len)
    X_valid, y_valid = prepare_tensor_data(valid_df, feature_cols, label_col, seq_len)
    
    # 构建 DataLoader
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 模型初始化 ---
    print(">>> 初始化 GRU 模型...")
    # 使用纯 GRU 模型，不再是 VAE
    model = GRUPredictor(
        num_factors=len(feature_cols),
        seq_len=seq_len,
        hidden_dim=128,
        num_layers=1, # 双层 GRU
        dropout=0.1
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # 也可以换成 CCCLoss
    # criterion = CCCLoss()

    # --- 训练循环 ---
    print("\n>>> 开始训练...")
    best_ic = -999
    
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(x_batch)
            
            # Loss Calculation (没有 KL Loss 了，只有 MSE)
            loss = criterion(preds, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
        # --- 验证阶段 ---
        p_ic, s_ic, _, _ = evaluate_model(model, valid_loader, device)
        avg_loss = train_loss_sum / len(train_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.5f} | Valid IC (P/S): {p_ic:.4f} / {s_ic:.4f}")
        
        # 保存最佳模型
        if p_ic > best_ic:
            best_ic = p_ic
            torch.save(model.state_dict(), "best_gru_model.pth")
            print(">>> Model Saved (Best RankIC)!")

    # --- 最终测试与 DQ 分析 ---
    print("\n>>> 训练结束。最终测试...")
    model.load_state_dict(torch.load("best_gru_model.pth"))
    p_ic, s_ic, preds, trues = evaluate_model(model, valid_loader, device)
    print(f"Final Test IC -> Pearson: {p_ic:.6f}, Spearman: {s_ic:.6f}")
    
    # --- DQ 分析 (保持你的原逻辑) ---
    valid_df_res = pd.DataFrame({'pred': preds, 'true': trues})
    # 因为 label 之前被标准化了，如果 DQ 需要真实金额概念，理论上这里应该 inverse_transform 回去
    # 但根据你的代码逻辑，你是直接乘 1000 放大的，这里保持一致
    valid_df_res['pred'] = valid_df_res['pred']*10
    
    thresholds = np.arange(0.0, 10.0, 0.01)
    max_dq = -np.inf
    results = []
    
    print(">>> Calculating DQ metrics...")
    for threshold in thresholds:
        buy_mask = valid_df_res['pred'] > threshold
        sell_mask = valid_df_res['pred'] < -1 * threshold
        
        pnl_buy_raw = valid_df_res.loc[buy_mask, 'true']
        pnl_sell_raw = -1 * valid_df_res.loc[sell_mask, 'true']
        
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
        # dq_pos = DQ + dq_neg # (Unused variable logic kept from original)
        dq_pos = (abs_move + DQ) / 2.0 # Fixed based on logic, but preserving original flow

        dqr = dq_pos / dq_neg if dq_neg != 0 else 0
        
        results.append({
            'Threshold': threshold, 
            'DQ': final_DQ, 
            'DQR': dqr, 
            'Count': count
        })
        
    df_res = pd.DataFrame(results)
    # df_res['DQ'] = df_res['DQ'] - 15*0.44*df_res['Count']
    df_res = df_res[(df_res['Count'] > 50) & (df_res['DQ'] > 0)]
    df_res.to_csv("test.csv")
    if not df_res.empty:
        sns.set_theme(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(df_res['Threshold'], df_res['DQ'], color='#2a9d8f', linewidth=2, label='DQ')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('DQ', color='#2a9d8f')
        
        ax2 = ax1.twinx()
        ax2.plot(df_res['Threshold'], df_res['DQR'], color='#e76f51', linestyle='--', linewidth=2, label='DQR')
        ax2.set_ylabel('DQR', color='#e76f51')
        
        plt.title(f"Max DQ Analysis (GRU Model)")
        plt.savefig("DQ_GRU.jpg")
        print("DQ plot saved to DQ_GRU.jpg")

if __name__ == "__main__":
    main()