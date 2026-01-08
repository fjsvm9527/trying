import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 6,
    'feature_fraction': 0.1,
    'bagging_fraction': 0.8, 
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'n_jobs': -1,
    'verbose': -1,
    'seed': 42
}

def train_lgbm_and_select_features(train_df, valid_df, label_col='prj2_1_label'):
    """
    训练 LightGBM 并返回特征重要性列表
    """
    # 1. 准备特征列表 (排除 Label 和 Timestamp)
    exclude_cols = ['timestamp', label_col]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    print(f"参与训练的特征数量: {len(feature_cols)}")
    
    # 2. 构建 LightGBM Dataset
    # 使用 reference 可以节省内存并对齐 bin
    train_data = lgb.Dataset(
        train_df[feature_cols], 
        label=train_df[label_col], 
        feature_name=feature_cols
    )
    
    valid_data = lgb.Dataset(
        valid_df[feature_cols], 
        label=valid_df[label_col], 
        feature_name=feature_cols,
        reference=train_data 
    )
    
    # 3. 训练模型
    print(">>> 开始训练 LightGBM...")
    model = lgb.train(
        LGB_PARAMS,
        train_data,
        num_boost_round=2000,           # 最大轮数
        valid_sets=[valid_data],        # 验证集
        callbacks=[
            lgb.early_stopping(stopping_rounds=50), # 100轮不提升则停止
            lgb.log_evaluation(period=10)            # 每50轮打印一次
        ]
    )
    
    # 4. 提取特征重要性
    print("\n>>> 计算特征重要性...")
    importance_gain = model.feature_importance(importance_type='gain') # 增益 (推荐)
    importance_split = model.feature_importance(importance_type='split') # 分裂次数
    
    feature_imp = pd.DataFrame({
        'feature': feature_cols,
        'gain': importance_gain,
        'split': importance_split
    })
    
    # 按 Gain 降序排列
    feature_imp = feature_imp.sort_values('gain', ascending=False).reset_index(drop=True)
    
    # 计算归一化重要性 (占比)
    feature_imp['gain_normalized'] = feature_imp['gain'] / feature_imp['gain'].sum()
    
    # 5. 在验证集上评估 IC (Information Coefficient)
    print("\n>>> 评估验证集 IC...")
    valid_preds = model.predict(valid_df[feature_cols], num_iteration=model.best_iteration)
    valid_df['pred'] = valid_preds
    thresholds = np.arange(0.0, 10.0, 0.01)
    max_dq = -np.inf
    results = []
    for threshold in thresholds:
            buy_mask = valid_df['pred'] > threshold
            sell_mask = valid_df['pred'] <  -1*threshold
            pnl_buy_raw = valid_df.loc[buy_mask,'prj2_1_label'] - 0.45
            pnl_sell_raw = -1*valid_df.loc[sell_mask,'prj2_1_label'] - 0.45
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
    if not df_res.empty:
            sns.set_theme(style="whitegrid")
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(df_res['Threshold'], df_res['DQ'], color='#2a9d8f', linewidth=2, label='DQ')
            ax2 = ax1.twinx()
            ax2.plot(df_res['Threshold'], df_res['DQR'], color='#e76f51', linestyle='--', linewidth=2, label='DQR')
            plt.title(f"Max DQ Analysis:")
            plt.show()
    valid_ic, _ = pearsonr(valid_preds, valid_df[label_col].values)
    print(f"Validation IC: {valid_ic:.4f}")
    
    return model, feature_imp

def filter_factors(feature_imp, top_n=None, threshold=0.0):
    """
    根据重要性表筛选因子
    :param top_n: 保留前 N 个
    :param threshold: 保留 gain > threshold 的因子
    """
    if top_n:
        selected = feature_imp.head(top_n)
    elif threshold > 0:
        selected = feature_imp[feature_imp['gain'] > threshold]
    else:
        selected = feature_imp[feature_imp['gain'] > 0] # 默认去除 0 重要性的
        
    print(f"筛选结果: 从 {len(feature_imp)} -> 保留 {len(selected)} 个因子")
    return selected['feature'].tolist()

def plot_importance(feature_imp, top_n=20):
    """画图"""
    plt.figure(figsize=(10, 8))
    sns.barplot(x="gain", y="feature", data=feature_imp.head(top_n))
    plt.title(f'Top {top_n} Features by LightGBM Gain')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_df = pd.read_pickle("traindata.pkl")
    valid_df = pd.read_pickle("validdata.pkl")
    if True:
        bst, imp_df = train_lgbm_and_select_features(train_df, valid_df)
        imp_df.to_csv("feature_importance.csv", index=False)
        print("特征重要性已保存至 feature_importance.csv")
        top_factors = filter_factors(imp_df, top_n=100)
        valid_factors = filter_factors(imp_df, threshold=0.0001)
        plot_importance(imp_df, top_n=20)