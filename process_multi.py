import pandas as pd
import numpy as np
import glob
import os
import re
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# ================= 配置 =================
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
        'LABEL_CAL_DQ_inst1_60', 'LABEL_CAL_DQ_inst1_900', 'featSite','is_pm','is_am','is_night','gate_session_decay','gate_open_impulse','trade_day','triggerInst.volume','rolling_amp'
    ]
}

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

def get_feature_columns(df):
    """获取纯数值特征列"""
    cols = [c for c in df.columns if c[:2]=='f_']
    numeric_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return sorted(numeric_cols)

def get_trade_day_from_filename(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'(\d{8})', filename)
    if match:
        return match.group(1)
    return None
import pandas as pd
import numpy as np


    
def process_data_to_dataframe():
    if not os.path.exists(CONFIG['save_dir']):
        os.makedirs(CONFIG['save_dir'])

    files = sorted(glob.glob(os.path.join(CONFIG['data_dir'], "*_features.parquet")))
    if not files:
        print("未找到文件")
        return None, None

    print(f"发现 {len(files)} 个文件，准备开始处理...")

    # 1. 确定特征列
    sample_df = pd.read_parquet(files[0])
    raw_feature_cols = get_feature_columns(sample_df)
    print(f"原始特征数量: {len(raw_feature_cols)}")
    
    # --- 筛选有效的高级特征 ---
    # 确保 config 里写的列名确实存在于数据中
    target_enhanced_cols = [c for c in CONFIG['enhanced_features'] if c in raw_feature_cols]
    print(f"将对其中 {len(target_enhanced_cols)} 个特征计算振幅和趋势")
    if len(target_enhanced_cols) == 0:
        print("警告: enhanced_features 为空或未匹配到任何列，将只计算均值！")

    train_chunks = []
    valid_chunks = []
    
    split_date_ts = int(CONFIG['split_date'])
    window = CONFIG['window_size']
    trend_k = 10 

    for f in tqdm(files, desc="Processing"):
        try:
            trade_day_str = get_trade_day_from_filename(f)
            if trade_day_str is None: continue

            # 读取
            req_cols = raw_feature_cols + [CONFIG['label_col'], 'ExchActionDay', 'ExchUpdateTime', 'ExchUpdateMillisec','LABEL_CAL_DQ_inst1_60','last','triggerInst.volume','av1','bv1','ap1','bp1']
            df = pd.read_parquet(f, columns=req_cols)
            df = create_timestamp_fast(df)
            df['trade_day'] = int(trade_day_str) 

            # 计算时间指标
            time_objs = df['timestamp'].dt.time
            seconds = np.array([t.hour * 3600 + t.minute * 60 + t.second for t in time_objs])
            df['is_night'] = 0.0
            df['is_am'] = 0.0
            df['is_pm'] = 0.0
            decay_minutes = np.full(len(df), 9999.0)
            impulse_minutes = np.full(len(df), 9999.0)
            
            # --- 夜盘逻辑 ---
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
            # --- 日盘逻辑 ---
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
            df['OBI_cum120'] = (df['av1']-df['bv1']).rolling(120).sum()
            df['rolling_high'] = df['last'].rolling(window=1600, min_periods=30).max()
            df['rolling_low']  = df['last'].rolling(window=1600, min_periods=30).min()
            df['rolling_amp'] = df['rolling_high'] - df['rolling_low']
            df['mid_price'] = (df['bp1'] + df['ap1']) / 2
            df.loc[df['bp1'] == 0, 'mid_price'] = df.loc[df['bp1'] == 0, 'ap1']
            df.loc[df['ap1'] == 0, 'mid_price'] = df.loc[df['ap1'] == 0, 'bp1']
            df['log_price'] = np.log(df['mid_price'])
            df['log_ret'] = df['log_price'].diff()
            df['abs_change'] = df['mid_price'].diff().abs()
            df['sq_ret'] = df['log_ret'] ** 2
            df['down_sq_ret'] = np.where(df['log_ret'] < 0, df['sq_ret'], 0)
            df['gate_rv'] = np.sqrt(df['sq_ret'].rolling(600).sum()) * 100
            down_vol = np.sqrt(df['down_sq_ret'].rolling(600).sum())
            total_vol = df['gate_rv'] / 100 # 还原回去
            df['gate_downside_ratio'] = down_vol / (total_vol + 1e-9)
            df['gate_rv_log'] = np.log(df['gate_rv'] + 1e-5)
            df['gate_range_log'] = np.log(df['rolling_amp'] + 1e-5)
            # ------------------------------------------------------------------
            # 分流计算逻辑
            # ------------------------------------------------------------------
            
            # 1. 基础特征 (均值): 对所有列计算
            # mean 计算很快，保留全量信息
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

            # ------------------------------------------------------------------
            # 采样与合并
            # ------------------------------------------------------------------
            indices = np.arange(window - 1, len(df), 60)
            if len(indices) == 0: continue

            concat_list = []

            # A. 添加均值 (所有特征)
            s_mean = feat_mean.iloc[indices].copy()
            s_mean.columns = [f"{c}_mean" for c in raw_feature_cols]
            concat_list.append(s_mean)
            
            # # # B. 添加振幅 (仅部分特征)
            # if feat_amp is not None:
            #     s_amp = feat_amp.iloc[indices].copy()
            #     s_amp.columns = [f"{c}_amp" for c in target_enhanced_cols]
            #     concat_list.append(s_amp)
            
            # # C. 添加趋势 (仅部分特征)
            # if feat_trend is not None:
            #     s_trend = feat_trend.iloc[indices].copy()
            #     s_trend.columns = [f"{c}_trend" for c in target_enhanced_cols]
            #     concat_list.append(s_trend)
            
            # D. Metadata
            meta_cols = [CONFIG['label_col'], 'timestamp', 'trade_day','triggerInst.volume','OBI_cum120','rolling_amp','LABEL_CAL_DQ_inst1_60','is_pm','is_am','is_night','gate_session_decay','gate_open_impulse','gate_rv','gate_downside_ratio','gate_rv_log','gate_range_log']
            s_meta = df[meta_cols].iloc[indices].copy()
            concat_list.append(s_meta)
            
            # 合并
            chunk = pd.concat(concat_list, axis=1)
            chunk.dropna(subset = [f"{c}_mean" for c in raw_feature_cols]+[ 'prj2_1_label', 'timestamp', 'trade_day', 'LABEL_CAL_DQ_inst1_60',
       'is_pm', 'is_am', 'is_night', 'gate_session_decay',
       'gate_open_impulse'],inplace=True)
            chunk['delta_vol_1'] = chunk['triggerInst.volume'].diff()
            chunk['delta_vol_2'] = chunk['triggerInst.volume'].diff(2)
            chunk['delta_vol_4'] = chunk['triggerInst.volume'].diff(4)
            chunk['delta_vol_8'] = chunk['triggerInst.volume'].diff(8)
            chunk['delta_vol_16'] = chunk['triggerInst.volume'].diff(16)
            # print(chunk.shape)
            # print(chunk['is_am'])
            if int(trade_day_str) < split_date_ts:
                train_chunks.append(chunk)
            else:
                valid_chunks.append(chunk)
                
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # 3. 内存拼接
    print("正在拼接 DataFrame...")
    if not train_chunks:
        print("错误：没有训练数据")
        return None, None
        
    train_df = pd.concat(train_chunks, ignore_index=True)
    valid_df = pd.concat(valid_chunks, ignore_index=True) if valid_chunks else pd.DataFrame()
    
    # 保存 Raw Data
    train_df.to_pickle(os.path.join(CONFIG['save_dir'], "traindata_origin_new.pkl"))
    valid_df.to_pickle(os.path.join(CONFIG['save_dir'], "validdata_origin_new.pkl"))
    
    print(f"Train Shape: {train_df.shape}")

    # 4. 标准化
    all_cols = train_df.columns
    exclude_final = [CONFIG['label_col'], 'timestamp', 'trade_day','LABEL_CAL_DQ_inst1_60','is_pm','is_am','is_night','gate_session_decay','gate_open_impulse']
    final_feature_cols = [c for c in all_cols if c[:2]=='f_']+['OBI_cum120','rolling_amp','gate_rv','gate_downside_ratio','gate_rv_log','gate_range_log','delta_vol_1','delta_vol_2','delta_vol_4','delta_vol_8','delta_vol_16']
    print(len(final_feature_cols))
    print(f"最终特征数量: {len(final_feature_cols)}")
    joblib.dump(final_feature_cols, os.path.join(CONFIG['save_dir'], 'feature_cols.pkl'))

    print("正在标准化特征...")
    scaler = StandardScaler()
    scaler.fit(train_df[final_feature_cols].values)
    
    joblib.dump(scaler, os.path.join(CONFIG['save_dir'], 'scaler.pkl'))
    
    train_df[final_feature_cols] = scaler.transform(train_df[final_feature_cols].values).astype(np.float32)
    if not valid_df.empty:
        valid_df[final_feature_cols] = scaler.transform(valid_df[final_feature_cols].values).astype(np.float32)
    
    context_features = ['OBI_cum120','rolling_amp','gate_rv','gate_downside_ratio','gate_rv_log','gate_range_log','delta_vol_1','delta_vol_2','delta_vol_4','delta_vol_8','delta_vol_16']
    train_df[context_features] = train_df[context_features].fillna(0.0).clip(-3,3)
    valid_df[context_features] = valid_df[context_features].fillna(0.0).clip(-3,3)
    def map_to_range(z_score, scale=1.0):
        sigmoid = 1 / (1 + np.exp(-z_score))
        return sigmoid * scale
    for feature in context_features:
        train_df[feature+'_map'] = map_to_range(train_df[feature])
        valid_df[feature+'_map'] = map_to_range(valid_df[feature])
    print("处理完成！")
    return train_df, valid_df

# ================= 运行 =================
if __name__ == "__main__":
    # 示例：如果在 CONFIG 里没填，可以在这里临时覆盖
    # CONFIG['enhanced_features'] = ['ask_price_1', 'bid_price_1'] 
    
    train_df, valid_df = process_data_to_dataframe()
    if train_df is not None:
        train_df.to_pickle(f"traindata_augmented_{CONFIG['window_size']}_v4.pkl")
        valid_df.to_pickle(f"validdata_augmented_{CONFIG['window_size']}_v4.pkl")