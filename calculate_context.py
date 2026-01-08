import pandas as pd
import numpy as np
import os
import glob
import warnings
from tqdm import tqdm
import numba
warnings.filterwarnings('ignore')
      
CONFIG = {
    'input_dir': '/home/zyyuan/project1/try/market_data',
    'output_dir': '/home/zyyuan/project2/processed_marketdata_parquet',
    'file_pattern': '*.csv' 
}

# ================= 1. 基础数据预处理 =================
def preprocess_raw_data(df):
    """
    清洗原始数据，生成标准时间戳
    """
    # 1. 过滤掉非交易时间或异常数据 (可选)
    # df = df[df['volume'] > 0] 
    
    # 2. 生成 Timestamp (假设列名为 ExchActionDay, ExchUpdateTime, ExchUpdateMillisec)
    # 向量化处理时间，速度最快
    if 'timestamp' not in df.columns:
        try:
            # 日期部分
            date_str = df['ExchActionDay'].astype(str)
            
            # 时间部分 HH:MM:SS -> Seconds
            # 假设 ExchUpdateTime 格式为 '14:00:01' 或类似
            time_str = df['ExchUpdateTime'].astype(str)
            
            # 拼接字符串转 datetime (比 apply 快)
            # 格式: YYYYMMDD HH:MM:SS.mmm
            full_time_str = date_str + ' ' + time_str + '.' + df['ExchUpdateMillisec'].astype(str).str.zfill(3)
            df['timestamp'] = pd.to_datetime(full_time_str, format='%Y%m%d %H:%M:%S.%f')
            
        except Exception as e:
            # Fallback: 如果格式不标准，尝试通用解析
            print(f"Time parsing warning: {e}")
            pass
            
    # 排序
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
        
    return df
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
# ================= 2. 特征计算逻辑 =============
import pandas as pd
import numpy as np
import warnings

# 忽略除以0等警告
warnings.filterwarnings('ignore')

def calculate_selected_features(df_input):
    df = df_input.copy()
    df = preprocess_for_calc(df)
    df['ret_min'] = df['mid_price'] / df['mid_price'].shift(1) - 1
    df['is_up'] = np.where(df['ret_min'] > 0, 1, 0)
    dollar_volume = df['mid_price'] * df['vol']
    dollar_volume = dollar_volume.replace(0, np.nan)
    illiq_ratio = df['ret_min'].abs() / dollar_volume * 1e6
    df['PSY_60'] = df['is_up'].rolling(window=60).mean()
    df['ret_kurt_120'] = df['ret_min'].rolling(window=120, min_periods=10).kurt()
    df['amihud_ratio_60'] = illiq_ratio.rolling(window=60, min_periods=10).mean()
    df['ret_5min'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['ar_10'] = (df['high'] - df['open']).rolling(10,min_periods=5).sum()/(df['open'] - df['low']).rolling(10,min_periods=5).sum()
    ma20 = df['mid_price'].rolling(window=20).mean()
    std20 = df['mid_price'].rolling(window=20).std()
    df['lower_bb'] = (ma20 - 2 * std20 - df['mid_price']) / df['mid_price']
    df['CV_30'] = df['ret_min'].rolling(30,min_periods=5).var() / np.abs(df['ret_min'].rolling(30,min_periods=5).mean())
    df['cvar_10'] = df['ret_min'].rolling(window=10).apply(_calculate_cvar_window, raw=True, args=(0.05,))
    target_cols = [
        'timestamp',
        'PSY_60', 
        'ret_kurt_120', 
        'amihud_ratio_60', 
        'ret_5min', 
        'ar_10', 
        'lower_bb', 
        'CV_30', 
        'rp_momentum_120',
        'bigorder_pct',
        'cvar_10'
    ]
    
    return df[target_cols]

def preprocess_for_calc(df_raw):
    df = df_raw.copy()
    df['mid_price'] = (df['bp1'] + df['ap1']) / 2
    df.loc[df['bp1'] == 0, 'mid_price'] = df.loc[df['bp1'] == 0, 'ap1']
    df.loc[df['ap1'] == 0, 'mid_price'] = df.loc[df['ap1'] == 0, 'bp1']
    df['vol'] = df['volume'] - df['volume'].shift(120)
    df['high'] = df['mid_price'].rolling(120).max()
    df['low'] = df['mid_price'].rolling(120).min()
    df['open'] = df['mid_price'].shift(119)
    df['delta_turnover'] = df['turnover'].diff()
    df['tick_ret'] = df['mid_price']/df['mid_price'].shift(1) - 1
    df['rp_momentum_120'] = (df['tick_ret']*df['delta_turnover']).rolling(120).sum()
    df['delta_volume'] = df['volume'].diff()
    df['bigorder_volume'] = 0
    df.loc[df['delta_volume']>=10,'bigorder_volume'] = df['delta_volume']
    df['is_bigorder'] = 0
    df.loc[df['bigorder_volume']>0,'is_bigorder'] = 1
    df['bigorder_pct'] = df['is_bigorder'].rolling(1200).mean()
    df['obi'] = (df['bv1'] - df['av1']) / (df['bv1'] + df['av1']+1e-6)
    b_change = np.where(df['bp1'] > df['bp1'].shift(1), df['bv1'], 
                        np.where(df['bp1'] < df['bp1'].shift(1), -df['bv1'].shift(1), 
                                 df['bv1'].diff()))
    a_change = np.where(df['ap1'] < df['ap1'].shift(1), df['av1'], 
                        np.where(df['ap1'] > df['ap1'].shift(1), -df['av1'].shift(1), 
                                 df['av1'].diff()))
    df['ofi'] = np.nan_to_num(b_change - a_change)
    for col in ['open', 'high', 'low', 'mid_price']:
        df[col] = df[col].clip(lower=1e-9)
    df = df[59::60]
    return df
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
import re
def process_aligned_features():
    # ================= 配置区域 =================
    CONFIG = {
        'input_dir': '/home/zyyuan/project1/try/market_data',       # 原始行情文件夹
        'factor_dir': '/home/zyyuan/project2/feature_generate_all_np_ag_v5_intern_parquet',  # 因子文件夹 (提供时间戳基准)
        'output_dir': '/home/zyyuan/project2/processed_marketdata_parquet',  # 输出文件夹
        'file_pattern': '*.csv'                    # 原始行情文件后缀
    }
    # ===========================================
    
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])

    # 1. 获取所有行情文件
    market_files = sorted(glob.glob(os.path.join(CONFIG['input_dir'], CONFIG['file_pattern'])))
    print(f"Found {len(market_files)} market files.")
    factor_all = []
    for m_path in tqdm(market_files, desc="Processing"):
        try:
            # --- A. 从文件名提取日期 (用于匹配因子文件) ---
            base_name = os.path.basename(m_path)
            # 正则匹配 8位数字 (如 20250425)
            date_match = re.search(r'(202[0-9]{5})', base_name)
            
            if not date_match:
                print(f"Skipping {base_name}: No date found in filename.")
                continue
                
            date_str = date_match.group(1)
            factor_search_pattern = os.path.join(CONFIG['factor_dir'], f"*{date_str}*.parquet")
            found_factors = glob.glob(factor_search_pattern)
            
            if not found_factors:
                print(f"Skipping {base_name}: No corresponding factor file found for date {date_str}.")
                continue
            
            factor_path = found_factors[0] # 取第一个匹配到的

            # --- C. 读取数据 ---
            
            # 1. 读取因子时间戳 (作为对齐基准)
            # 仅读取 timestamp 列，速度极快
            df_factor_ref = pd.read_parquet(factor_path, columns=['ExchActionDay', 'ExchUpdateTime', 'ExchUpdateMillisec','prj2_1_label'])
            df_factor_ref = create_timestamp_fast(df_factor_ref)
            # 2. 读取原始行情数据
            df_market = pd.read_csv(m_path)
            # --- D. 构造行情时间戳 ---
            df_market['hms'] = pd.to_datetime(df_market['hms'])
            ms_delta = pd.to_timedelta(df_market['ms'], unit='ms')
            df_market['timestamp'] = df_market['hms'] + ms_delta
            if df_factor_ref['timestamp'].dtype != df_market['timestamp'].dtype:
                 df_factor_ref['timestamp'] = pd.to_datetime(df_factor_ref['timestamp'])
                 df_market['timestamp'] = pd.to_datetime(df_market['timestamp'])
            
            df_aligned = pd.merge(
                df_factor_ref,    # 左表：基准时间轴
                df_market,        # 右表：原始数据
                on='timestamp',   # 对齐键
                how='left'        # 保证结果长度和因子文件完全一致
            )
            
            # --- F. 计算特征 ---
            # 此时 df_aligned 已经是经过筛选、行数与因子完全一致的数据了
            if df_aligned.empty:
                print(f"Warning: {base_name} aligned data is empty!")
                continue

            df_features = calculate_selected_features(df_aligned)
            float_cols = df_features.select_dtypes(include=['float64']).columns
            df_features[float_cols] = df_features[float_cols].astype(np.float32)
            
            # save_name = base_name.replace('.csv', '_aligned_features.parquet')
            # save_path = os.path.join(CONFIG['output_dir'], save_name)
            factor_all.append(df_features)
            # df_features.to_parquet(save_path, index=False, compression='snappy')
            
        except Exception as e:
            print(f"Error processing {m_path}: {e}")
            import traceback
            traceback.print_exc()
    factor_all = pd.concat(factor_all)
    factor_all.to_pickle("factor_all_valid.pkl")

if __name__ == "__main__":
    process_aligned_features()