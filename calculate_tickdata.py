## 这个文件用于处理tick级别的数据和信号计算
import pandas as pd
import numpy as np
calendar = pd.read_csv("au_calendar.csv")
calendar['date'] = pd.to_datetime(calendar['date'])
calendar = calendar[(calendar['date']>=pd.to_datetime('20200101'))&(calendar['date']<=pd.to_datetime('20241120'))]
calendar['contract'] = calendar['contract'].str[:-4]
dates = calendar['date'].unique()
calendar.set_index(['date'],inplace = True)
import numpy as np
from numba import njit
from numba import jit, float64,int64
def group_stats_matrix(df_date, group_ids, min_samples=5):
    """
    基于矩阵运算的分组统计计算
    :param df_date: (n_samples,) 数值数组
    :param group_ids: (n_samples,) 分组标识数组
    :return: (n_groups, 7) 矩阵 [mean, std, range, skew, kurtosis, delta, count]
    """
    # 预排序加速分组定位
    sort_idx = np.argsort(group_ids)
    sorted_df_date = df_date[sort_idx]
    sorted_groups = group_ids[sort_idx]
    
    # 获取分组边界
    group_boundaries = np.where(np.diff(sorted_groups))[0] + 1
    group_starts = np.concatenate([[0], group_boundaries])
    group_ends = np.concatenate([group_boundaries, [len(df_date)]])
    n_groups = len(group_starts)
    
    # 预分配结果矩阵
    stats_matrix = np.empty((n_groups, 7), dtype=np.float64)
    
    # 核心计算
    for i in range(n_groups):
        start, end = group_starts[i], group_ends[i]
        group_df_date = sorted_df_date[start:end]
        stats_matrix[i] = _calc_stats(group_df_date, min_samples)
    
    return stats_matrix

@njit(fastmath=True)
def _calc_stats(arr, min_samples):
    """JIT加速的单组统计计算"""
    n = len(arr)
    if n < min_samples:
        return np.array([np.nan]*6)
    
    # 基础统计量
    mean = np.mean(arr)
    std = np.std(arr)
    df_date_range = np.max(arr) - np.min(arr)
    delta = arr[-1] - arr[0] if n > 0 else 0
    
    # 高阶矩优化计算
    if n >= 10:  # 小样本跳过复杂计算
        centered = arr - mean
        m2 = np.sum(centered**2)
        m3 = np.sum(centered**3)
        m4 = np.sum(centered**4)
        
        skewness = (m3/n) / (m2/n)**1.5 if m2 > 0 else 0
        kurt = (m4/n) / (m2/n)**2 - 3 if m2 > 0 else 0
    else:
        skewness = kurt = np.nan
    
    return np.array([mean, std, df_date_range, skewness, kurt, delta])

@njit(fastmath=True)
def numba_ema(series, window):
    """Numba加速的EMA计算"""
    result = np.empty_like(series)
    
    alpha = 2.0 / (window + 1.0)
    result[0] = series[0]
    
    for i in range(1, len(series)):
        if np.isnan(result[i-1]):
            result[i] = series[i]
        else:
            result[i] = series[i] * alpha + result[i-1] * (1 - alpha)
    
    return result

@jit(float64[:](float64[:], int64), nopython=True)
def numba_rma(series, window):
    """Numba加速的RMA计算(用于RSI)"""
    result = np.empty_like(series)
    result[:] = np.nan
    
    if len(series) < window:
        return result
    
    alpha = 1.0 / window
    result[window-1] = series[:window].mean()
    
    for i in range(window, len(series)):
        result[i] = series[i] * alpha + result[i-1] * (1 - alpha)
    
    return result

@jit(nopython=True)
def numba_macd(close, fast=12, slow=26, signal=9):
    """Numba加速的MACD计算"""
    ema_fast = numba_ema(close, fast)
    ema_slow = numba_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = numba_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line
import numpy as np
from numba import jit, float64, int64

@jit(nopython=True)
def numba_rsi(close, window=14):
    """
    使用Numba加速的RSI计算
    参数:
        close: 收盘价数组
        window: 计算窗口(默认14)
    返回:
        RSI值数组
    """
    n = len(close)
    rsi = np.empty(n)
    # 计算价格变化
    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    
    # 分离上涨和下跌
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    
    # 计算初始平均值
    avg_gain = np.mean(up[:window])
    avg_loss = np.mean(down[:window])
    
    if avg_loss == 0:
        rsi[window-1] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[window-1] = 100.0 - (100.0 / (1.0 + rs))
    
    # 迭代计算剩余RSI值
    for i in range(window, n):
        avg_gain = (avg_gain * (window-1) + up[i]) / window
        avg_loss = (avg_loss * (window-1) + down[i]) / window
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi
test_cols = ['spread', 'bid_depth', 'ask_depth',
       'total_depth', 'order_imbalance', 'OI_MA_10', 'OI_MA_60', 'OI_MA_120',
       'OI_MA_300', 'OI_MA_600', 'OI_momentum_10', 'OI_momentum_120',
       'OI_momentum_60', 'OI_momentum_300', 'OI_momentum_600',
       'OI_cumulative_10', 'OI_cumulative_60', 'OI_cumulative_120',
       'OI_cumulative_300', 'OI_cumulative_600', 'OI_std_120', 'OI_std_300',
       'OI_std_600', 'OI_skew_120', 'OI_skew_600', 'D_k', 'ES',
       'effective_spread', 'market_pressure', 'SOIR1', 'SOIR2', 'SOIR3',
       'SOIR4', 'SOIR5', 'bid_ask_volume_ratio', 'relative_spread', 'SOIR',
       'SOIR_weighted', 'OFI1', 'OFI2', 'OFI3', 'OFI4', 'OFI5', 'MOFI',
       'vol_std_120', 'vol_std_600', 'vol_skew_120', 'vol_skew_600']
@njit(fastmath=True)
def fast_linregress(x, y):
        n = len(x)
        x_mean, y_mean = 0.0, 0.0
        cov, var_x = 0.0, 0.0
        
        # 单次遍历计算统计量
        for i in range(n):
            x_mean += x[i]
            y_mean += y[i]
        x_mean /= n
        y_mean /= n
        
        for i in range(n):
            x_diff = x[i] - x_mean
            cov += x_diff * (y[i] - y_mean)
            var_x += x_diff**2
        
        slope = cov / (var_x + 1e-10)
        intercept = y_mean - slope * x_mean
        return slope, intercept
def hurst_matrix_optimized(series, n_values=[10, 20, 50, 100]):
        series = np.asarray(series, dtype=np.float64)
        T = len(series)
        log_n = np.log(n_values)
        log_F_n = np.zeros_like(log_n)
        
        # 预计算全局均值
        global_mean = np.mean(series)
        
        for i, n in enumerate(n_values):
            segments = T // n
            if segments == 0:
                continue
                
            # 矩阵化计算
            seg_matrix = series[:segments*n].reshape(-1, n)
            seg_means = seg_matrix.mean(axis=1, keepdims=True)
            
            # 向量化R/S计算
            cum_dev = np.cumsum(seg_matrix - seg_means, axis=1)
            ranges = np.ptp(cum_dev, axis=1)
            stds = np.std(seg_matrix, axis=1, ddof=0)
            
            valid_mask = stds > 1e-10
            RS = np.where(valid_mask, ranges / stds, 0)
            log_F_n[i] = np.log(np.mean(RS[valid_mask])) if np.any(valid_mask) else 0
        
        # 使用手动回归替代scipy
        H, _ = fast_linregress(log_n, log_F_n)
        return H
    # 滚动窗口的并行计算
PRICE_COLS = ['preclose', 'open', 'high', 'low', 'last','presettle', 'highlimit', 'lowlimit']+['askp'+str(i) for i in range(1,6)]+['bidp'+str(i) for i in range(1,6)]
date_ic = pd.DataFrame(columns = test_cols)
all_30s = []
last_contract = None
from scipy.stats import entropy
cumulative_factor = 1
# for i in range(len(dates)-1,-1,-1):
#     date = dates[i]
#     print(date)
#     df_read = pd.read_feather('D:\JT_Summer\\au\\'+date.strftime("%Y%m%d")+'\\'+calendar.loc[date]['contract'].lower()+'.feather')
    
#     ## 前复权操作
#     if cumulative_factor != 1.0:
#             for col in PRICE_COLS:
#                 if col in df_read.columns:
#                     df_read[col] *= cumulative_factor
#     if i>0 and calendar.loc[date]['contract']!= calendar.loc[dates[i-1]]['contract']:
#         ref_price_current = df_read['presettle'].iloc[0] / cumulative_factor # 要用原始價格計算
#         df_prev = pd.read_feather('D:\JT_Summer\\au\\'+dates[i-1].strftime("%Y%m%d")+'\\'+calendar.loc[dates[i-1]]['contract'].lower()+'.feather')        
#                 # 舊合約基準價：前一天(較舊)的最後一筆成交價
#         ref_price_prev_close = df_prev['last'].iloc[-1]
#         if ref_price_prev_close > 0 and ref_price_current > 0:
#                     rollover_factor = ref_price_current / ref_price_prev_close
#                     cumulative_factor *= rollover_factor
#                     print(f"{date.strftime("%Y%m%d")},新因子: {rollover_factor:.6f} | 累計因子: {cumulative_factor:.6f}")
import os
def get_immediate_subfile_paths(folder_path):
    """
    获取指定文件夹下直接的子文件路径。
    
    参数:
        folder_path (str): 目标文件夹的路径。
        
    返回:
        list: 包含所有直接子文件完整路径的列表。
    """
    subfile_paths = []
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个有效的文件夹路径。")
        return subfile_paths
        
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)
        if os.path.isfile(item_path):  # 检查是否为文件
            subfile_paths.append(item_path)
    return subfile_paths
calendar_list = [x.lower() for x in calendar['contract'].unique().tolist()]
print(calendar_list)
for i in range(len(dates)):
    date = dates[i]
    print(date)
    date_datas = get_immediate_subfile_paths('D:\JT_Summer\\au\\'+date.strftime("%Y%m%d")+'\\')
    
    for path in date_datas:
        contract = os.path.basename(path)[:-8]
        if contract not in calendar_list:
            continue
        print(contract)
        # break
        df_read = pd.read_feather(path)
        # df_read = pd.read_feather('D:\JT_Summer\\au\\'+date.strftime("%Y%m%d")+'\\'+calendar.loc[date]['contract'].lower()+'.feather')

        if df_read.columns[0] != 'time':       
            df_read = df_read.iloc[:,1:]
        # df_read.set_index(['time'],inplace = True)
        # df_date = calculate_tick_vwap_and_return(df_read)
        df_date = df_read
        df_date['delta_volume'] = df_date['totalvolume'].diff()
        df_date['bigorder_volume'] = 0
        df_date.loc[df_date['delta_volume']>=10,'bigorder_volume'] = df_date['delta_volume']
        df_date.iloc[0, df_date.columns.get_loc('delta_volume')] = df_date.iloc[0]['totalvolume']
        df_date['delta_turnover'] = df_date['totalturnover'].diff()
        df_date.iloc[0, df_date.columns.get_loc('delta_turnover')] = df_date.iloc[0]['totalturnover']
        df_date['mid_price'] = (df_date['bidp1'] + df_date['askp1'])/2
        df_date.loc[df_date['bidp1']==0,'mid_price'] = df_date.loc[df_date['bidp1']==0,'askp1']
        df_date.loc[df_date['askp1']==0,'mid_price'] = df_date.loc[df_date['askp1']==0,'bidp1']
        df_date[['askp1','bidp1','askp2','bidp2','askp3','bidp3','askp4','bidp4','askp5','bidp5']] = df_date[['askp1','bidp1','askp2','bidp2','askp3','bidp3','askp4','bidp4','askp5','bidp5']].replace(0,np.nan)
        price_diff = df_date['last'].diff()
        tick_direction = price_diff.ffill().apply(np.sign)

        # 應用混合規則推斷交易方向
        # 1: 主動買入, -1: 主動賣出
        conditions = [
            df_date['last'] > df_date['mid_price'],  # Quote Rule: 買家擊穿賣一價
            df_date['last'] < df_date['mid_price'],  # Quote Rule: 賣家擊穿買一價
            tick_direction == 1,                               # Tick Rule: 上升 Tick
            tick_direction == -1                               # Tick Rule: 下降 Tick
        ]
        choices = [1, -1, 1, -1]
        df_date['trade_direction'] = np.select(conditions, choices, default=0)
        df_date['buy_volume'] = np.where(df_date['trade_direction'] == 1, df_date['delta_volume'], 0)
        df_date['sell_volume'] = np.where(df_date['trade_direction'] == -1, df_date['delta_volume'], 0)
        df_date['net_inflow'] = df_date['buy_volume'] - df_date['sell_volume']

        # 根據方向預估實際成交價
        exec_price_conditions = [
            df_date['trade_direction'] == 1,
            df_date['trade_direction'] == -1
        ]
        exec_price_choices = [
            df_date['askp1'], # 主動買入，成交在賣一價
            df_date['bidp1']  # 主動賣出，成交在買一價
        ]
        df_date['estimated_exec_price'] = np.select(exec_price_conditions, exec_price_choices, default=df_date['last'])
        df_date['spread'] = df_date['askp1'] - df_date['bidp1']
        df_date['bid_depth'] = df_date['bidv1']+df_date['bidv2']+df_date['bidv3']+df_date['bidv4']+df_date['bidv5']
        df_date['ask_depth'] = df_date['askv1']+df_date['askv2']+df_date['askv3']+df_date['askv4']+df_date['askv5']
        df_date['total_depth'] = df_date['bid_depth'] + df_date['ask_depth'] 
        df_date['order_imbalance'] = (df_date['bid_depth']-df_date['ask_depth'])/(df_date['bid_depth']+df_date['ask_depth'])
        df_date['OI_MA_10'] = df_date['order_imbalance'].rolling(window=10).mean()
        df_date['OI_MA_60'] = df_date['order_imbalance'].rolling(window=60).mean()
        df_date['OI_MA_120'] = df_date['order_imbalance'].rolling(window=120).mean()
        df_date['OI_MA_300'] = df_date['order_imbalance'].rolling(window=300).mean()
        df_date['OI_MA_600'] = df_date['order_imbalance'].rolling(window=600,min_periods=120).mean()
        df_date['OI_momentum_10'] = df_date['order_imbalance'].diff(10)
        df_date['OI_momentum_120'] = df_date['order_imbalance'].diff(120)
        df_date['OI_momentum_60'] = df_date['order_imbalance'].diff(60)
        df_date['OI_momentum_300'] = df_date['order_imbalance'].diff(300)
        df_date['OI_momentum_600'] = df_date['order_imbalance'].diff(600)
        df_date['OI_std_120'] = df_date['order_imbalance'].rolling(window=120).std()
        df_date['OI_std_300'] = df_date['order_imbalance'].rolling(window=300,min_periods=120).std()
        df_date['OI_std_600'] = df_date['order_imbalance'].rolling(window=600,min_periods=120).std()
        df_date['OI_skew_120'] = df_date['order_imbalance'].rolling(window=120).skew()
        df_date['OI_skew_600'] = df_date['order_imbalance'].rolling(window=600,min_periods=120).skew()
        def calculate_direction(P_t, mid_price):
            """
            根据成交价与最优报价中间价确定买卖方向.
            """
            if P_t > mid_price:
                return 1  # 买单
            elif P_t < mid_price:
                return -1  # 卖单
            else:
                return 0  # 无法确定
        def calculate_effective_spread(df):
            """
            计算有效价差因子 ES.
            """
            df['D_k'] = df.apply(lambda row: calculate_direction(row['last'], row['mid_price']), axis=1)
            df['ES'] = 2 * df['D_k'] * (df['last'] - df['mid_price']) / df['mid_price']
            return df['ES']
        df_date['effective_spread'] = calculate_effective_spread(df_date)
        def calculate_ofi(df, bid_price, ask_price, bid_volume, ask_volume):
            # 计算买卖价和买卖量的变化
            delta_bid_vol = np.where(df[bid_price] < df[bid_price].shift(1), -df[bid_volume].shift(1),
                                    np.where(df[bid_price] > df[bid_price].shift(1), df[bid_volume],
                                            df[bid_volume] - df[bid_volume].shift(1)))

            delta_ask_vol = np.where(df[ask_price] < df[ask_price].shift(1), df[ask_volume],
                                    np.where(df[ask_price] > df[ask_price].shift(1), -df[ask_volume].shift(1),
                                            df[ask_volume] - df[ask_volume].shift(1)))
            # 计算 OFI
            ofi = delta_bid_vol - delta_ask_vol
            return ofi
        df_date['market_pressure'] = (df_date['bidp1']*df_date['bidv1']-df_date['askp1']*df_date['askv1'])/(df_date['bidp1']*df_date['bidv1']+df_date['askp1']*df_date['askv1'])
        df_date['SOIR1'] = (df_date['bidv1']-df_date['askv1'])/(df_date['bidv1']+df_date['askv1'])
        df_date['SOIR2'] = (df_date['bidv2']-df_date['askv2'])/(df_date['bidv2']+df_date['askv2'])
        df_date['SOIR3'] = (df_date['bidv3']-df_date['askv3'])/(df_date['bidv3']+df_date['askv3'])
        df_date['SOIR4'] = (df_date['bidv4']-df_date['askv4'])/(df_date['bidv4']+df_date['askv4'])
        df_date['SOIR5'] = (df_date['bidv5']-df_date['askv5'])/(df_date['bidv5']+df_date['askv5'])
        df_date['bid_ask_volume_ratio'] = (df_date['bidv1']+df_date['bidv2']+df_date['bidv3']+df_date['bidv4']+df_date['bidv5'])/(df_date['askv1']+df_date['askv2']+df_date['askv3']+df_date['askv4']+df_date['askv5'])
        df_date['relative_spread'] = (df_date['askp1'] - df_date['bidp1'])/((df_date['askp1'] + df_date['bidp1'])/2)
        df_date['SOIR'] = df_date['SOIR1']+df_date['SOIR2']+df_date['SOIR3']+df_date['SOIR4']+df_date['SOIR5']
        df_date['SOIR_weighted'] = 5*df_date['SOIR1']+4*df_date['SOIR2']+3*df_date['SOIR3']+2*df_date['SOIR4']+df_date['SOIR5']

        df_date['OFI1'] = calculate_ofi(df_date, 'bidp1', 'askp1', 'bidv1', 'askv1')
        df_date['OFI2'] = calculate_ofi(df_date, 'bidp2', 'askp2', 'bidv2', 'askv2')
        df_date['OFI3'] = calculate_ofi(df_date, 'bidp3', 'askp3', 'bidv3', 'askv3')
        df_date['OFI4'] = calculate_ofi(df_date, 'bidp4', 'askp4', 'bidv4', 'askv4')
        df_date['OFI5'] = calculate_ofi(df_date, 'bidp5', 'askp5', 'bidv5', 'askv5')
        df_date['MOFI'] = df_date['OFI1']+df_date['OFI2']+df_date['OFI3']+df_date['OFI4']+df_date['OFI5']
        df_date['vol_std_120'] = df_date['delta_volume'].rolling(window=120).std()
        df_date['vol_std_600'] = df_date['delta_volume'].rolling(window=600).std()
        df_date['vol_skew_120'] = df_date['delta_volume'].rolling(window=120).skew()
        df_date['vol_skew_600'] = df_date['delta_volume'].rolling(window=600).skew()
        df_date['ret_600'] = df_date['mid_price'].shift(-601)/df_date['mid_price'].shift(-1) - 1
        df_date['tick_ret'] = df_date['mid_price']/df_date['mid_price'].shift(1) - 1
        df_date['vol_mid_corr_120'] = df_date['delta_volume'].rolling(window=120).corr(df_date['tick_ret'])
        df_date['vol_mid_corr_600'] = df_date['delta_volume'].rolling(window=600,min_periods=120).corr(df_date['tick_ret'])
        df_date['rp_momentum_120'] = (df_date['tick_ret']*df_date['delta_turnover']).rolling(120).sum()
        df_date['rp_momentum_600'] = (df_date['tick_ret']*df_date['delta_turnover']).rolling(600,min_periods=120).sum()
        df_date['rp_momentum_20'] = (df_date['tick_ret']*df_date['delta_turnover']).rolling(20).sum()
        df_date['delta2_tick'] = ((df_date['mid_price'] +df_date['mid_price'].shift(2) - 2*df_date['mid_price'].shift(1))/(0.25))
        df_date['delta2_tick_rolling_1min'] = df_date['delta2_tick'].rolling(120).mean()
        df_date['delta2_tick_rolling_5min'] = df_date['delta2_tick'].rolling(600,min_periods=120).mean()
        df_date['delta2_10s'] = (df_date['mid_price'] +df_date['mid_price'].shift(40) - 2*df_date['mid_price'].shift(20))/(100)
        df_date['delta2_10s_rolling_1min'] = df_date['delta2_10s'].rolling(120).mean()
        df_date['delta2_10s_rolling_5min'] = df_date['delta2_10s'].rolling(600,min_periods=120).mean()
        df_date['delta2_60s'] = (df_date['mid_price'] +df_date['mid_price'].shift(240) - 2*df_date['mid_price'].shift(120))/(3600)
        df_date['tick_smooth_price_change'] = (df_date['mid_price'] - df_date['mid_price'].shift(1)).ewm(alpha=0.2).mean()
        df_date['elasticity'] = (df_date['total_depth'] - df_date['total_depth'].shift(1))/df_date['tick_smooth_price_change'].abs()
        df_date['elasticity_rolling_60s'] = df_date['elasticity'].rolling(120).mean()
        calendar_path = "D:\JT_Summer\\au_tickfactor\\"+contract+"\\"
        if not os.path.exists(calendar_path):
            os.makedirs(calendar_path, exist_ok=True)
        df_date.to_pickle(calendar_path+date.strftime("%Y%m%d")+".pkl")