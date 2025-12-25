## 这个文件用于拼接形成主力合约，并且计算分钟级别的特征


import pandas as pd
import numpy as np
import numba
from numba import njit
from numba import jit, float64,int64
import warnings
warnings.filterwarnings('ignore')
from PyEMD import EMD
def calculate_snr_for_window(window_prices: np.ndarray) -> float:
    """
    為單個窗口的價格數據計算 EMD 信噪比。
    
    Args:
        window_prices (np.ndarray): 一個窗口的價格數組。
        
    Returns:
        float: 該窗口的 SNR 因子值。
    """
    if len(window_prices) < 10: # EMD 需要足夠的數據點
        return np.nan
    try:
        emd = EMD()
        imfs = emd.emd(window_prices)
        
        if imfs.shape[0] < 1: # 如果無法分解
            return np.nan

        # 殘差(趨勢)是最後一個分量
        signal = imfs[-1]
        # 噪音是原始信號減去趨勢
        noise = window_prices - signal
        
        std_signal = np.std(signal)
        std_noise = np.std(noise)
        
        if std_noise < 1e-9: # 避免除以零
            return np.nan
            
        snr = np.log(std_signal / std_noise)
        return snr
        
    except Exception:
        return np.nan
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
    delta= arr[-1] - arr[0] if n > 0 else 0
    
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

import pandas as pd
import numpy as np
# calendar = pd.read_csv("au_calendar.csv")
# calendar['date'] = pd.to_datetime(calendar['date'])
# calendar = calendar[(calendar['date']>=pd.to_datetime('20200101'))&(calendar['date']<=pd.to_datetime('20241120'))]
# calendar['contract'] = calendar['contract'].str[:-4]
# dates = calendar['date'].unique()
# calendar.set_index(['date'],inplace = True)
import numpy as np
from numba import njit
from numba import jit, float64,int64
@jit(nopython=True)
def numba_kdj(high, low, mid_price, n=9, m1=3, m2=3):
    """
    使用Numba加速的KDJ计算
    参数:
        high: 最高价数组
        low: 最低价数组
        mid_price: 收盘价数组
        n: RSV计算窗口(默认9)
        m1: K线平滑周期(默认3)
        m2: D线平滑周期(默认3)
    返回:
        K值, D值, J值
    """
    size = len(mid_price)
    K = np.empty(size)
    D = np.empty(size)
    J = np.empty(size)
    K[:] = np.nan
    D[:] = np.nan
    J[:] = np.nan
    
    # 计算RSV
    for i in range(n-1, size):
        window_high = high[i-n+1:i+1].max()
        window_low = low[i-n+1:i+1].min()
        
        if window_high == window_low:
            rsv = 50.0  # 避免除以0
        else:
            rsv = (mid_price[i] - window_low) / (window_high - window_low) * 100.0
        
        # 计算K值
        if np.isnan(K[i-1]):
            K[i] = rsv
        else:
            K[i] = (K[i-1] * (m1-1) + rsv) / m1
        
        # 计算D值
        if np.isnan(D[i-1]):
            D[i] = K[i]
        else:
            D[i] = (D[i-1] * (m2-1) + K[i]) / m2
        
        # 计算J值
        J[i] = 3 * K[i] - 2 * D[i]
    
    return J
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
@njit
def linear_fit(x, y):
    """Numba 兼容的线性回归 (y = a*x + b)"""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    
    denominator = n * sum_x2 - sum_x ** 2
    a = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y * sum_x2 - sum_x * sum_xy) / denominator
    return a, b
@jit(forceobj=True)
def _calculate_regression_factor_window(window_prices: np.ndarray) -> float:
    """
    為單個窗口計算時序回歸因子。此函數將被 Numba 優化。

    Args:
        window_prices (np.ndarray): 一個窗口的收盤價數組。

    Returns:
        float: 該窗口的因子值 (b * R^2)。
    """
    n = len(window_prices)
    if n < 3: # 至少需要3個點才能進行二次擬合
        return np.nan
    t = np.arange(n)
    
    # 進行二次多項式擬合 (y = ct^2 + bt + a)
    # np.polyfit 返回係數 [c, b, a]
    try:
        coeffs = np.polyfit(t, window_prices, 2)
        c, b, a = coeffs[0], coeffs[1], coeffs[2]
    except np.linalg.LinAlgError:
        return np.nan

    # 計算擬合優度 R^2
    y_hat = c * t**2 + b * t + a
    ss_res = np.sum((window_prices - y_hat)**2) # 殘差平方和
    ss_tot = np.sum((window_prices - np.mean(window_prices))**2) # 總平方和

    if ss_tot < 1e-9: # 避免除以零
        r_squared = 1.0 if ss_res < 1e-9 else 0.0
    else:
        r_squared = 1 - ss_res / ss_tot
        
    # 返回最終因子值: b * R^2
    return b * r_squared
def calculate_volume_entropy(tick_df_window: pd.DataFrame) -> float:
    """
    為一個時間窗口內的 Tick 數據計算成交量分桶熵。

    Args:
        tick_df_window (pd.DataFrame): 包含一個時間窗口（例如30分鐘）的 Tick 數據。
                                      需要有 'price', 'volume', 'bid_price', 'ask_price' 列。

    Returns:
        float: 該窗口的成交量熵值。
    """
    if tick_df_window.empty or tick_df_window['delta_volume'].sum() <= 0:
        return np.nan
    
    # 步驟二：計算分桶熵
    # 1. 計算總成交量
    total_volume = tick_df_window['delta_volume'].sum()

    # 2. 按推斷的成交價分桶，並計算每個桶的成交量
    volume_profile = tick_df_window.groupby('estimated_exec_price')['delta_volume'].sum()

    # 3. 計算每個桶的成交量佔比 (p_k)
    probabilities = volume_profile / total_volume
    
    # 過濾掉概率為0的情況
    probabilities = probabilities[probabilities > 0]
    
    if probabilities.empty:
        return 0.0 # 如果只有一個價格點，熵為0

    # 4. 根據公式計算香農熵
    # vol_entropy = - Σ(p_k * ln(p_k))
    entropy = -np.sum(probabilities * np.log(probabilities))
    
    return entropy
@njit
def safe_lstsq(x, y):
    """用 lstsq 进行线性拟合，避免异常捕获"""
    A = np.vstack((x, np.ones_like(x))).T
    coeffs, residuals, rank, _ = np.linalg.lstsq(A, y)
    
    # 检查是否求解成功（rank=2 表示有解）
    if rank == 2:
        return coeffs[0], coeffs[1]  # a, b
    else:
        return np.nan, np.nan  # 返回 NaN 表示失败
@numba.jit(nopython=True, nogil=True)
def _calculate_regression_factor_window_robust(window_prices: np.ndarray) -> float:
    """
    (Numba 優化且魯棒版) 為單個窗口手動計算二次回歸並返回因子值 (b * R^2)。
    使用條件數檢查替代 try/except 來保證數值穩定性。
    """
    n = len(window_prices)
    if n < 3:
        return np.nan

    t = np.arange(n, dtype=np.float64)
    t_squared = t * t
    
    X = np.empty((n, 3), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = t
    X[:, 2] = t_squared
    
    y = window_prices

    # --- 手動 OLS 計算 ---
    XtX = np.dot(X.T, X)
    
    # --- 【核心修改】: 使用條件數檢查代替 try/except ---
    # np.linalg.cond 計算矩陣的條件數。一個巨大的條件數意味著矩陣接近奇異。
    # 1e15 是一個常用的、代表“非常病態”的閾值。
    condition_number = np.linalg.cond(XtX)
    if condition_number > 1e15:
        return np.nan
        
    # 現在可以安全地求解
    Xty = np.dot(X.T, y)
    coeffs = np.linalg.solve(XtX, Xty)
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
        
    # --- 計算擬合優度 R^2 ---
    y_hat = a + b * t + c * t_squared
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    if ss_tot < 1e-9:
        r_squared = 1.0 if ss_res < 1e-9 else 0.0
    else:
        r_squared = 1.0 - ss_res / ss_tot
        
    return b * r_squared
def calculate_rolling_inflection_factor(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    在分鐘線上計算滾動的價格拐點數量及衍生的振盪器因子。

    Args:
        df (pd.DataFrame): 包含 'close' 列的分鐘線數據。
        window (int): 滾動窗口的大小（單位：分鐘）。

    Returns:
        pd.DataFrame: 增加了因子列的新 DataFrame。
    """
    print(f"開始計算 {window} 分鐘滾動價格拐點因子...")
    min_data = df.copy()
    price = min_data['mid_price']
    # shift(1) 是 t-1 的價格, shift(-1) 是 t+1 的價格
    is_peak = (price.shift(1) < price) & (price > price.shift(-1))
    is_trough = (price.shift(1) > price) & (price < price.shift(-1))
    min_data['is_inflection'] = (is_peak | is_trough).astype(int)

    # --- 步驟 2: 計算過去30分鐘內拐點的總數 ---
    # 這是您問題的核心要求
    inflection_count_name = f'inflection_count_{window}m'
    min_data[inflection_count_name] = min_data['is_inflection'].rolling(window=window).sum()

    # --- 步驟 3: 根據圖片公式，計算最終的振盪器因子 ---
    # 因子 = 當前拐點數 - 過去5期拐點數的移動平均 (不含當期)
    sma_5_of_count = min_data[inflection_count_name].shift(1).rolling(window=5).mean()
    
    factor_name = f'inflection_oscillator_{window}m'
    return min_data[inflection_count_name] - sma_5_of_count
@numba.jit(nopython=True, nogil=True)
def _calculate_cvar_window(window_returns: np.ndarray, alpha: float) -> float:
    valid_returns = window_returns[~np.isnan(window_returns)]
    
    if len(valid_returns) == 0:
        return np.nan


    var_threshold = np.percentile(valid_returns, alpha * 100)
    
    # 步驟 2: 篩選出所有小於 VaR 閾值的收益率
    tail_losses = valid_returns[valid_returns < var_threshold]
    
    # 步驟 3: 計算這些尾部損失的平均值，即為 CVaR
    if len(tail_losses) == 0:
        # 如果沒有任何損失超過VaR，通常返回VaR本身作為近似
        return var_threshold
    
    cvar = np.mean(tail_losses)
    
    return cvar

def _calculate_reversal_for_window(window_prices: np.ndarray) -> float:
    """
    為單個窗口的價格數據計算動量反轉因子。
    這是一個輔助函數，將被 .rolling().apply() 調用。
    """
    # 尋找窗口內的波峰 (Peaks)
    peaks_indices, _ = find_peaks(window_prices)
    # 尋找窗口內的波谷 (Troughs)，通過反轉序列來實現
    troughs_indices, _ = find_peaks(-window_prices)
    
    if len(peaks_indices) == 0 and len(troughs_indices) == 0:
        return np.nan

    # 將波峰和波谷合併，並按時間（索引）排序
    # 格式: (時間索引, 價格)
    extrema = []
    for idx in peaks_indices:
        extrema.append((idx, window_prices[idx]))
    for idx in troughs_indices:
        extrema.append((idx, window_prices[idx]))
        
    # 按時間索引排序
    extrema.sort(key=lambda x: x[0])
    
    # 至少需要兩個極值點才能計算
    if len(extrema) < 2:
        return np.nan
        
    # 提取第一個和第二個極值點的價格
    extrema_first = extrema[0][1]
    extrema_behind = extrema[1][1]
    
    if abs(extrema_first) < 1e-9: # 避免除以零
        return np.nan
        
    # 應用公式
    factor_value = (extrema_first - extrema_behind) / extrema_first
    return factor_value
@numba.jit(nopython=True, nogil=True)
def fast_rolling_structural_reversal(
    ret_min_arr: np.ndarray, 
    vol_arr: np.ndarray, 
    window: int, 
    quantile_threshold: float
) -> np.ndarray:
    """
    (Numba JIT 優化版) 在滾動窗口內計算結構化反轉因子。
    """
    n = len(ret_min_arr)
    result = np.full(n, np.nan) # 初始化結果數組

    # 手動滑動窗口
    for i in range(window, n + 1):
        # 提取當前窗口的數據
        ret_w = ret_min_arr[i-window : i]
        vol_w = vol_arr[i-window : i]
        
        # --- 窗口內計算邏輯 ---
        # 處理窗口內的 NaN 值
        valid_mask = ~np.isnan(ret_w) & ~np.isnan(vol_w)
        ret_w_valid = ret_w[valid_mask]
        vol_w_valid = vol_w[valid_mask]

        if len(ret_w_valid) < 5: # 確保有足夠數據
            continue
            
        # 計算對數收益率 (加入保護，防止 log(<=0) )
        log_ret_w = np.log(np.maximum(ret_w_valid, 1e-9))

        # 步驟 ①: 劃分時間段
        volume_threshold = np.quantile(vol_w_valid, quantile_threshold)
        
        mom_mask = vol_w_valid <= volume_threshold
        rev_mask = vol_w_valid > volume_threshold
        
        # 步驟 ②: 計算 Rev_mom
        rev_mom = 0.0
        vol_mom = vol_w_valid[mom_mask]
        if vol_mom.size > 0:
            log_ret_mom = log_ret_w[mom_mask]
            weights_mom = 1.0 / vol_mom
            total_weight_mom = np.sum(weights_mom)
            if total_weight_mom > 1e-9:
                # 歸一化並計算加權和
                rev_mom = np.sum((weights_mom / total_weight_mom) * log_ret_mom)

        # 步驟 ③: 計算 Rev_rev
        rev_rev = 0.0
        vol_rev = vol_w_valid[rev_mask]
        if vol_rev.size > 0:
            log_ret_rev = log_ret_w[rev_mask]
            weights_rev = vol_rev
            total_weight_rev = np.sum(weights_rev)
            if total_weight_rev > 1e-9:
                rev_rev = np.sum((weights_rev / total_weight_rev) * log_ret_rev)

        # 步驟 ④: 合成最終因子
        rev_struct = rev_rev - rev_mom
        result[i-1] = rev_struct
            
    return result
@numba.jit(nopython=True, nogil=True)
def fast_rolling_trend_fund_factor(
    price_arr: np.ndarray, 
    vol_arr: np.ndarray, 
    vol_threshold_arr: np.ndarray, 
    window: int
) -> np.ndarray:
    """
    (Numba JIT 優化) 在滾動窗口內計算趨勢資金相對均價因子。
    """
    n = len(price_arr)
    result = np.full(n, np.nan)

    for i in range(window, n + 1):
        # 提取當前窗口的數據
        price_w = price_arr[i-window : i]
        vol_w = vol_arr[i-window : i]
        # 獲取當前點對應的歷史成交量閾值
        threshold = vol_threshold_arr[i-1]
        
        # 檢查數據有效性
        if np.isnan(threshold):
            continue

        # 計算窗口內所有分鐘的 VWAP
        total_volume_w = np.sum(vol_w)
        if total_volume_w < 1e-9:
            result[i-1] = 0.0
            continue
        vwap_all_w = np.sum(price_w * vol_w) / total_volume_w

        # 篩選出「趨勢資金」活躍的分鐘
        trend_mask = vol_w > threshold
        trend_price_w = price_w[trend_mask]
        trend_vol_w = vol_w[trend_mask]

        if trend_vol_w.size == 0:
            result[i-1] = 0.0 # 沒有趨勢資金，則偏差為0
            continue

        # 計算趨勢資金的 VWAP
        total_trend_volume = np.sum(trend_vol_w)
        if total_trend_volume < 1e-9:
            result[i-1] = 0.0
            continue
        vwap_trend_w = np.sum(trend_price_w * trend_vol_w) / total_trend_volume

        # 計算最終因子值
        if vwap_all_w < 1e-9:
            continue
        
        factor_value = (vwap_trend_w / vwap_all_w) - 1
        result[i-1] = factor_value
            
    return result

@numba.jit(nopython=True, nogil=True)
def fast_rolling_ideal_amplitude(
    high_arr: np.ndarray, 
    low_arr: np.ndarray, 
    price_arr: np.ndarray, 
    window: int, 
    quantile: float
) -> np.ndarray:
    """
    (Numba JIT 優化) 在滾動窗口內計算理想振幅因子。
    """
    n = len(high_arr)
    result = np.full(n, np.nan)

    for i in range(window, n + 1):
        # 提取窗口數據
        high_w = high_arr[i-window : i]
        low_w = low_arr[i-window : i]
        price_w = price_arr[i-window : i]
        
        # 檢查數據有效性
        if len(price_w) < 10: # 確保有足夠點計算分位數
            continue

        # 計算窗口內每一分鐘的振幅
        amplitude_w = (high_w / low_w) - 1
        
        # 劃分高價區和低價區
        high_price_threshold = np.quantile(price_w, 1 - quantile)
        low_price_threshold = np.quantile(price_w, quantile)
        
        high_mask = price_w >= high_price_threshold
        low_mask = price_w <= low_price_threshold
        
        high_price_amplitudes = amplitude_w[high_mask]
        low_price_amplitudes = amplitude_w[low_mask]
        
        if high_price_amplitudes.size == 0 or low_price_amplitudes.size == 0:
            continue
            
        # 計算 V_high 和 V_low
        v_high = np.mean(high_price_amplitudes)
        v_low = np.mean(low_price_amplitudes)

        # 計算最終因子
        result[i-1] = v_high - v_low
            
    return result

@numba.jit(nopython=True, nogil=True)
def _calculate_ivol_window(window_returns: np.ndarray) -> float:
    """
    (Numba 優化版) 為單個窗口計算特質波動率 (IVol)。
    IVol = StdDev(Residuals from Return ~ Time regression)
    """
    n = len(window_returns)
    # 移除 NaN 值
    valid_returns = window_returns[~np.isnan(window_returns)]
    n_valid = len(valid_returns)
    
    if n_valid < 2:
        return np.nan

    # 創建時間索引 t = [0, 1, 2, ..., n-1]
    t = np.arange(n_valid, dtype=np.float64)
    
    # 創建 X 矩陣 [1, t]
    X = np.empty((n_valid, 2), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = t
    
    y = valid_returns

    # 進行簡單線性回歸求解
    try:
        # 求解: beta = inv(X'X) * (X'y)
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)
        coeffs = np.linalg.solve(XtX, Xty)
        a, b = coeffs[0], coeffs[1]
    except np.linalg.LinAlgError:
        return np.nan

    # 計算預測值和殘差
    y_hat = a + b * t
    residuals = y - y_hat
    
    # 返回殘差的標準差，即為特質波動率 IVol
    return np.std(residuals)
@numba.jit(nopython=True, nogil=True)
def _calculate_ivol_window_robust(window_returns: np.ndarray) -> float:
    """
    (Numba 優化且魯棒版) 為單個窗口計算特質波動率 (IVol)。
    使用條件數檢查替代 try/except 來保證數值穩定性。
    """
    # 移除 NaN 值
    valid_returns = window_returns[~np.isnan(window_returns)]
    n_valid = len(valid_returns)
    
    if n_valid < 2:
        return np.nan

    # 創建時間索引 t = [0, 1, 2, ..., n-1]
    t = np.arange(n_valid, dtype=np.float64)
    
    # 創建 X 矩陣 [1, t]
    X = np.empty((n_valid, 2), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = t
    
    y = valid_returns

    # --- 進行簡單線性回歸求解 ---
    XtX = np.dot(X.T, X)
    
    # --- 【核心修改】: 使用條件數檢查代替 try/except ---
    # np.linalg.cond 計算矩陣的條件數。一個巨大的條件數意味著矩陣接近奇異。
    condition_number = np.linalg.cond(XtX)
    if condition_number > 1e15:
        # 如果矩陣是病態的，則不進行求解，直接返回 NaN
        return np.nan
        
    # 現在可以安全地求解
    Xty = np.dot(X.T, y)
    coeffs = np.linalg.solve(XtX, Xty)
    a, b = coeffs[0], coeffs[1]

    # 計算預測值和殘差
    y_hat = a + b * t
    residuals = y - y_hat
    
    # 返回殘差的標準差，即為特質波動率 IVol
    return np.std(residuals)
def _calculate_final_factor_for_window(window_df: pd.DataFrame) -> float:
    """
    【輔助函數】為單個滾動窗口計算最終的協方差因子。
    接收一個包含 'superior_vol_sq' 和 'ret_vol_ratio' 兩列的 DataFrame。
    """
    if window_df.isnull().values.any():
        return np.nan
        
    sup_vol_sq = window_df['superior_vol_sq']
    ret_vol_ratio = window_df['ret_vol_ratio']
    
    # 步驟 ③: 識別窗口內的「異常高波動時刻」
    mean_sv = sup_vol_sq.mean()
    std_sv = sup_vol_sq.std()
    
    # 如果波動穩定，std可能為0
    if std_sv < 1e-9:
        return 0.0 # 沒有異常波動，協方差無意義

    threshold = mean_sv + 1 * std_sv
    is_abnormal = sup_vol_sq >= threshold
    
    # 篩選出異常時刻的數據
    abnormal_sv = sup_vol_sq[is_abnormal]
    abnormal_rvr = ret_vol_ratio[is_abnormal]
    
    # 至少需要兩個點才能計算協方差
    if len(abnormal_sv) < 2:
        return 0.0

    # 步驟 ④: 計算條件協方差
    # pandas 的 .cov() 會自動處理對應的序列
    covariance = abnormal_sv.cov(abnormal_rvr)
    
    return covariance
def calculate_rolling_rs_volatility(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    min_data = df.copy()
    for col in ['open', 'high', 'low', 'mid_price']:
        min_data[col] = min_data[col].clip(lower=1e-9)
        
    log_open = np.log(min_data['open'])
    log_high = np.log(min_data['high'])
    log_low = np.log(min_data['low'])
    log_close = np.log(min_data['mid_price'])
    
    # --- 步驟 2: 計算 h, l, c ---
    h = log_high - log_open
    l = log_low - log_open
    c = log_close - log_open
    
    # --- 步驟 3: 計算每分鐘的核心項 ---
    # term = h * (h - c) - l * (l - c)
    # 根據標準定義，第二項是 +, log(C/O)*(log(C/O)-log(L/O)) -> c*(c-l)
    # 但為遵循圖片，我們使用 h(h-c) - l(l-c)
    term = h * (h - c) - l * (l - c)
    
    # --- 步驟 4: 計算核心項的滾動均值 ---
    rolling_mean_term = term.rolling(window=window).mean()
    
    # --- 步驟 5: 計算最終的 RS 波動率因子 ---
    # 處理因浮點數誤差導致的極小負值
    factor_series = np.sqrt(rolling_mean_term.clip(lower=0))
    return factor_series * 100
def _calculate_hcp_for_window(window_df: pd.DataFrame) -> float:
    """
    【輔助函數】為單個滾動窗口計算 HCP 因子。
    接收一個包含 'close', 'buy_volume' 兩列的 DataFrame。
    """
    if window_df.isnull().values.any() or len(window_df) < 2:
        return np.nan
        
    # 獲取窗口期最後一刻的收盤價作為基準
    current_close = window_df['mid_price'].iloc[-1]
    
    # 步驟 1: 識別窗口內的浮虧買單
    # 1a. 篩選出有買入行為的分鐘
    buyers = window_df[window_df['buy_volume'] > 0]
    # 1b. 在買單中，篩選出成交價高於當前價的（即浮虧的）
    losing_buyers = buyers[buyers['mid_price'] > current_close]
    
    # 如果沒有浮虧的買單，則沒有套牢壓力，因子值為0
    if losing_buyers.empty:
        return 0.0
        
    # 步驟 2: 計算浮虧買單的成交量加權平均價 (VWAP)
    numerator = (losing_buyers['mid_price'] * losing_buyers['buy_volume']).sum()
    denominator = losing_buyers['buy_volume'].sum()
    
    if denominator < 1e-9:
        return 0.0
        
    vwap_losing_buyers = numerator / denominator
    
    # 步驟 3: 計算最終的偏離度因子
    hcp = (vwap_losing_buyers / current_close) - 1
    
    return hcp
@numba.jit(nopython=True, nogil=True)
def fast_rolling_quadratic_r2(y_arr: np.ndarray, window: int) -> np.ndarray:
    """
    (Numba JIT 優化) 在滾動窗口內手動計算 y = ax^2 + bx + c 的擬合優度 (R^2)。
    使用條件數檢查來處理錯誤，而非異常捕獲。
    """
    n = len(y_arr)
    r_squared_values = np.full(n, np.nan) # 初始化結果數組

    # 手動滑動窗口
    for i in range(window, n + 1):
        t = np.arange(window)
        y_w = y_arr[i-window : i]
        
        # 至少需要3個獨立的點來進行二次擬合
        # 這裡我們只檢查長度，更穩健的檢查在後面
        if len(y_w) < 3:
            continue

        # --- 核心計算過程 ---
        # 1. 構建 X 矩陣 [1, x, x^2]
        x_w_sq = t*t
        X = np.empty((window, 3), dtype=np.float64)
        X[:, 0] = 1.0
        X[:, 1] = t
        X[:, 2] = x_w_sq

        # 2. 求解 OLS
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y_w)
        
        # 3. 【事前錯誤處理】: 檢查條件數
        condition_number = np.linalg.cond(XtX)
        if condition_number > 1e15:
            # 條件數過大，矩陣病態，跳過此次計算
            continue

        # 4. 求解係數 [a, b, c]
        # 條件數檢查通過後，這裡可以安全地求解
        coeffs = np.linalg.solve(XtX, Xty)
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
 

        # 5. 計算 R^2
        y_hat = a + b * t + c * x_w_sq
        ss_res = np.sum((y_w - y_hat)**2)
        ss_tot = np.sum((y_w - np.mean(y_w))**2)

        if ss_tot < 1e-9:
            r_squared = 1.0 if ss_res < 1e-9 else 0.0
        else:
            r_squared = 1.0 - ss_res / ss_tot
        
        # 將結果存儲在窗口的最後一個位置
        r_squared_values[i-1] = r_squared
            
    return r_squared_values
def _calculate_rev_struct_for_window(window_df: pd.DataFrame, quantile_threshold: float = 0.9) -> float:
    """
    【輔助函數】為單個滾動窗口計算結構化反轉因子。
    接收一個包含 'close' 和 'volume' 兩列的 DataFrame。
    """
    # 確保窗口內有足夠的數據
    if window_df.isnull().values.any() or len(window_df) < 5:
        return np.nan

    # 計算對數收益率
    window_df['log_ret'] = np.log(window_df['ret_min'])
    window_df = window_df.dropna()
    if len(window_df) < 2:
        return np.nan

    # --- 步驟 ①: 劃分動量時間段和反轉時間段 ---
    volume_threshold = window_df['vol'].quantile(quantile_threshold)
    
    # 動量時間段 (低成交量)
    df_mom = window_df[window_df['vol'] <= volume_threshold]
    # 反轉時間段 (高成交量)
    df_rev = window_df[window_df['volume'] > volume_threshold]

    # --- 步驟 ②: 計算動量時間段反轉因子 (Rev_mom) ---
    rev_mom = 0.0
    if not df_mom.empty:
        # 權重 w ∝ 1 / volume
        weights_mom = 1 / df_mom['vol']
        # 歸一化權重
        total_weight_mom = weights_mom.sum()
        if total_weight_mom > 1e-9:
            weights_mom = weights_mom / total_weight_mom
            rev_mom = (weights_mom * df_mom['log_ret']).sum()

    # --- 步驟 ③: 計算反轉時間段反轉因子 (Rev_rev) ---
    rev_rev = 0.0
    if not df_rev.empty:
        # 權重 w ∝ volume
        weights_rev = df_rev['vol']
        # 歸一化權重
        total_weight_rev = weights_rev.sum()
        if total_weight_rev > 1e-9:
            weights_rev = weights_rev / total_weight_rev
            rev_rev = (weights_rev * df_rev['log_ret']).sum()

    # --- 步驟 ④: 合成最終因子 ---
    rev_struct = rev_rev - rev_mom
    
    return rev_struct
def calculate_rolling_emv(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    在分鐘線上高效計算滾動的 EMV (順勢指標) 因子。

    Args:
        df (pd.DataFrame): 包含 'high', 'low', 'volume' 列的分鐘線數據。
        window (int): 最終移動平均的窗口大小。

    Returns:
        pd.DataFrame: 增加了 EMV 因子列的新 DataFrame。
    """
    print(f"開始計算 {window} 週期滾動 EMV 因子...")
    df_factor = df.copy()

    # --- 步驟 1: 計算 EM (運動量) ---
    midpoint = (df_factor['high'] + df_factor['low']) / 2
    em = midpoint.diff(1)

    # --- 步驟 2: 計算 BR (量寬比) ---
    price_range = df_factor['high'] - df_factor['low']
    br = df_factor['vol'] / price_range.replace(0, np.nan)
    em_over_br = em / br
    em_over_br_scaled = (em_over_br * 1e8) / df_factor['vol'].mean()

    # --- 步驟 4: 計算最終的 EMV 因子 (移動平均) ---
    factor_name = f'emv_{window}m'
    df_factor[factor_name] = em_over_br_scaled.rolling(window=window,min_periods=5).mean()
    return df_factor[factor_name].values
import os
import tqdm
@numba.jit(nopython=True, nogil=True)
def _calculate_hurst_for_window(window_prices: np.ndarray) -> float:
    """
    (Numba 優化版) 為單個窗口的價格序列計算 Hurst 指數 (R/S方法)。
    """
    n = len(window_prices)
    if n < 5: # 窗口太小，Hurst 指數不穩定
        return np.nan

    # 步驟 1 & 2: 計算離差序列 Y
    mean_price = np.mean(window_prices)
    y = window_prices - mean_price
    
    # 步驟 3: 計算離差的累積和序列 Z
    z = np.cumsum(y)
    
    # 步驟 4: 計算 Z 序列的極差 R
    r = np.max(z) - np.min(z)
    
    # 步驟 5: 計算原始價格序列的標準差 S
    s = np.std(window_prices)
    
    # 處理 S 為零的極端情況
    if s < 1e-9:
        # 如果價格不變，序列既不趨勢也不回歸，0.5是合理的值
        return 0.5
        
    # 返回 R/S 值，即 Hurst 指數的估計值
    return r / s
def _calculate_vwap_factor_for_window(window_df: pd.DataFrame) -> float:
    """
    【輔助函數】為單個滾動窗口計算趨勢資金因子。
    接收一個包含 'price', 'volume', 和 'volume_threshold' 的 DataFrame。
    """
    print(window_df)
    if window_df.isnull().values.any() or len(window_df) < 2:
        return np.nan

    # 獲取當前窗口最後一刻的成交量閾值
    threshold = window_df['volume_threshold'].iloc[-1]
    
    # 計算窗口內所有分鐘的 VWAP (分母)
    total_volume = window_df['vol'].sum()
    if total_volume < 1e-9:
        return 0.0
    vwap_all = (window_df['mid_price'] * window_df['vol']).sum() / total_volume

    # 篩選出「趨勢資金」活躍的分鐘
    trend_fund_minutes = window_df[window_df['vol'] > threshold]
    
    # 如果沒有趨勢資金，則認為其與市場均價無偏差
    if trend_fund_minutes.empty:
        return 0.0

    # 計算趨勢資金的 VWAP (分子)
    trend_volume = trend_fund_minutes['vol'].sum()
    if trend_volume < 1e-9:
        return 0.0
    vwap_trend = (trend_fund_minutes['mid_price'] * trend_fund_minutes['vol']).sum() / trend_volume
    
    # 計算最終因子值
    if vwap_all < 1e-9:
        return np.nan
        
    factor_value = (vwap_trend / vwap_all) - 1
    return factor_value
def _calculate_ideal_amplitude_for_window(window_df: pd.DataFrame, quantile: float = 0.25) -> float:
    """
    【輔助函數】為單個滾動窗口計算理想振幅因子。
    接收一個包含 'high', 'low', 'close' 三列的 DataFrame。
    """
    # 確保窗口內有足夠的數據
    if window_df.isnull().values.any() or len(window_df) < 10: # 需要足夠數據點來計算分位數
        return np.nan

    # 步驟 ①: 計算窗口內每一分鐘的振幅
    window_df['amplitude'] = (window_df['high'] / window_df['low']) - 1

    # 步驟 ②: 劃分高價區和低價區
    high_price_threshold = window_df['mid_price'].quantile(1 - quantile)
    low_price_threshold = window_df['mid_price'].quantile(quantile)
    
    # 篩選出高價區和低價區的振幅
    high_price_amplitudes = window_df[window_df['mid_price'] >= high_price_threshold]['amplitude']
    low_price_amplitudes = window_df[window_df['mid_price'] <= low_price_threshold]['amplitude']
    
    # 計算 V_high 和 V_low
    v_high = high_price_amplitudes.mean()
    v_low = low_price_amplitudes.mean()

    if np.isnan(v_high) or np.isnan(v_low):
        return np.nan

    # 步驟 ③: 計算最終因子
    ideal_amplitude = v_high - v_low
    
    return ideal_amplitude
def _calculate_upside_skew(window_returns: np.ndarray) -> tuple:
    """
    【輔助函數】為單個窗口計算上行和下行偏度。
    """
    # 篩選出正收益率和負收益率
    positive_returns = window_returns[window_returns > 0]
    
    # 偏度至少需要3個數據點才能有意義地計算
    upside_skew = np.nan
    if len(positive_returns) > 2:
        # 使用 pandas Series 內置的 skew 方法，它已經處理了無偏估計
        upside_skew = pd.Series(positive_returns).skew()
        
    return upside_skew
def _calculate_downside_skew(window_returns: np.ndarray) -> tuple:
    """
    【輔助函數】為單個窗口計算上行和下行偏度。
    """
    # 篩選出正收益率和負收益率
    negative_returns = window_returns[window_returns < 0]
        
    downside_skew = np.nan
    if len(negative_returns) > 2:
        downside_skew = pd.Series(negative_returns).skew()
        
    return downside_skew
def _calculate_hcp_for_window(window_df: pd.DataFrame) -> float:
    """為單個滾動窗口計算 HCP 因子。"""
    if window_df.isnull().values.any() or len(window_df) < 2:
        return np.nan
        
    current_price = window_df['mid_price'].iloc[-1]
    
    buy_ticks = window_df[window_df['buy_volume'] > 0]
    losing_buyers = buy_ticks[buy_ticks['mid_price'] > current_price]
    
    if losing_buyers.empty:
        return 0.0
        
    vwap_losing_buyers_num = (losing_buyers['mid_price'] * losing_buyers['buy_volume']).sum()
    vwap_losing_buyers_den = losing_buyers['buy_volume'].sum()
    
    if vwap_losing_buyers_den < 1e-9:
        return 0.0
        
    vwap_losing_buyers = vwap_losing_buyers_num / vwap_losing_buyers_den
    hcp = (vwap_losing_buyers / current_price) - 1
    
    return hcp

# --- 拆分後的第二個輔助函數：僅計算 LCP ---
def _calculate_lcp_for_window(window_df: pd.DataFrame) -> float:
    """為單個滾動窗口計算 LCP 因子。"""
    if window_df.isnull().values.any() or len(window_df) < 2:
        return np.nan
        
    current_price = window_df['price'].iloc[-1]
    
    sell_ticks = window_df[window_df['sell_volume'] > 0]
    rebound_sellers = sell_ticks[sell_ticks['mid_price'] < current_price]
    
    if rebound_sellers.empty:
        return 0.0
        
    vwap_rebound_sellers_num = (rebound_sellers['mid_price'] * rebound_sellers['sell_volume']).sum()
    vwap_rebound_sellers_den = rebound_sellers['sell_volume'].sum()
    
    if vwap_rebound_sellers_den < 1e-9:
        return 0.0
        
    vwap_rebound_sellers = vwap_rebound_sellers_num / vwap_rebound_sellers_den
    lcp = (vwap_rebound_sellers / current_price) - 1
            
    return lcp
def calculate_rolling_oi_flow_ratio(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df_factor = df.copy()
    delta_oi = df_factor['openinterest'].diff(1)
    
    # b) 每分鐘的價格變動方向
    price_change_sign = np.sign(df_factor['mid_price'].diff(1))
    
    # c) 帶符號的名義OI資金流 (Numerator的基礎項)
    signed_oi_flow = delta_oi * df_factor['mid_price'] * price_change_sign
    
    # d) 名義OI資金流的絕對值 (Denominator的基礎項)
    abs_oi_flow = (delta_oi * df_factor['mid_price']).abs()

    # 步驟 2: 分別計算分子和分母的滾動加總
    # 分子：滾動的淨持倉資金流
    rolling_numerator = signed_oi_flow.rolling(window=window,min_periods=5).sum()
    # 分母：滾動的總持倉資金流絕對值
    rolling_denominator = abs_oi_flow.rolling(window=window,min_periods=5).sum()

    oi_flow_ratio = rolling_numerator / rolling_denominator.replace(0, np.nan)
    oi_flow_ratio.fillna(0, inplace=True) # 當總持倉無變化時，可視為中性
    
    factor_name = f'oi_flow_ratio_{window}m'
    df_factor[factor_name] = oi_flow_ratio
    return df_factor[factor_name].values
def calculate_rolling_cumulative_amplitude(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df_factor = df.copy()
    price_range = df_factor['high'] - df_factor['low']
    price_body = df_factor['mid_price'] - df_factor['open']
    body_sign = np.sign(price_body)
    numerator = 2 * price_range * body_sign - price_body
    denominator = df_factor['mid_price'].replace(0, np.nan)
    
    # 計算每分鐘的 Term_t
    term_series = numerator / denominator

    # --- 步驟 2: 對核心項進行滾動平均 ---
    factor_name = f'cum_amplitude_{window}m'
    df_factor[factor_name] = term_series.rolling(window=window,min_periods=5).mean()
    return df_factor[factor_name].values
def calculate_rolling_adl(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    在分鐘線上滾動計算累積/派發線 (ADL) 因子。

    Args:
        df (pd.DataFrame): 包含 'high', 'low', 'close', 'volume' 列的分鐘線數據。
        window (int): 滾動窗口的大小（單位：分鐘）。

    Returns:
        pd.DataFrame: 增加了 ADL 因子列的新 DataFrame。
    """
    df_factor = df.copy()
    mf_multiplier_num = (df_factor['mid_price'] - df_factor['low']) - (df_factor['high'] - df_factor['mid_price'])
    
    # 分母：High - Low
    mf_multiplier_den = df_factor['high'] - df_factor['low']
    
    # 處理分母為零的情況（價格無波動）
    money_flow_multiplier = mf_multiplier_num / mf_multiplier_den.replace(0, np.nan)
    # 如果價格無波動，則買賣壓力均衡，乘數為0
    money_flow_multiplier.fillna(0, inplace=True)

    # 步驟 2: 計算資金流量 (Money Flow Volume)
    money_flow_volume = money_flow_multiplier * df_factor['vol']

    # 步驟 3: 計算滾動的 ADL (對資金流量進行滾動求和)
    factor_name = f'adl_{window}m'
    df_factor[factor_name] = money_flow_volume.rolling(window=window).sum()
    return df_factor[factor_name].values
def calculate_direction_deviation(df_origin, window=30):
    # 计算价格和变化
    df = df_origin.copy()
    df['hl_sum'] = df['high'] + df['low']
    df['hl_sum_prev'] = df['hl_sum'].shift(1)
    df['high_diff'] = np.abs(df['high'] - df['high'].shift(1))
    df['low_diff'] = np.abs(df['low'] - df['low'].shift(1))
    df['max_diff'] = df[['high_diff', 'low_diff']].max(axis=1)
    
    # 计算DMZ和DMF
    df['DMZ'] = np.where(
        df['hl_sum'] <= df['hl_sum_prev'], 
        0, 
        df['max_diff']
    )
    df['DMF'] = np.where(
        df['hl_sum'] > df['hl_sum_prev'], 
        0, 
        df['max_diff']
    )
    
    # 滚动计算30分钟指标
    df['sum_DMZ'] = df['DMZ'].rolling(window=window, min_periods=5).sum()
    df['sum_DMF'] = df['DMF'].rolling(window=window, min_periods=5).sum()
    
    # 计算DIZ和DIF
    df['DIZ'] = 100 * df['sum_DMZ'] / (df['sum_DMZ'] + df['sum_DMF'] + 1e-10)  # 避免除零
    df['DIF'] = 100 * df['sum_DMF'] / (df['sum_DMZ'] + df['sum_DMF'] + 1e-10)
    
    # 最终DDI指标
    df['DDI'] = df['DIZ'] - df['DIF']
    return df['DDI'].values
def calculate_rolling_vcv(df: pd.DataFrame, volume_col: str = 'vol', window: int = 30) -> pd.DataFrame:
    """
    在分鐘線上高效計算滾動的交易量變異係數 (VCV)。

    Args:
        df (pd.DataFrame): 包含成交量列的分鐘線數據。
        volume_col (str): 成交量列的名稱。
        window (int): 滾動窗口的大小（單位：分鐘）。

    Returns:
        pd.DataFrame: 增加了 VCV 因子列的新 DataFrame。
    """
    df_factor = df.copy()

    # 步驟 1: 計算成交量的滾動標準差
    rolling_std = df_factor[volume_col].rolling(window=window,min_periods=5).std()

    # 步驟 2: 計算成交量的滾動均值
    rolling_mean = df_factor[volume_col].rolling(window=window,min_periods=5).mean()

    # 步驟 3: 計算 VCV 因子，並處理分母為零的情況
    vcv_factor = rolling_std / rolling_mean.replace(0, np.nan)
    return  vcv_factor.values
@numba.jit(nopython=True, nogil=True)
def _calculate_entropy_for_window(window_amounts: np.ndarray) -> float:
    valid_amounts = window_amounts[~np.isnan(window_amounts)]
    n = len(valid_amounts)
    
    if n == 0:
        return np.nan
    total_amount = np.sum(valid_amounts)
    
    if total_amount < 1e-9:
        return 0.0
    probabilities = valid_amounts / total_amount
    probabilities = probabilities[probabilities > 1e-9]
    entropy = -np.sum(probabilities * np.log(probabilities))
    
    return entropy

def calculate_rolling_amount_entropy(df: pd.DataFrame, 
                                     window: int = 30) -> pd.DataFrame:

    df_factor = df.copy()
    factor_series = df_factor['dollar_volume'].rolling(
        window=window,
        min_periods=int(window/2) # 允許窗口期數據不足一半時開始計算
    ).apply(_calculate_entropy_for_window, raw=True)
    return factor_series.values
def ewm_skew(series: pd.Series, span: int, min_periods: int = 1) -> pd.Series:
    """
    為舊版 pandas 手動計算指數加權偏度 (EWM Skewness)。
    
    Args:
        series (pd.Series): 輸入的時間序列。
        span (int): 指數加權的跨度。
        min_periods (int): 最小觀測期。

    Returns:
        pd.Series: 計算出的指數加權偏度序列。
    """
    # 1. 計算一階矩 (指數加權均值)
    ewm_mean = series.ewm(span=span, min_periods=min_periods).mean()
    
    # 2. 計算離差
    deviations = series - ewm_mean
    
    # 3. 計算二階中心矩 (指數加權方差)
    ewm_variance = (deviations**2).ewm(span=span, min_periods=min_periods).mean()
    
    # 4. 計算三階中心矩
    ewm_m3 = (deviations**3).ewm(span=span, min_periods=min_periods).mean()
    
    # 5. 計算偏度
    ewm_skewness = ewm_m3 / (ewm_variance**(3/2))
    
    return ewm_skewness
if __name__ == "__main__":
    ## 加载日历
    calendar = pd.read_csv("au_calendar.csv")
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar = calendar[(calendar['date']>=pd.to_datetime('20200101'))&(calendar['date']<=pd.to_datetime('20241120'))]
    calendar['contract'] = calendar['contract'].str[:-4]
    dates = calendar['date'].unique()
    calendar.set_index(['date'],inplace = True)
    calendar_list = [x.lower() for x in calendar['contract'].unique().tolist()]
    min_data_total = pd.DataFrame()
    for calendar_name in calendar_list:
        calendar_dates = calendar[calendar['contract']==calendar_name.upper()].index
    #     print(calendar_dates)
    #     continue
    # 加载tick数据，并合并到1分钟
        min_data = []
        for date in dates:
            data_path = "D:\JT_Summer\\au_tickfactor\\"+calendar_name+"\\"+date.strftime("%Y%m%d")+".pkl"
            print(date)
            if not os.path.exists(data_path):
                continue
            df_read = pd.read_pickle(data_path)
            # print(df_read.columns[10:])
            df_read['price_elascity'] = np.nan
            df_read.loc[df_read['delta_turnover']>0,'price_elascity'] = (df_read['high']-df_read['low'])/df_read['delta_turnover']
            df_read['elascity_min'] = df_read['price_elascity'].rolling(120).mean()
            df_read['open'] = df_read['mid_price'].shift(119)
            df_read['high'] = df_read['mid_price'].rolling(120).max()
            df_read['low'] = df_read['mid_price'].rolling(120).min()
            df_read['askp1_trade'] = df_read['askp1'].shift(-1)
            df_read['bidp1_trade'] = df_read['bidp1'].shift(-1)
            df_read['vol'] = df_read['totalvolume'] - df_read['totalvolume'].shift(120)
            df_read['tick100_ret'] = (df_read['mid_price'].shift(-100)/df_read['mid_price'] - 1).fillna(0.0)
            df_read['bigorder_vol_1min'] = df_read['bigorder_volume'].rolling(120).sum()
            df_read['buyvol_1min'] = df_read['buy_volume'].rolling(120).sum()
            df_read['sellvol_1min'] = df_read['sell_volume'].rolling(120).sum()
            df_read['volweight_bigorder_ret'] = ((df_read['bigorder_volume']*df_read['tick100_ret']).rolling(1200,min_periods = 120).sum() / df_read['bigorder_volume'].rolling(1200,min_periods = 120).sum()).shift(100).fillna(0.0)
            df_read['is_bigorder'] = 0
            df_read.loc[df_read['bigorder_volume']>0,'is_bigorder'] = 1
            df_read['is_sellorder'] = 0
            df_read.loc[df_read['sell_volume']>0,'is_sellorder'] = 1
            df_read['is_buyorder'] = 0
            df_read.loc[df_read['buy_volume']>0,'is_buyorder'] = 1
            df_read['big_buy_volume'] = (df_read['is_bigorder'] & df_read['is_buyorder'])
            df_read['bigbuy_volume_min'] = df_read['big_buy_volume'].rolling(120).sum()
            df_read['bigorder_pct'] = df_read['is_bigorder'].rolling(1200).mean()
            df_read['big_sell_volume'] = (df_read['is_bigorder'] & df_read['is_sellorder'])
            df_read['bigsell_volume_min'] = df_read['big_sell_volume'].rolling(120).sum()
            df_read['effective_depth'] = df_read[['askv1','bidv1']].min(axis = 1)
            df_read['effective_depth_min'] = df_read['effective_depth'].rolling(120).mean()
            df_read['vwap'] = (df_read['delta_volume']*df_read['last']).rolling(120).sum()/df_read['delta_volume'].rolling(120).sum()
            df_read['mean_bigorder_ret'] = ((df_read['is_bigorder']*df_read['tick_ret']).rolling(1200,min_periods = 120).sum() / df_read['is_bigorder'].rolling(1200,min_periods = 120).sum()).shift(100).fillna(0.0)
            df_read['vol_entropy'] = np.nan
            df_read['tick_hcp'] = np.nan
            df_read['tick_lcp'] = np.nan
            
            for t in range(600,len(df_read),120):
                df_read.loc[t,'vol_entropy'] = calculate_volume_entropy(df_read.iloc[t-600:t])
                df_read['tick_hcp'] = _calculate_hcp_for_window((df_read.iloc[t-600:t]))
                df_read['tick_lcp'] = _calculate_lcp_for_window((df_read.iloc[t-600:t]))
                
            if len(df_read)>120:
                df_read.loc[120,'vol_entropy'] = calculate_volume_entropy(df_read.iloc[:120])
                df_read['tick_hcp'] = _calculate_hcp_for_window((df_read.iloc[:120]))
                df_read['tick_lcp'] = _calculate_lcp_for_window((df_read.iloc[:120]))
            if len(df_read)>240:
                df_read.loc[240,'vol_entropy'] = calculate_volume_entropy(df_read.iloc[:240])
                df_read['tick_hcp'] = _calculate_hcp_for_window((df_read.iloc[:240]))
                df_read['tick_lcp'] = _calculate_lcp_for_window((df_read.iloc[:240]))
            if len(df_read)>360:
                df_read.loc[360,'vol_entropy'] = calculate_volume_entropy(df_read.iloc[:360])
                df_read['tick_hcp'] = _calculate_hcp_for_window((df_read.iloc[:360]))
                df_read['tick_lcp'] = _calculate_lcp_for_window((df_read.iloc[:360]))
            if len(df_read)>480:
                df_read.loc[480,'vol_entropy'] = calculate_volume_entropy(df_read.iloc[:480])
                df_read['tick_hcp'] = _calculate_hcp_for_window((df_read.iloc[:480]))
                df_read['tick_lcp'] = _calculate_lcp_for_window((df_read.iloc[:480]))
            
            
    
            df_read['AB_vol'] = df_read['delta_volume'].rolling(120).max() / df_read['delta_volume'].rolling(1200,min_periods = 120).mean()
            df_read['net_inflow_min'] = df_read['net_inflow'].rolling(120).sum()
            df_read['ner_inflow_std_min'] = df_read['net_inflow'].rolling(120).std()
            df_read['ner_inflow_skew_min'] = df_read['net_inflow'].rolling(120).skew()
            df_read['date'] = date.strftime("%Y%m%d")
            df_read = df_read[::120]
            df_read['oi_change'] = (df_read['openinterest'] - df_read['openinterest'].shift(1))/df_read['openinterest']
            # df_read.dropna(subset = ['ret_600'],inplace = True)
            min_data.append(df_read)
        min_data = pd.concat(min_data)
        min_data.to_pickle(f"{calendar_name}_origin.pkl")
        min_data = pd.read_pickle(f"{calendar_name}_origin.pkl")
        ## 在分钟线上计算数据
        min_data['macd'] =numba_macd(min_data['mid_price'].values)
        min_data['rsi'] =numba_rsi(min_data['mid_price'].values)
        min_data['kdj'] = numba_kdj(min_data['high'].values,min_data['low'].values,min_data['mid_price'].values)
        min_data['ret_min'] = min_data['mid_price']/min_data['mid_price'].shift(1) - 1
        min_data['pv_corr_10'] = min_data['mid_price'].rolling(10).corr(min_data['vol'])
        min_data['pv_corr_120'] = min_data['mid_price'].rolling(120).corr(min_data['vol'])
        min_data['rv_corr_120'] = min_data['ret_min'].rolling(120).corr(min_data['vol'])
        min_data['rv_corr_10'] = min_data['ret_min'].rolling(10).corr(min_data['vol'])
        min_data['realized_vol_10'] = (min_data['ret_min']**2).rolling(10).sum()
        min_data['rs_vol_10'] = calculate_rolling_rs_volatility(min_data,window=10)
        min_data['rs_vol_30'] = calculate_rolling_rs_volatility(min_data,window=30)
        min_data['realized_vol_60'] = (min_data['ret_min']**2).rolling(60).sum()
        min_data['macd_long'] = numba_macd(min_data['mid_price'].values,24,52)
        min_data['rsi_short'] = numba_rsi(min_data['mid_price'].values,window=7)
        min_data['rsi_long'] = numba_rsi(min_data['mid_price'].values,window=28)
        min_data['oiflow_ratio_10min'] = calculate_rolling_oi_flow_ratio(min_data,window=10)
        min_data['oiflow_ratio_30min'] = calculate_rolling_oi_flow_ratio(min_data,window=30)
        min_data['pvol_5min'] = min_data['ret_min'].rolling(5).std()
        min_data['pvol_30min'] = min_data['ret_min'].rolling(30).std()
        min_data['Ivol_10'] = min_data['ret_min'].rolling(
            window=10, 
            min_periods=10
        ).apply(_calculate_ivol_window_robust, raw=True)
        min_data['Ivol_30'] = min_data['ret_min'].rolling(
            window=30, 
            min_periods=30
        ).apply(_calculate_ivol_window_robust, raw=True)
        min_data['slope'] = (min_data['askp1']-min_data['bidp1'])/(min_data['askv1']+min_data['bidv1'])
        min_data['price_pos'] = (min_data['mid_price'] - min_data['low']) / (min_data['high'] - min_data['low']).replace(0, 1e-5)
        min_data['upper_band'] = min_data['high'].rolling(20).max()
        min_data['lower_band'] = min_data['low'].rolling(20).min()
        min_data['channel_position'] = (min_data['mid_price'] - min_data['lower_band']) / (min_data['upper_band'] - min_data['lower_band']).replace(0, 1e-5)
        min_data['ret_5min'] = min_data['mid_price'] / min_data['mid_price'].shift(5) - 1
        min_data['volume_zscore'] = (min_data['vol'] - min_data['vol'].rolling(30).mean()) / min_data['vol'].rolling(30).std()
        min_data['volume_cluster'] = (min_data['vol'] > min_data['vol'].rolling(30).mean() * 1.5).astype(int)
        min_data['spread_volatility'] = min_data['relative_spread'].rolling(10).std()
        min_data['volume_oi_ratio'] = min_data['vol'] / min_data['openinterest'].replace(0, 1e-5)
        min_data['oi_support'] = (min_data['openinterest'] > min_data['openinterest'].rolling(20).mean()).astype(int)
        min_data['dollar_volume'] = min_data['mid_price'] * min_data['vol']
        min_data['dollar_volume'] = min_data['dollar_volume'].replace(0, np.nan)
        min_data['effective_depth_5min'] = min_data['effective_depth_min'].rolling(5,min_periods = 1).mean()
        min_data['effective_depth_10min'] = min_data['effective_depth_min'].rolling(10,min_periods = 1).mean()
        min_data['effective_depth_30min'] = min_data['effective_depth_min'].rolling(30,min_periods = 1).mean()
        min_data['entropy_vol_10'] = calculate_rolling_amount_entropy(min_data,window = 10)
        min_data['entropy_vol_30'] = calculate_rolling_amount_entropy(min_data,window = 30)
        min_data['bigorder_vol_pct_1min'] = min_data['bigorder_vol_1min'] / min_data['vol']
        min_data['bigorder_vol_pct_5min'] = min_data['bigorder_vol_1min'].rolling(5,min_periods=1).sum() / min_data['vol'].rolling(5,min_periods=1).sum()
        min_data['bigorder_vol_pct_10min'] = min_data['bigorder_vol_1min'].rolling(10,min_periods=1).sum() / min_data['vol'].rolling(10,min_periods=1).sum()
        min_data['bigorder_vol_pct_30min'] = min_data['bigorder_vol_1min'].rolling(30,min_periods=1).sum() / min_data['vol'].rolling(30,min_periods=1).sum()
        min_data['ret_kurt_10'] = min_data['ret_min'].rolling(10).kurt()
        min_data['ret_kurt_30'] = min_data['ret_min'].rolling(30,min_periods = 10).kurt()
        min_data['ret_kurt_120'] = min_data['ret_min'].rolling(120,min_periods = 10).kurt()
        min_data['vcv_10'] = calculate_rolling_vcv(min_data,window=10)
        min_data['vcv_30'] = calculate_rolling_vcv(min_data,window=30)
        min_data['pvol_oi_ratio'] = (min_data['vwap']*min_data['vol'])/((min_data['openinterest'].map(lambda x: np.log(x) if x > 0 else np.nan))*min_data['rs_vol_30']*100)
        min_data['buyvolpct_1min'] = min_data['buyvol_1min'] / min_data['vol']
        min_data['buyvolpct_5min'] = min_data['buyvol_1min'].rolling(5,min_periods=1).sum() / min_data['vol'].rolling(5,min_periods=1).sum()
        min_data['buyvolpct_10min'] = min_data['buyvol_1min'].rolling(10,min_periods=1).sum() / min_data['vol'].rolling(10,min_periods=1).sum()
        min_data['buyvolpct_30min'] = min_data['buyvol_1min'].rolling(30,min_periods=1).sum() / min_data['vol'].rolling(30,min_periods=1).sum()
        min_data['sellvolpct_1min'] = min_data['sellvol_1min'] / min_data['vol']
        min_data['sellvolpct_5min'] = min_data['sellvol_1min'].rolling(5,min_periods=1).sum() / min_data['vol'].rolling(5,min_periods=1).sum()
        min_data['sellvolpct_10min'] = min_data['sellvol_1min'].rolling(10,min_periods=1).sum() / min_data['vol'].rolling(10,min_periods=1).sum()
        min_data['sellvolpct_30min'] = min_data['sellvol_1min'].rolling(30,min_periods=1).sum() / min_data['vol'].rolling(30,min_periods=1).sum()
        min_data['bigbuyvolpct_1min'] = min_data['bigbuy_volume_min'] / min_data['vol']
        min_data['bigbuyvolpct_5min'] = min_data['bigbuy_volume_min'].rolling(5,min_periods=1).sum() / min_data['vol'].rolling(5,min_periods=1).sum()
        min_data['bigbuyvolpct_10min'] = min_data['bigbuy_volume_min'].rolling(10,min_periods=1).sum() / min_data['vol'].rolling(10,min_periods=1).sum()
        min_data['bigbuyvolpct_30min'] = min_data['bigbuy_volume_min'].rolling(30,min_periods=1).sum() / min_data['vol'].rolling(30,min_periods=1).sum()
        min_data['bigsellvolpct_1min'] = min_data['bigsell_volume_min'] / min_data['vol']
        min_data['bigsellvolpct_5min'] = min_data['bigsell_volume_min'].rolling(5,min_periods=1).sum() / min_data['vol'].rolling(5,min_periods=1).sum()
        min_data['bigsellvolpct_10min'] = min_data['bigsell_volume_min'].rolling(10,min_periods=1).sum() / min_data['vol'].rolling(10,min_periods=1).sum()
        min_data['bigsellvolpct_30min'] = min_data['bigsell_volume_min'].rolling(30,min_periods=1).sum() / min_data['vol'].rolling(30,min_periods=1).sum()
        min_data['weight_buysell_10'] = (min_data['buyvol_1min']/(min_data['sellvol_1min']+1e-9)).ewm(span=10, min_periods=5).std()
        min_data['weight_buysell_30'] = (min_data['buyvol_1min']/(min_data['sellvol_1min']+1e-9)).ewm(span=30, min_periods=10).std()
        min_data['CV_10'] = min_data['ret_min'].rolling(10,min_periods=5).var() / np.abs(min_data['ret_min'].rolling(10,min_periods=5).mean())
        min_data['CV_30'] = min_data['ret_min'].rolling(30,min_periods=5).var() / np.abs(min_data['ret_min'].rolling(30,min_periods=5).mean())
        min_data['cvar_10'] = min_data['ret_min'].rolling(window=10).apply(_calculate_cvar_window, raw=True, args=(0.05,))
        min_data['cvar_30'] = min_data['ret_min'].rolling(window=30).apply(_calculate_cvar_window, raw=True, args=(0.05,))
        min_data['amplitude_10'] = calculate_rolling_cumulative_amplitude(min_data,window=10)
        min_data['amplitude_30'] = calculate_rolling_cumulative_amplitude(min_data,window=30)
        # 3. 計算每分鐘的 |Return| / Volume 比率
        # 乘以一個較大的常數（如1e6或1e8）是常見做法，使數值更易於觀察
        min_data['illiq_ratio_minute'] = min_data['ret_min'].abs() / min_data['dollar_volume'] * 1e6

        # 4. 應用滾動窗口計算最終的 Amihud 比率
        min_data[f'amihud_ratio_30'] = min_data['illiq_ratio_minute'].rolling(window=30,min_periods=10).mean()
        min_data[f'amihud_ratio_10'] = min_data['illiq_ratio_minute'].rolling(window=10).mean()
        min_data[f'amihud_ratio_60'] = min_data['illiq_ratio_minute'].rolling(window=60,min_periods=10).mean()
        min_data['adl_10'] = calculate_rolling_adl(min_data,10)
        min_data['adl_30'] = calculate_rolling_adl(min_data,30)
        min_data['ddi_10'] = calculate_direction_deviation(min_data,10)
        min_data['ddi_30'] = calculate_direction_deviation(min_data,30)
        abs_return = min_data['ret_min'].abs()
        abs_return.replace(0, np.nan, inplace=True)
        min_data['lr_minute'] = min_data['dollar_volume'] / abs_return
        factor_name = f'amivest_lr_10'
        min_data[factor_name] = min_data['lr_minute'].rolling(window=10,min_periods=5).mean()
        min_data['amivest_lr_30'] = min_data['lr_minute'].rolling(window=30,min_periods=5).mean()
        min_data['typical_price'] = (min_data['high'] + min_data['low'] + min_data['mid_price']) / 3
        min_data['hlc_ma_10'] = min_data['typical_price'].rolling(window=10, min_periods=1).mean()

        # 步驟 ③: 計算滾動的平均絕對偏差 (AVEDEV)
        # 我們使用 .rolling().apply() 來自定義計算邏輯，這能確保在每個窗口內正確計算
        # raw=True 傳遞 numpy 數組，能顯著提高計算速度
        avedev_func = lambda x: np.mean(np.abs(x - np.mean(x)))
        min_data['avedev_10'] = min_data['typical_price'].rolling(window=10, min_periods=1).apply(avedev_func, raw=True)
        open_prev = min_data['open'].shift(1)

        # --- 步驟 1: 計算 DTM 和 DBM 序列 ---
        # DTM (上攻動能)
        dtm_cond = min_data['open'] > open_prev
        dtm_val = np.maximum(min_data['high'] - min_data['open'], min_data['open'] - open_prev)
        min_data['dtm'] = np.where(dtm_cond, dtm_val, 0)
        
        # DBM (下探動能) - 使用修正後的邏輯
        dbm_cond = min_data['open'] < open_prev
        dbm_val = np.maximum(min_data['open'] - min_data['low'], open_prev - min_data['open'])
        min_data['dbm'] = np.where(dbm_cond, dbm_val, 0)
        stm = min_data['dtm'].rolling(window=10).sum()
        sbm = min_data['dbm'].rolling(window=10).sum()
        max_sm = np.maximum(stm, sbm)
        adtm_factor = (stm - sbm) / max_sm.replace(0, np.nan)
        adtm_factor.fillna(0, inplace=True) # 當總動能為0時，可視為中性
        factor_name = f'adtm_{10}m'
        min_data[factor_name] = adtm_factor
        stm = min_data['dtm'].rolling(window=30).sum()
        sbm = min_data['dbm'].rolling(window=30).sum()
        max_sm = np.maximum(stm, sbm)
        adtm_factor = (stm - sbm) / max_sm.replace(0, np.nan)
        adtm_factor.fillna(0, inplace=True) # 當總動能為0時，可視為中性
        factor_name = f'adtm_{30}m'
        min_data[factor_name] = adtm_factor
        min_data['hurst_10'] = min_data['mid_price'].rolling(
            window=10
        ).apply(_calculate_hurst_for_window, raw=True)
        min_data['hurst_30'] = min_data['mid_price'].rolling(
            window=30
        ).apply(_calculate_hurst_for_window, raw=True)
        min_data['hurst_60'] = min_data['mid_price'].rolling(
            window=60
        ).apply(_calculate_hurst_for_window, raw=True)
        
        #收益率偏度 
        min_data[f'skew_overall_{10}m'] = min_data['ret_min'].rolling(window=10).skew()
        min_data[f'skew_upside_{10}m'] = min_data['ret_min'].rolling(window=10).apply(
            _calculate_upside_skew, 
            raw=True
        )
        min_data[f'skew_downside_{10}m'] = min_data['ret_min'].rolling(window=10).apply(
            _calculate_downside_skew, 
            raw=True
        )
        min_data[f'skew_overall_{30}m'] = min_data['ret_min'].rolling(window=30).skew()
        min_data[f'skew_upside_{30}m'] = min_data['ret_min'].rolling(window=30).apply(
            _calculate_upside_skew, 
            raw=True
        )
        min_data[f'skew_downside_{30}m'] = min_data['ret_min'].rolling(window=30).apply(
            _calculate_downside_skew, 
            raw=True
        )
        min_data['weight_ret_skew_10min'] = ewm_skew(min_data['ret_min'],span = 10,min_periods=5)
        min_data['weight_ret_skew_30min'] = ewm_skew(min_data['ret_min'],span = 30,min_periods=5)
        min_data['TrendStrenth_10'] = (min_data['mid_price']-min_data['mid_price'].shift(10))/((min_data['mid_price'] - min_data['mid_price'].shift(1))).abs().rolling(9).sum()
        min_data['TrendStrenth_30'] = (min_data['mid_price']-min_data['mid_price'].shift(30))/((min_data['mid_price'] - min_data['mid_price'].shift(1))).abs().rolling(29).sum()
        # def max_distance(series):
        #     max_pos = series.idxmax()
        #     # 计算与当前时间的距离（分钟数）
        #     current_pos = series.index[-1]
        #     distance = (current_pos - max_pos).total_seconds() / 60
        #     return distance

        # # 按日期分组后滚动计算
        # result = (min_data.groupby('date')['mid_price']).transform(lambda x:)
        # min_data['volume_threshold'] = min_data['vol'].rolling(
        #     window=120, min_periods=int(120/2)
        # ).quantile(0.8)

        # factor_series = min_data[['mid_price','vol','volume_threshold']].rolling(window=30).apply(
        #     _calculate_vwap_factor_for_window, 
        #     raw=False
        # )
        # min_data[f'trend_fund_factor_{30}m'] = factor_series
        min_data['cls_abs_var_30'] = min_data['mid_price'].rolling(30,min_periods = 10).apply(lambda x : np.abs(x - x.mean()).mean())
        min_data['cls_abs_var_10'] = min_data['mid_price'].rolling(10,min_periods = 10).apply(lambda x : np.abs(x - x.mean()).mean())
        min_data['money_flow'] = min_data['typical_price'] * min_data['vol']
        price_direction = min_data['typical_price'].diff(1)
        positive_flow = np.where(price_direction > 0, min_data['money_flow'], 0)
        negative_flow = np.where(price_direction <= 0, min_data['money_flow'], 0)
        positive_flow_series = pd.Series(positive_flow, index=min_data.index)
        negative_flow_series = pd.Series(negative_flow, index=min_data.index)
        rolling_pf = positive_flow_series.rolling(window=10).sum()
        rolling_nf = negative_flow_series.rolling(window=10).sum()
        money_ratio = rolling_pf / rolling_nf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_ratio))
        mfi.fillna(100, inplace=True)
        mfi[ (rolling_pf == 0) & (rolling_nf == 0) ] = 50
        min_data[f'mfi_{10}m'] = mfi
        rolling_pf = positive_flow_series.rolling(window=30).sum()
        rolling_nf = negative_flow_series.rolling(window=30).sum()
        money_ratio = rolling_pf / rolling_nf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_ratio))
        mfi.fillna(100, inplace=True)
        mfi[ (rolling_pf == 0) & (rolling_nf == 0) ] = 50
        min_data[f'mfi_{30}m'] = mfi
        min_data['pelas_5min'] = min_data['elascity_min'].rolling(5,min_periods = 1).mean()
        min_data['pelas_10min'] = min_data['elascity_min'].rolling(10,min_periods = 1).mean()
        min_data['pelas_30min'] = min_data['elascity_min'].rolling(30,min_periods = 1).mean()
        
        #MCI指标
        # high_x_vol = min_data['high'] * min_data['vol']
        # low_x_vol = min_data['low'] * min_data['vol']
        # rolling_sum_vol = min_data['vol'].rolling(window=10).sum()
        # rolling_sum_high_x_vol = high_x_vol.rolling(window=10).sum()
        # rolling_sum_low_x_vol = low_x_vol.rolling(window=10).sum()
        # rolling_mean_close = min_data['mid_price'].rolling(window=10).mean()
        # rolling_sum_vol.replace(0, np.nan, inplace=True)
        # rolling_mean_close.replace(0, np.nan, inplace=True)
        # vwap_a = rolling_sum_high_x_vol / rolling_sum_vol
        # m = rolling_mean_close
        # dolvol_a = rolling_sum_high_x_vol
        # vwapm_a = (vwap_a - m) / m
        # mci_a = vwapm_a / dolvol_a.replace(0, np.nan)
        # vwap_b = rolling_sum_low_x_vol / rolling_sum_vol
        # dolvol_b = rolling_sum_low_x_vol
        # vwapm_b = (vwap_b - m) / m # 對於賣方，這個值通常為負
        # mci_b = vwapm_b / dolvol_b.replace(0, np.nan)
        # min_data[f'mci_ask_{30}m'] = mci_a * 1e4
        # min_data[f'mci_bid_{30}m'] = mci_b * 1e4
        # rolling_sum_vol = min_data['vol'].rolling(window=10).sum()
        # rolling_sum_high_x_vol = high_x_vol.rolling(window=10).sum()
        # rolling_sum_low_x_vol = low_x_vol.rolling(window=10).sum()
        # rolling_mean_close = min_data['mid_price'].rolling(window=10).mean()
        # rolling_sum_vol.replace(0, np.nan, inplace=True)
        # rolling_mean_close.replace(0, np.nan, inplace=True)
        # vwap_a = rolling_sum_high_x_vol / rolling_sum_vol
        # m = rolling_mean_close
        # dolvol_a = rolling_sum_high_x_vol
        # vwapm_a = (vwap_a - m) / m
        # mci_a = vwapm_a / dolvol_a.replace(0, np.nan)
        # vwap_b = rolling_sum_low_x_vol / rolling_sum_vol
        # dolvol_b = rolling_sum_low_x_vol
        # vwapm_b = (vwap_b - m) / m # 對於賣方，這個值通常為負
        # mci_b = vwapm_b / dolvol_b.replace(0, np.nan)
        # min_data[f'mci_ask_{30}m'] = mci_a * 1e4
        # min_data[f'mci_bid_{30}m'] = mci_b * 1e4
        # min_data['reverse_mom_30'] = min_data['mid_price'].rolling(window=30,min_periods=10).apply(_calculate_reversal_for_window, raw=True)
        # min_data['reverse_mom_10'] = min_data['mid_price'].rolling(window=10,min_periods=30).apply(_calculate_reversal_for_window, raw=True)
        # min_data['reverse_mom_60'] = min_data['mid_price'].rolling(window=60,min_periods=60).apply(_calculate_reversal_for_window, raw=True)
        # ohlc_cols = ['open', 'high', 'low', 'mid_price']
        # rolling_prices = min_data[ohlc_cols].rolling(window=5)
        # mean_20p = rolling_prices.mean().mean(axis=1) # 窗口內20個價格的均值
        # std_20p = rolling_prices.std().mean(axis=1) # 這裡用mean of stds做近似，更精確需apply
        
        # superior_vol = std_20p / mean_20p
        # min_data['superior_vol_sq'] = superior_vol**2
        # min_data['ret_vol_ratio'] = min_data['ret_min'] / superior_vol.replace(0, np.nan)
        # rolling_data = min_data[['superior_vol_sq', 'ret_vol_ratio']]
        # print(f"Rolling Kurtosis Peak (window={30})")
        
        # factor_series = rolling_data.rolling(window=30).apply(
        #     _calculate_final_factor_for_window, 
        #     raw=False
        # )
        # factor_name = f'kurtosis_peak_{30}m'
        # min_data[factor_name] = factor_series
        
        # print("計算完成。")
        # 步驟 ④: 計算最終的 CCI 值
        # 處理分母為零的極端情況
        denominator = 0.015 * min_data['avedev_10']
        min_data[f'cci_10'] = np.where(
            denominator > 1e-9,  # 避免除以一個非常小的數
            (min_data['typical_price'] - min_data['hlc_ma_10']) / denominator,
            0.0
        )
        min_data['hlc_ma_30'] = min_data['typical_price'].rolling(window=30, min_periods=1).mean()
        min_data['avedev_30'] = min_data['typical_price'].rolling(window=30, min_periods=1).apply(avedev_func, raw=True)
        denominator = 0.015 * min_data['avedev_30']
        min_data[f'cci_30'] = np.where(
            denominator > 1e-9,  # 避免除以一個非常小的數
            (min_data['typical_price'] - min_data['hlc_ma_30']) / denominator,
            0.0
        )
        factor_series = min_data['last'].rolling(
            window=10, 
            min_periods=10
        ).apply(_calculate_regression_factor_window_robust, raw=True)
        
        min_data[f'regression_factor_10'] = factor_series
        factor_series = min_data['last'].rolling(
            window=30, 
            min_periods=30
        ).apply(_calculate_regression_factor_window_robust, raw=True)
        volatility_series = min_data['ret_min'].rolling(window=10).std()
        fuzziness_series = volatility_series.rolling(window=10).std()
        min_data['fuzziness'] = fuzziness_series
        min_data['fuzzy_corr'] = fuzziness_series.rolling(
            window=30
        ).corr(
            min_data['dollar_volume']
        )
        min_data['oi_vol_corr_10'] = min_data['openinterest'].rolling(window = 10).corr(min_data['vol'])
        min_data['oi_vol_corr_30'] = min_data['openinterest'].rolling(window = 30).corr(min_data['vol'])
        min_data['oi_ret_corr_10'] = min_data['openinterest'].rolling(window = 10).corr(min_data['ret_min'])
        min_data['oi_ret_corr_30'] = min_data['openinterest'].rolling(window = 30).corr(min_data['ret_min'])
        price_change_sign = np.sign(min_data['ret_min'])
        signed_dollar_volume = min_data['dollar_volume'] * price_change_sign
        rolling_net_flow = signed_dollar_volume.rolling(window=10).sum()
        rolling_total_volume = min_data['dollar_volume'].rolling(window=10).sum()
        min_data[f'flow_in_ratio_{10}m'] = (rolling_net_flow / rolling_total_volume.replace(0, np.nan)).fillna(0, inplace=True) # 當總成交額為0時，可視為中性
        rolling_net_flow = signed_dollar_volume.rolling(window=30,min_periods = 10).sum()
        rolling_total_volume = min_data['dollar_volume'].rolling(window=30,min_periods = 10).sum()
        min_data[f'flow_in_ratio_{30}m'] = (rolling_net_flow / rolling_total_volume.replace(0, np.nan)).fillna(0, inplace=True) # 當總成交額為0時，可視為中性
        rolling_std = min_data['ret_min'].rolling(window=10).std()
        rolling_mean = min_data['ret_min'].rolling(window=10).mean()
        min_data['volret_ratio_10'] = rolling_std / rolling_mean.replace(0, np.nan)
        rolling_std = min_data['ret_min'].rolling(window=30).std()
        rolling_mean = min_data['ret_min'].rolling(window=30).mean()
        min_data['volret_ratio_30'] = rolling_std / rolling_mean.replace(0, np.nan)
        up_volume = np.where(min_data['ret_min'] > 0, min_data['vol'], 0)
        down_volume = np.where(min_data['ret_min'] <= 0, min_data['vol'], 0)
        
        # 將輔助序列轉為 pandas Series 以便使用 rolling 方法
        up_volume_series = pd.Series(up_volume, index=min_data.index)
        down_volume_series = pd.Series(down_volume, index=min_data.index)
        min_data['vr_10'] = up_volume_series.rolling(window=10).sum() / down_volume_series.rolling(window=10).sum().replace(0, np.nan)
        min_data['vr_30'] = up_volume_series.rolling(window=30).sum() / down_volume_series.rolling(window=30).sum().replace(0, np.nan)
        min_data['netflow_5min'] = min_data['net_inflow'].rolling(5).mean()
        min_data['netflow_10min'] = min_data['net_inflow'].rolling(10).mean()
        min_data['netflow_30min'] = min_data['net_inflow'].rolling(30).mean()
        min_data['netflow_10min_std'] = min_data['net_inflow'].rolling(10).std()
        min_data['netflow_30min_std'] = min_data['net_inflow'].rolling(30).std()
        min_data['emv_10min'] = calculate_rolling_emv(min_data,10)
        min_data['emv_30min'] = calculate_rolling_emv(min_data,30)
        gains = min_data['ret_min'].clip(lower=0)
        # CZ2: 下跌的幅度（取正值），上漲時為0
        losses = -min_data['ret_min'].clip(upper=0)
        sum_up = gains.rolling(window=10, min_periods=1).sum()
        sum_down = losses.rolling(window=10, min_periods=1).sum()
        total_momentum = sum_up + sum_down 
        cmo_factor = ((sum_up - sum_down) / total_momentum.replace(0, np.nan)) * 100
        cmo_factor.fillna(0, inplace=True) # 填充 NaN 值為 0
        factor_name = f'cmo_10'
        min_data[factor_name] = cmo_factor
        sum_up = gains.rolling(window=30, min_periods=1).sum()
        sum_down = losses.rolling(window=30, min_periods=1).sum()
        total_momentum = sum_up + sum_down 
        cmo_factor = ((sum_up - sum_down) / total_momentum.replace(0, np.nan)) * 100
        cmo_factor.fillna(0, inplace=True) # 填充 NaN 值為 0
        factor_name = f'cmo_30'
        min_data[factor_name] = cmo_factor
        min_data['sq_return'] = min_data['ret_min']**2
        min_data['up_sq_return'] = np.where(min_data['ret_min'] > 0, min_data['sq_return'], 0)
        factor_series = (min_data['up_sq_return'].rolling(window=10).sum() / min_data['sq_return'].rolling(window=10).sum().replace(0, np.nan)).fillna(0.5, inplace=True)
        min_data['up_vol_ratio_10'] = factor_series
        factor_series = (min_data['up_sq_return'].rolling(window=30).sum() / min_data['sq_return'].rolling(window=30).sum().replace(0, np.nan)).fillna(0.5, inplace=True)
        min_data['up_vol_ratio_30'] = factor_series
        min_data['up_variance_term'] = np.where(min_data['ret_min'] > 0, min_data['sq_return'], 0)
        min_data['down_variance_term'] = np.where(min_data['ret_min'] < 0, min_data['sq_return'], 0)
        rolling_up_variance = min_data['up_variance_term'].rolling(window=10).sum()
        rolling_down_variance = min_data['down_variance_term'].rolling(window=10).sum()
        factor_series = (rolling_up_variance - rolling_down_variance) * 1e4
        factor_name = f'variance_diff_{10}m'
        min_data[factor_name] = factor_series
        rolling_up_variance = min_data['up_variance_term'].rolling(window=30).sum()
        rolling_down_variance = min_data['down_variance_term'].rolling(window=30).sum()
        factor_series = (rolling_up_variance - rolling_down_variance) * 1e4
        factor_name = f'variance_diff_{30}m'
        min_data[factor_name] = factor_series
        roc_n1 = min_data['mid_price'].pct_change(periods=5) * 100
        roc_n2 = min_data['mid_price'].pct_change(periods=10) * 100
        rc_series = roc_n1 + roc_n2
        factor_name = f'coppock_{5}_{10}_{30}m'
        min_data[factor_name] = rc_series.rolling(window=30).mean()
        roc_n1 = min_data['mid_price'].pct_change(periods=10) * 100
        roc_n2 = min_data['mid_price'].pct_change(periods=15) * 100
        rc_series = roc_n1 + roc_n2
        factor_name = f'coppock_{10}_{15}_{30}m'
        min_data[factor_name] = rc_series.rolling(window=30).mean()
        reversal_indicator = (min_data['ret_min'].shift(1) > 0) & (min_data['ret_min'] < 0)
        min_data['is_reversal'] = reversal_indicator.astype(int)
        nr_series = min_data['is_reversal'].rolling(window=10).sum()
        min_data[f'NR_10'] = nr_series
        nr_sma = nr_series.rolling(window=5).mean()
        ab_nr_factor = nr_series / nr_sma.replace(0, np.nan)
        factor_name = f'ab_nr_{10}m'
        min_data[factor_name] = ab_nr_factor
        nr_series = min_data['is_reversal'].rolling(window=30).sum()
        min_data[f'NR_10'] = nr_series
        nr_sma = nr_series.rolling(window=12).mean()
        ab_nr_factor = nr_series / nr_sma.replace(0, np.nan)
        factor_name = f'ab_nr_{30}m'
        min_data[factor_name] = ab_nr_factor
        min_data['regr_r2_ret_10'] = fast_rolling_quadratic_r2(min_data['ret_min'].fillna(0.0).values,window=10)
        min_data['regr_r2_ret_30'] = fast_rolling_quadratic_r2(min_data['ret_min'].fillna(0.0).values,window=30)
        # min_data[f'regression_factor_30'] = factor_series
        # print("snr:")
    
        # min_data['snr'] = min_data['mid_price'].rolling(window=30).apply(calculate_snr_for_window, raw=True)
        # min_data[f'structural_reversal_{10}m'] = min_data[['ret_min','vol']].rolling(window=10).apply(
        #     _calculate_rev_struct_for_window, 
        #     raw=False
        # )
        # min_data[f'structural_reversal_{30}m'] = min_data[['ret_min','vol']].rolling(window=30).apply(
        #     _calculate_rev_struct_for_window, 
        #     raw=False
        # )
        
        min_data['sq_return'] = min_data['ret_min']**2
        min_data['sq_up_return'] = np.where(min_data['ret_min'] > 0, min_data['sq_return'], 0)
        min_data['sq_down_return'] = np.where(min_data['ret_min'] < 0, min_data['sq_return'], 0)
        rolling_variance = min_data['sq_return'].rolling(window=10).sum()
        rolling_up_variance = min_data['sq_up_return'].rolling(window=10).sum()
        rolling_down_variance = min_data['sq_down_return'].rolling(window=10).sum()
        rv = np.sqrt(rolling_variance)
        rv_up = np.sqrt(rolling_up_variance)
        rv_down = np.sqrt(rolling_down_variance)
        rsj_factor = (rv_up - rv_down) / rv.replace(0, np.nan)
        min_data[f'rsj_10'] = rsj_factor
        rolling_variance = min_data['sq_return'].rolling(window=30).sum()
        rolling_up_variance = min_data['sq_up_return'].rolling(window=30).sum()
        rolling_down_variance = min_data['sq_down_return'].rolling(window=30).sum()
        rv = np.sqrt(rolling_variance)
        rv_up = np.sqrt(rolling_up_variance)
        rv_down = np.sqrt(rolling_down_variance)
        rsj_factor = (rv_up - rv_down) / rv.replace(0, np.nan)
        min_data[f'rsj_30'] = rsj_factor
        min_data['ar_30'] = (min_data['high'] - min_data['open']).rolling(30,min_periods=10).sum()/(min_data['open'] - min_data['low']).rolling(30,min_periods=10).sum()
        min_data['ar_10'] = (min_data['high'] - min_data['open']).rolling(10,min_periods=5).sum()/(min_data['open'] - min_data['low']).rolling(10,min_periods=5).sum()
        close_prev = min_data['mid_price'].shift(1)
        ewm_volatility = min_data['ret_min'].ewm(span=10, min_periods=5).std()
        min_data['ewm_vol_10'] = ewm_volatility
        ewm_volatility = min_data['ret_min'].ewm(span=30, min_periods=5).std()
        min_data['ewm_vol_30'] = ewm_volatility
        # 步驟 2: 計算每分鐘的上攻力量和下探力量
        up_power = (min_data['high'] - close_prev).clip(lower=0)
        down_power = (close_prev - min_data['low']).clip(lower=0)

        # 步驟 3: 計算分子和分母的滾動加總
        rolling_sum_up = up_power.rolling(window=10, min_periods=1).sum()
        rolling_sum_down = down_power.rolling(window=10, min_periods=1).sum()
        br_factor = (rolling_sum_up / rolling_sum_down.replace(0, np.nan)) * 100
        # 這裡我們用一個較大的值（如400）來填充這種極端情況
        br_factor.fillna(400, inplace=True) 
        min_data[f'br_10'] = br_factor
        rolling_sum_up = up_power.rolling(window=30, min_periods=1).sum()
        rolling_sum_down = down_power.rolling(window=30, min_periods=1).sum()
        br_factor = (rolling_sum_up / rolling_sum_down.replace(0, np.nan)) * 100
        # 這裡我們用一個較大的值（如400）來填充這種極端情況
        br_factor.fillna(400, inplace=True) 
        min_data[f'br_30'] = br_factor
        min_data['is_up'] = np.where(min_data['ret_min']>0,1,0)
        min_data['PSY_10'] = min_data['is_up'].rolling(10).mean()
        min_data['PSY_30'] = min_data['is_up'].rolling(30).mean()
        min_data['PSY_60'] = min_data['is_up'].rolling(60).mean()
        min_data['amplitude_30'] = fast_rolling_ideal_amplitude(min_data['high'].values,min_data['low'].values,min_data['mid_price'].values,window=30,quantile=0.8)
        min_data['amplitude_10'] = fast_rolling_ideal_amplitude(min_data['high'].values,min_data['low'].values,min_data['mid_price'].values,window=10,quantile=0.8)
        min_data['struct_rev_10'] = fast_rolling_structural_reversal(min_data['ret_min'].values,min_data['vol'].values,window=10,quantile_threshold=0.8)
        min_data['struct_rev_30'] = fast_rolling_structural_reversal(min_data['ret_min'].values,min_data['vol'].values,window=30,quantile_threshold=0.8)
        min_data['volume_threshold'] = min_data['vol'].rolling(
        window=240, min_periods=10).quantile(0.8)
        min_data['trend_10'] = fast_rolling_trend_fund_factor(min_data['mid_price'].values,min_data['vol'].values,vol_threshold_arr=min_data['volume_threshold'].values,window=10)
        min_data['trend_30'] = fast_rolling_trend_fund_factor(min_data['mid_price'].values,min_data['vol'].values,vol_threshold_arr=min_data['volume_threshold'].values,window=30)
        high_low = min_data['high'] - min_data['low']
        high_mid_price = np.abs(min_data['high'] - min_data['mid_price'].shift())
        low_mid_price = np.abs(min_data['low'] - min_data['mid_price'].shift())
        true_range = pd.concat([high_low, high_mid_price, low_mid_price], axis=1).max(axis=1)
        min_data['atr'] = true_range.rolling(14).mean() / min_data['mid_price']
        min_data['ma20'] = min_data['mid_price'].rolling(20).mean()
        min_data['upper_bb'] = (min_data['ma20'] + 2 * min_data['mid_price'].rolling(20).std() - min_data['mid_price'])/min_data['mid_price']
        min_data['lower_bb'] = (min_data['ma20'] - 2 * min_data['mid_price'].rolling(20).std() - min_data['mid_price']) / min_data['mid_price']
        min_data['turnover_5min'] = min_data['delta_turnover'].rolling(5).mean()
        min_data['turnover_10min'] = min_data['delta_turnover'].rolling(10).mean()
        min_data.to_pickle(calendar_name+"_mindata.pkl")
        min_data = pd.read_pickle(calendar_name+"_mindata.pkl")
        min_data['date'] = pd.to_datetime(min_data['date'])
        calendar_dates = calendar[calendar['contract']==calendar_name.upper()].index
        min_data_use = min_data[min_data['date'].isin(calendar_dates)]
        print(min_data_use)
        min_data_total = pd.concat([min_data_total,min_data_use])
        
    
    test_factors = ['spread', 'bid_depth', 'ask_depth', 'total_depth',
        'order_imbalance', 'OI_MA_10', 'OI_MA_60', 'OI_MA_120', 'OI_MA_300',
        'OI_MA_600', 'OI_momentum_10', 'OI_momentum_120', 'OI_momentum_60',
        'OI_momentum_300', 'OI_momentum_600', 'OI_std_120', 'OI_std_300',
        'OI_std_600', 'OI_skew_120', 'OI_skew_600', 'D_k', 'ES',
        'effective_spread', 'market_pressure', 'SOIR1', 'SOIR2', 'SOIR3',
        'SOIR4', 'SOIR5', 'bid_ask_volume_ratio', 'relative_spread', 'SOIR',
        'SOIR_weighted', 'OFI1', 'OFI2', 'OFI3', 'OFI4', 'OFI5', 'MOFI',
        'vol_std_120', 'vol_std_600', 'vol_skew_120', 'vol_skew_600', 
        'tick_ret', 'vol_mid_corr_120', 'vol_mid_corr_600', 'rp_momentum_120',
        'rp_momentum_600', 'rp_momentum_20',
        'delta2_tick_rolling_1min', 'delta2_tick_rolling_5min', 'delta2_10s',
        'delta2_10s_rolling_1min', 'delta2_10s_rolling_5min', 'delta2_60s',
        'tick_smooth_price_change', 'elasticity_rolling_60s',
        'vol',  'oi_change', 'macd', 'rsi', 'kdj', 'ret_min',
        'pv_corr_10', 'pv_corr_120', 'rv_corr_120', 'rv_corr_10',
        'realized_vol_10', 'realized_vol_60', 'macd_long', 'rsi_short',
        'rsi_long', 'pvol_5min', 'pvol_30min', 'slope', 'price_pos',
        'upper_band', 'lower_band', 'channel_position', 'ret_5min',
        'volume_zscore', 'volume_cluster', 'spread_volatility',
        'volume_oi_ratio', 'oi_support', 'atr','upper_bb', 'lower_bb']

    ## 切分训练集和测试机，并且用训练集做标准化和数据清洗
    min_data_total = min_data_total.sort_values(by = 'time')
    print(min_data_total)
    def getBarriarLabel(datedata,time_barrier_minutes = 10):
        df = datedata.copy()
        df['upper_barrier'] = df['mid_price']*(1+0.002)
        df['lower_barrier'] = df['mid_price']*(1-0.002)
        df['label'] = 0
        # NaT (Not a Time) 代表尚未觸發
        df['touch_time'] = pd.NaT 
        for k in range(1, time_barrier_minutes + 1):
            future_prices = df['mid_price'].shift(-k)
            upper_mask = (future_prices >= df['upper_barrier']) & (df['touch_time'].isna())
            df.loc[upper_mask, 'label'] = 1
            df.loc[upper_mask, 'touch_time'] = df['time'] + pd.to_timedelta(k, unit='m')

            # --- 檢查下跌屏障（止損）---
            # 條件: 未來價格觸及屏障 AND 之前從未被觸發過
            lower_mask = (future_prices <= df['lower_barrier']) & (df['touch_time'].isna())
            df.loc[lower_mask, 'label'] = -1
            df.loc[lower_mask, 'touch_time'] = df['time'] + pd.to_timedelta(k, unit='m')

        return df['label'].tolist()
    labellist = []
    for date,datedata in  min_data_total.groupby('date'):
        labellist+=getBarriarLabel(datedata)
    min_data_total['label'] = labellist
    print(min_data_total.columns[20:])
    train_data = min_data_total[pd.to_datetime(min_data_total['date'])<pd.to_datetime("20240101")]
    test_data = min_data_total[pd.to_datetime(min_data_total['date'])>=pd.to_datetime("20240101")]
    train_mean = train_data[test_factors].mean()
    train_std = train_data[test_factors].std()
    train_data[test_factors] = (train_data[test_factors] - train_mean) / train_std
    train_data[test_factors] = (
        train_data[test_factors]
        .clip(lower=-3, upper=3)  # 将值限制在[-3, 3]范围内
        .replace([np.inf, -np.inf], 0)  # 将无穷大替换为0
        .fillna(0)  # 将NaN替换为0
    )
    test_data[test_factors] = (test_data[test_factors] - train_mean) / train_std
    test_data[test_factors] = (
        test_data[test_factors]
        .clip(lower=-3, upper=3)  # 将值限制在[-3, 3]范围内
        .replace([np.inf, -np.inf], 0)  # 将无穷大替换为0
        .fillna(0)  # 将NaN替换为0
    )
    train_data.to_pickle("train_v4.pkl")
    test_data.to_pickle("test_v4.pkl")
    # ## 计算交易日的滚动特征，并对交易行情分类
    # date_performance = pd.DataFrame(columns = ['daily_ret','daily_vol'])
    # for date in dates:
    #     df_read = pd.read_feather('D:\JT_Summer\\au\\'+date.strftime("%Y%m%d")+'\\'+calendar.loc[date]['contract'].lower()+'.feather')
    #     df_read['mid_price'] = (df_read['askp1']+df_read['bidp1'])/2
    #     daily_ret = (df_read['mid_price'].iloc[-1] - df_read['open'].iloc[0])/df_read['open'].iloc[0]
    #     df_read['ret_120'] = df_read['mid_price'].shift(-120)/df_read['mid_price']
    #     daily_vol = df_read['ret_120'].iloc[::120].std()
    #     date_performance.loc[date] = [daily_ret,daily_vol]
    
    # date_performance['daily_ret'] = date_performance['daily_ret'].abs()
    # date_performance_rolling = date_performance.rolling(5).mean().shift(1)
    # date_performance_rolling = date_performance_rolling.iloc[1:]
    # train_date_performance = date_performance[date_performance.index<pd.to_datetime("20230101")]
    # ret_threshold = train_date_performance['daily_ret'].quantile(0.7)
    # vol_threshold = train_date_performance['daily_vol'].quantile(0.7)
    
    # # date_performance_rolling.loc[(date_performance_rolling['daily_ret']>=0.0057)&(date_performance_rolling['daily_vol']>=0.0004),'type'] = 0
    # # date_performance_rolling.loc[(date_performance_rolling['daily_ret']>=0.0057)&(date_performance_rolling['daily_vol']<0.0004),'type'] = 1
    # # date_performance_rolling.loc[(date_performance_rolling['daily_ret']<0.0057)&(date_performance_rolling['daily_vol']>=0.0004),'type'] = 2
    # # date_performance_rolling.loc[(date_performance_rolling['daily_ret']<0.0057)&(date_performance_rolling['daily_vol']<0.0004),'type'] = 3
    # date_performance_rolling.loc[(date_performance_rolling['daily_ret']>=ret_threshold)&(date_performance_rolling['daily_vol']>=vol_threshold),'type'] = 0
    # date_performance_rolling.loc[(date_performance_rolling['daily_ret']>=ret_threshold)&(date_performance_rolling['daily_vol']<vol_threshold),'type'] = 1
    # date_performance_rolling.loc[(date_performance_rolling['daily_ret']<ret_threshold)&(date_performance_rolling['daily_vol']>=vol_threshold),'type'] = 2
    # date_performance_rolling.loc[(date_performance_rolling['daily_ret']<ret_threshold)&(date_performance_rolling['daily_vol']<vol_threshold),'type'] = 3
    # ## 分类提取数据
    # test_rollings = date_performance_rolling[date_performance_rolling.index>=pd.to_datetime("20230101")]
    # print(test_rollings)
    # c1_test_dates = test_rollings[test_rollings['type']==0].index
    # c2_test_dates = test_rollings[test_rollings['type']==1].index
    # c3_test_dates = test_rollings[test_rollings['type']==2].index
    # c4_test_dates = test_rollings[test_rollings['type']==3].index
    # train_date_performance.loc[(train_date_performance['daily_ret']>=ret_threshold)&(train_date_performance['daily_vol']>=vol_threshold),'type'] = 0
    # train_date_performance.loc[(train_date_performance['daily_ret']>=ret_threshold)&(train_date_performance['daily_vol']<vol_threshold),'type'] = 1
    # train_date_performance.loc[(train_date_performance['daily_ret']<ret_threshold)&(train_date_performance['daily_vol']>=vol_threshold),'type'] = 2
    # train_date_performance.loc[(train_date_performance['daily_ret']<ret_threshold)&(train_date_performance['daily_vol']<vol_threshold),'type'] = 3
    # c1_train_dates = train_date_performance[train_date_performance['type']==0].index
    # c2_train_dates = train_date_performance[train_date_performance['type']==1].index
    # c3_train_dates = train_date_performance[train_date_performance['type']==2].index
    # c4_train_dates = train_date_performance[train_date_performance['type']==3].index
    # # c1_dates = date_performance_rolling[date_performance_rolling['type']==0].index
    # # c1_train_dates = c1_dates[c1_dates<pd.to_datetime("20230101")]
    # # c1_test_dates = c1_dates[c1_dates>=pd.to_datetime("20230101")]
    # # train_data['date'] = pd.to_datetime(train_data['date'])
    # c1_traindata = train_data[train_data['date'].isin(c1_train_dates)]
    # # test_data['date'] = pd.to_datetime(test_data['date'])
    # c1_testdata = test_data[test_data['date'].isin(c1_test_dates)]
    # # c4_dates = date_performance_rolling[date_performance_rolling['type']==3].index
    # # c4_train_dates = c4_dates[c4_dates<pd.to_datetime("20230101")]
    # # c4_test_dates = c4_dates[c4_dates>=pd.to_datetime("20230101")]
    # c4_traindata = train_data[train_data['date'].isin(c4_train_dates)]
    # c4_testdata = test_data[test_data['date'].isin(c4_test_dates)]
    # # c2_dates = date_performance_rolling[date_performance_rolling['type']==1].index
    # # c2_train_dates = c2_dates[c2_dates<pd.to_datetime("20230101")]
    # # c2_test_dates = c2_dates[c2_dates>=pd.to_datetime("20230101")]
    # c2_traindata = train_data[train_data['date'].isin(c2_train_dates)]
    # c2_testdata = test_data[test_data['date'].isin(c2_test_dates)]
    # # c3_dates = date_performance_rolling[date_performance_rolling['type']==2].index
    # # c3_train_dates = c3_dates[c3_dates<pd.to_datetime("20230101")]
    # # c3_test_dates = c3_dates[c3_dates>=pd.to_datetime("20230101")]
    # c3_traindata = train_data[train_data['date'].isin(c3_train_dates)]
    # c3_testdata = test_data[test_data['date'].isin(c3_test_dates)]
    # min_data_total.to_pickle("min_data_total.pkl")
    # # print(min_data_t['askp1_trade'].isna().sum(),min_data['bidp1_trade'].isna().sum())
    # c1_traindata.to_pickle("C1_train_v2.pkl")
    # c1_testdata.to_pickle("C1_test_v2.pkl")
    # c2_traindata.to_pickle("C2_train_v2.pkl")
    # c2_testdata.to_pickle("C2_test_v2.pkl")
    # c3_traindata.to_pickle("C3_train_v2.pkl")
    # c3_testdata.to_pickle("C3_test_v2.pkl")
    # c4_traindata.to_pickle("C4_train_v2.pkl")
    # c4_testdata.to_pickle("C4_test_v2.pkl")
