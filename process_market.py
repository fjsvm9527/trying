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
from scipy.stats import norm

def calc_active_flow_matrix(df, price_type='mid', algo_type='lee_ready', 
                            multiplier=15.0, bvc_window=60):
    """
    通用主动买卖流计算矩阵 (3x3 = 9种组合)
    
    参数:
    df: 包含 volume, turnover, ap1, bp1, mid_price (或 last_price) 的 DataFrame
    price_type: 基准价格选择
        - 'mid': 中间价 ((ap1+bp1)/2)
        - 'last': 最新成交价
        - 'vwap': 区间成交均价 (Interval VWAP)
    algo_type: 分配算法选择
        - 'lee_ready': 改良版 Lee-Ready 规则 (Tick/Quote Rule)
        - 'bvc': Bulk Volume Classification (基于波动率概率)
        - 'proportional': 价格位置比例分配 (Linear Interpolation)
    multiplier: 合约乘数 (仅当 price_type='vwap' 时需要)
    bvc_window: BVC算法的滚动窗口 (仅当 algo_type='bvc' 时需要)
    """
    
    # ================= 1. 基础数据预处理 =================
    # 计算增量
    delta_vol = df['volume'].diff().fillna(0).clip(lower=0)
    
    # ================= 2. 确定基准价格 (Benchmark Price) =================
    # 目标：生成一个序列 P_t，代表当前时刻的价格水平
    
    if price_type == 'mid':
        # 使用当前时刻的 Mid Price
        benchmark_price = (df['ap1'] + df['bp1']) / 2
        # 修正 0 值
        benchmark_price = benchmark_price.replace(0, np.nan).fillna(method='ffill')
        
    elif price_type == 'last':
        # 如果没有 last_price，退化为 mid
        if 'last' in df.columns:
            benchmark_price = df['last']
        else:
            benchmark_price = df['mid_price'] # 假设外部已计算 mid_price
            
    elif price_type == 'vwap':
        # 计算区间成交均价 (Interval VWAP)
        delta_amt = df['turnover'].diff().fillna(0)
        # 防止除以0
        valid_vol = delta_vol.replace(0, np.nan)
        # ATP = d_Amt / d_Vol / Multiplier
        benchmark_price = (delta_amt / valid_vol) / multiplier
        # 填充 NaN (无成交时刻，价格沿用上一时刻 Mid 或 Last，保持连续性)
        # 这里为了后续计算方便，先用 mid 填充空缺
        temp_mid = (df['ap1'] + df['bp1']) / 2
        benchmark_price = benchmark_price.fillna(temp_mid)
        
    else:
        raise ValueError("price_type must be 'mid', 'last', or 'vwap'")

    # ================= 3. 确定分配比例 (Buy Ratio) =================
    # 目标：生成一个序列 ratio \in [0, 1]，代表 delta_vol 中有多少是买入
    
    # 准备上一时刻的参考数据 (t-1)
    prev_ask = df['ap1'].shift(1)
    prev_bid = df['bp1'].shift(1)
    prev_price = benchmark_price.shift(1) # 用于 Tick Rule 或 BVC diff
    
    # 初始化 ratio
    buy_ratio = pd.Series(0.5, index=df.index)
    
    # --- 算法 A: Lee-Ready (离散分类) ---
    if algo_type == 'lee_ready':
        # Quote Rule
        mask_buy = benchmark_price >= prev_ask
        mask_sell = benchmark_price <= prev_bid
        
        # Tick Rule (使用基准价格的变动方向)
        mask_up = (benchmark_price > prev_price) & (~mask_buy)
        mask_down = (benchmark_price < prev_price) & (~mask_sell)
        
        buy_ratio[mask_buy] = 1.0
        buy_ratio[mask_sell] = 0.0
        buy_ratio[mask_up] = 1.0
        buy_ratio[mask_down] = 0.0
        
    # --- 算法 B: BVC (概率连续分类) ---
    elif algo_type == 'bvc':
        # 计算基准价格的变化
        delta_p = benchmark_price.diff().fillna(0)
        
        # 计算滚动标准差 (Sigma)
        sigma_p = delta_p.rolling(window=bvc_window).std().fillna(0).replace(0, 1.0)
        
        # Z-score
        z_score = delta_p / sigma_p
        
        # CDF 映射到概率
        buy_ratio = norm.cdf(z_score)
        
        # 转换为 Series 以匹配索引
        buy_ratio = pd.Series(buy_ratio, index=df.index)
        
    # --- 算法 C: Proportional (位置比例分配) ---
    elif algo_type == 'proportional':
        # 计算 spread
        spread = prev_ask - prev_bid
        spread = spread.replace(0, np.nan) # 防止除零
        
        # 计算相对位置: (P_t - Bid_{t-1}) / (Ask_{t-1} - Bid_{t-1})
        # 结果 > 1 代表击穿卖一， < 0 代表击穿买一
        lambda_pos = (benchmark_price - prev_bid) / spread
        
        # 填充 NaN (无 Spread 时设为 0.5)
        lambda_pos = lambda_pos.fillna(0.5)
        
        # 截断到 [0, 1]
        buy_ratio = lambda_pos.clip(0, 1)
        
    else:
        raise ValueError("algo_type must be 'lee_ready', 'bvc', or 'proportional'")

    df[f'active_buy_vol'] = delta_vol * buy_ratio
    df[f'active_sell_vol'] = delta_vol * (1 - buy_ratio)
    
    return df

def agg_ewma(df, col_name='liquidity_skewness', span=120):
    """
    1. 时间衰减聚合 (EWMA)
    """
    result_col = f'{col_name}_ewma_{span}'
    print(f"Calculating EWMA (span={span})...")
    
    # adjust=False: 采用递归公式 y_t = alpha * x_t + (1-alpha) * y_{t-1}
    # 这更符合"当前权重最高，过去指数衰减"的直觉
    return df[col_name].ewm(span=span, adjust=False).mean()

def get_weighted_price_robust(row, price_cols, vol_cols, target_vol):
    """
    计算吃掉 target_vol 量后的加权均价 (Robust Version)
    包含平方根阻尼模型
    """
    if target_vol <= 0 or np.isnan(target_vol):
        return row[price_cols[0]]

    remains = target_vol
    total_cost = 0
    filled_vol = 0
    
    for p_col, v_col in zip(price_cols, vol_cols):
        p = row[p_col]
        v = row[v_col]
        if np.isnan(p) or np.isnan(v): continue

        take = min(remains, v)
        total_cost += take * p
        remains -= take
        filled_vol += take
        if remains <= 0: break
    
    if remains > 0:
        p1 = row[price_cols[0]]
        p5 = row[price_cols[-1]]
        last_price = p5
        
        # 判断方向
        if row[price_cols[0]] < row[price_cols[-1]]: is_ask = True
        elif row[price_cols[0]] > row[price_cols[-1]]: is_ask = False
        else: is_ask = True 

        current_width = abs(p5 - p1)
        min_width = 1.0 
        if current_width < 1e-6: current_width = min_width

        total_depth_5 = max(filled_vol, 1.0)
        ratio = remains / total_depth_5
        sensitivity = 0.5 
        
        extra_slippage = current_width * np.sqrt(ratio) * sensitivity
        max_slippage = max(current_width * 5.0, 20.0) 
        extra_slippage = min(extra_slippage, max_slippage)
        
        avg_price_deviation = min_width + (extra_slippage / 2)
        
        if is_ask: penalty_price = last_price + avg_price_deviation
        else: penalty_price = last_price - avg_price_deviation
            
        total_cost += remains * penalty_price
        
    return total_cost / target_vol

def process_single_file_basic(df):
    """
    读取单日文件，计算基础指标
    新增：调用 Lee-Ready 算法并计算滚动买卖需求
    """
    try:
        df['mid_price'] = (df['ap1'] + df['bp1']) / 2
        df.loc[df['bp1'] == 0, 'mid_price'] = df.loc[df['bp1'] == 0, 'ap1']
        df.loc[df['ap1'] == 0, 'mid_price'] = df.loc[df['ap1'] == 0, 'bp1']
        df['vwap_300'] = (df['turnover'].diff(600) / (df['volume'].diff(600).replace(0,np.nan)*15)).shift(-600)
        df['label'] = (df['vwap_300'] - df['mid_price']).abs()
        df['label_ap'] = (df['vwap_300'] - df['ap1'])
        df['label_bp'] = df['bp1'] - df['vwap_300']
        df['label_dir'] = df['vwap_300'] - df['mid_price']
        # 3. 计算主动买卖流 (New Implementation)
        df = calc_active_flow_matrix(df, price_type='mid', algo_type='proportional')
        window_size = 60
        
        # 计算过去30s的累计主动买入量
        df['rolling_active_buy'] = df['active_buy_vol'].rolling(window=window_size).sum().fillna(0)
        # 计算过去30s的累计主动卖出量
        df['rolling_active_sell'] = df['active_sell_vol'].rolling(window=window_size).sum().fillna(0)
        
        # 生成 Orderflow 指标 (平均每周期流量)
        df['dynamic_buy_orderflow'] = (df['rolling_active_buy'] / window_size)*6
        df['dynamic_sell_orderflow'] = (df['rolling_active_sell'] / window_size)*6
        
        # 为了兼容之前的代码逻辑，保留一个总的 orderflow 用于跨日比例计算参考
        # 但实际计算 spread 时将使用分开的 buy/sell flow
        df['dynamic_orderflow_total'] = (df['dynamic_buy_orderflow'] + df['dynamic_sell_orderflow'])
        
        return df
        
    except Exception as e:
        print(f"基础处理出错: {e}")
        return None, 0, None
    
def calc_wbas_5_level(df):
    """
    计算五档加权价差 (Vectorized)
    df: 包含 ap1~ap5, av1~av5, bp1~bp5, bv1~bv5 的 DataFrame
    """
    # 1. 计算卖方 (Ask) 总名义金额 (Notional) 和 总量 (Volume)
    ask_notional = (df['ap1'] * df['av1'] + 
                    df['ap2'] * df['av2'] + 
                    df['ap3'] * df['av3'] + 
                    df['ap4'] * df['av4'] + 
                    df['ap5'] * df['av5'])
                    
    ask_volume = (df['av1'] + df['av2'] + df['av3'] + df['av4'] + df['av5'])
    
    # 计算卖方 VWAP (防止除零)
    ask_vwap = ask_notional / ask_volume.replace(0, np.nan)

    # 2. 计算买方 (Bid) 总名义金额 (Notional) 和 总量 (Volume)
    bid_notional = (df['bp1'] * df['bv1'] + 
                    df['bp2'] * df['bv2'] + 
                    df['bp3'] * df['bv3'] + 
                    df['bp4'] * df['bv4'] + 
                    df['bp5'] * df['bv5'])
                    
    bid_volume = (df['bv1'] + df['bv2'] + df['bv3'] + df['bv4'] + df['bv5'])
    
    # 计算买方 VWAP
    bid_vwap = bid_notional / bid_volume.replace(0, np.nan)

    # 3. 计算加权价差
    df['wbas'] = ask_vwap - bid_vwap
    
    return df

def calculate_features(df_date):
    """
    计算 tick 级的基础特征
    """
    # 防止除以0
    epsilon = 1e-9
    if True:
        df_date['delta_volume'] = df_date['volume'].diff()
        df_date['bigorder_volume'] = 0
        df_date.loc[df_date['delta_volume']>=10,'bigorder_volume'] = df_date['delta_volume']
        df_date.iloc[0, df_date.columns.get_loc('delta_volume')] = df_date.iloc[0]['volume']
        df_date['delta_turnover'] = df_date['turnover'].diff()
        df_date.iloc[0, df_date.columns.get_loc('delta_turnover')] = df_date.iloc[0]['turnover']
        df_date['mid_price'] = (df_date['bp1'] + df_date['ap1'])/2
        df_date.loc[df_date['bp1']==0,'mid_price'] = df_date.loc[df_date['bp1']==0,'ap1']
        df_date.loc[df_date['ap1']==0,'mid_price'] = df_date.loc[df_date['ap1']==0,'bp1']
        df_date[['ap1','bp1','ap2','bp2','ap3','bp3','ap4','bp4','ap5','bp5']] = df_date[['ap1','bp1','ap2','bp2','ap3','bp3','ap4','bp4','ap5','bp5']].replace(0,np.nan)
        price_diff = df_date['last'].diff()
        tick_direction = price_diff.ffill().apply(np.sign)
        conditions = [
            df_date['last'] > df_date['mid_price'], 
            df_date['last'] < df_date['mid_price'],
            tick_direction == 1,                    
            tick_direction == -1           
        ]
        choices = [1, -1, 1, -1]
        df_date['trade_direction'] = np.select(conditions, choices, default=0)
        df_date['buy_volume'] = np.where(df_date['trade_direction'] == 1, df_date['delta_volume'], 0)
        df_date['sell_volume'] = np.where(df_date['trade_direction'] == -1, df_date['delta_volume'], 0)
        df_date['net_inflow'] = df_date['buy_volume'] - df_date['sell_volume']
        exec_price_conditions = [
            df_date['trade_direction'] == 1,
            df_date['trade_direction'] == -1
        ]
        exec_price_choices = [
            df_date['ap1'],
            df_date['bp1']
        ]
        df_date['estimated_exec_price'] = np.select(exec_price_conditions, exec_price_choices, default=df_date['last'])
        df_date['spread'] = df_date['ap1'] - df_date['bp1']
        df_date['bid_depth'] = df_date['bv1']+df_date['bv2']+df_date['bv3']+df_date['bv4']+df_date['bv5']
        df_date['ask_depth'] = df_date['av1']+df_date['av2']+df_date['av3']+df_date['av4']+df_date['av5']
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
                return 1
            elif P_t < mid_price:
                return -1
            else:
                return 0
        def calculate_effective_spread(df):
            """
            计算有效价差因子 ES.
            """
            df['D_k'] = df.apply(lambda row: calculate_direction(row['last'], row['mid_price']), axis=1)
            df['ES'] = 2 * df['D_k'] * (df['last'] - df['mid_price']) / df['mid_price']
            return df['ES']
        df_date['effective_spread'] = calculate_effective_spread(df_date)
        def calculate_ofi(df, bid_price, ask_price, bid_volume, ask_volume):
            delta_bid_vol = np.where(df[bid_price] < df[bid_price].shift(1), -df[bid_volume].shift(1),
                                    np.where(df[bid_price] > df[bid_price].shift(1), df[bid_volume],
                                            df[bid_volume] - df[bid_volume].shift(1)))

            delta_ask_vol = np.where(df[ask_price] < df[ask_price].shift(1), df[ask_volume],
                                    np.where(df[ask_price] > df[ask_price].shift(1), -df[ask_volume].shift(1),
                                            df[ask_volume] - df[ask_volume].shift(1)))
            ofi = delta_bid_vol - delta_ask_vol
            return ofi
        df_date['market_pressure'] = (df_date['bp1']*df_date['bv1']-df_date['ap1']*df_date['av1'])/(df_date['bp1']*df_date['bv1']+df_date['ap1']*df_date['av1'])
        df_date['SOIR1'] = (df_date['bv1']-df_date['av1'])/(df_date['bv1']+df_date['av1'])
        df_date['SOIR2'] = (df_date['bv2']-df_date['av2'])/(df_date['bv2']+df_date['av2'])
        df_date['SOIR3'] = (df_date['bv3']-df_date['av3'])/(df_date['bv3']+df_date['av3'])
        df_date['SOIR4'] = (df_date['bv4']-df_date['av4'])/(df_date['bv4']+df_date['av4'])
        df_date['SOIR5'] = (df_date['bv5']-df_date['av5'])/(df_date['bv5']+df_date['av5'])
        df_date['bid_ask_volume_ratio'] = (df_date['bv1']+df_date['bv2']+df_date['bv3']+df_date['bv4']+df_date['bv5'])/(df_date['av1']+df_date['av2']+df_date['av3']+df_date['av4']+df_date['av5'])
        df_date['relative_spread'] = (df_date['ap1'] - df_date['bp1'])/((df_date['ap1'] + df_date['bp1'])/2)
        df_date['SOIR'] = df_date['SOIR1']+df_date['SOIR2']+df_date['SOIR3']+df_date['SOIR4']+df_date['SOIR5']
        df_date['SOIR_weighted'] = 5*df_date['SOIR1']+4*df_date['SOIR2']+3*df_date['SOIR3']+2*df_date['SOIR4']+df_date['SOIR5']

        df_date['OFI1'] = calculate_ofi(df_date, 'bp1', 'ap1', 'bv1', 'av1')
        df_date['OFI2'] = calculate_ofi(df_date, 'bp2', 'ap2', 'bv2', 'av2')
        df_date['OFI3'] = calculate_ofi(df_date, 'bp3', 'ap3', 'bv3', 'av3')
        df_date['OFI4'] = calculate_ofi(df_date, 'bp4', 'ap4', 'bv4', 'av4')
        df_date['OFI5'] = calculate_ofi(df_date, 'bp5', 'ap5', 'bv5', 'av5')
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
        df_date['price_elascity'] = np.nan
        if True:
            df_date['high'] = df_date['mid_price'].rolling(120).max()
            df_date['low'] = df_date['mid_price'].rolling(120).min()
            df_date.loc[df_date['delta_turnover']>0,'price_elascity'] = (df_date['high']-df_date['low'])/df_date['delta_turnover']
            df_date['elascity_min'] = df_date['price_elascity'].rolling(120).mean()
            df_date['open'] = df_date['mid_price'].shift(119)
            df_date['ap1_trade'] = df_date['ap1'].shift(-1)
            df_date['bp1_trade'] = df_date['bp1'].shift(-1)
            df_date['vol'] = df_date['volume'] - df_date['volume'].shift(120)
            df_date['tick100_ret'] = (df_date['mid_price'].shift(-100)/df_date['mid_price'] - 1).fillna(0.0)
            df_date['bigorder_vol_1min'] = df_date['bigorder_volume'].rolling(120).sum()
            df_date['buyvol_1min'] = df_date['buy_volume'].rolling(120).sum()
            df_date['sellvol_1min'] = df_date['sell_volume'].rolling(120).sum()
            df_date['volweight_bigorder_ret'] = ((df_date['bigorder_volume']*df_date['tick100_ret']).rolling(1200,min_periods = 120).sum() / df_date['bigorder_volume'].rolling(1200,min_periods = 120).sum()).shift(100).fillna(0.0)
            df_date['is_bigorder'] = 0
            df_date.loc[df_date['bigorder_volume']>0,'is_bigorder'] = 1
            df_date['is_sellorder'] = 0
            df_date.loc[df_date['sell_volume']>0,'is_sellorder'] = 1
            df_date['is_buyorder'] = 0
            df_date.loc[df_date['buy_volume']>0,'is_buyorder'] = 1
            df_date['big_buy_volume'] = (df_date['is_bigorder'] & df_date['is_buyorder'])
            df_date['bigbuy_volume_min'] = df_date['big_buy_volume'].rolling(120).sum()
            df_date['bigorder_pct'] = df_date['is_bigorder'].rolling(1200).mean()
            df_date['big_sell_volume'] = (df_date['is_bigorder'] & df_date['is_sellorder'])
            df_date['bigsell_volume_min'] = df_date['big_sell_volume'].rolling(120).sum()
            df_date['effective_depth'] = df_date[['av1','bv1']].min(axis = 1)
            df_date['effective_depth_min'] = df_date['effective_depth'].rolling(120).mean()
            df_date['vwap'] = (df_date['delta_volume']*df_date['last']).rolling(120).sum()/df_date['delta_volume'].rolling(120).sum()
            df_date['mean_bigorder_ret'] = ((df_date['is_bigorder']*df_date['tick_ret']).rolling(1200,min_periods = 120).sum() / df_date['is_bigorder'].rolling(1200,min_periods = 120).sum()).shift(100).fillna(0.0)
            df_date['vol_entropy'] = np.nan
            df_date['AB_vol'] = df_date['delta_volume'].rolling(120).max() / df_date['delta_volume'].rolling(1200,min_periods = 120).mean()
            df_date['net_inflow_min'] = df_date['net_inflow'].rolling(120).sum()
            df_date['net_inflow_std_min'] = df_date['net_inflow'].rolling(120).std()
            df_date['net_inflow_skew_min'] = df_date['net_inflow'].rolling(120).skew()



from numba import njit
from numba import jit, float64,int64
import warnings
warnings.filterwarnings('ignore')
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
import numba
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
    df_date = df.copy()
    price = df_date['mid_price']
    # shift(1) 是 t-1 的價格, shift(-1) 是 t+1 的價格
    is_peak = (price.shift(1) < price) & (price > price.shift(-1))
    is_trough = (price.shift(1) > price) & (price < price.shift(-1))
    df_date['is_inflection'] = (is_peak | is_trough).astype(int)

    # --- 步驟 2: 計算過去30分鐘內拐點的總數 ---
    # 這是您問題的核心要求
    inflection_count_name = f'inflection_count_{window}m'
    df_date[inflection_count_name] = df_date['is_inflection'].rolling(window=window).sum()

    # --- 步驟 3: 根據圖片公式，計算最終的振盪器因子 ---
    # 因子 = 當前拐點數 - 過去5期拐點數的移動平均 (不含當期)
    sma_5_of_count = df_date[inflection_count_name].shift(1).rolling(window=5).mean()
    
    factor_name = f'inflection_oscillator_{window}m'
    return df_date[inflection_count_name] - sma_5_of_count
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
    df_date = df.copy()
    for col in ['open', 'high', 'low', 'mid_price']:
        df_date[col] = df_date[col].clip(lower=1e-9)
        
    log_open = np.log(df_date['open'])
    log_high = np.log(df_date['high'])
    log_low = np.log(df_date['low'])
    log_close = np.log(df_date['mid_price'])
    
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

def calculate_minute_factors(df_date):
        df_date['oi_change'] = (df_date['openinterest'] - df_date['openinterest'].shift(1))/df_date['openinterest']
        df_date['macd'] =numba_macd(df_date['mid_price'].values)
        df_date['rsi'] =numba_rsi(df_date['mid_price'].values)
        df_date['kdj'] = numba_kdj(df_date['high'].values,df_date['low'].values,df_date['mid_price'].values)
        df_date['ret_min'] = df_date['mid_price']/df_date['mid_price'].shift(1) - 1
        df_date['pv_corr_10'] = df_date['mid_price'].rolling(10).corr(df_date['vol'])
        df_date['rv_corr_10'] = df_date['ret_min'].rolling(10).corr(df_date['vol'])
        df_date['realized_vol_10'] = (df_date['ret_min']**2).rolling(10).sum()
        df_date['rs_vol_10'] = calculate_rolling_rs_volatility(df_date,window=10)
        df_date['rs_vol_30'] = calculate_rolling_rs_volatility(df_date,window=30)
        df_date['realized_vol_60'] = (df_date['ret_min']**2).rolling(60).sum()
        df_date['macd_long'] = numba_macd(df_date['mid_price'].values,24,52)
        df_date['rsi_short'] = numba_rsi(df_date['mid_price'].values,window=7)
        df_date['rsi_long'] = numba_rsi(df_date['mid_price'].values,window=28)
        df_date['pvol_5min'] = df_date['ret_min'].rolling(5).std()
        df_date['pvol_30min'] = df_date['ret_min'].rolling(30).std()
        df_date['Ivol_10'] = df_date['ret_min'].rolling(
            window=10, 
            min_periods=10
        ).apply(_calculate_ivol_window_robust, raw=True)
        df_date['Ivol_30'] = df_date['ret_min'].rolling(
            window=30, 
            min_periods=30
        ).apply(_calculate_ivol_window_robust, raw=True)
        df_date['slope'] = (df_date['ap1']-df_date['bp1'])/(df_date['av1']+df_date['bv1'])
        df_date['price_pos'] = (df_date['mid_price'] - df_date['low']) / (df_date['high'] - df_date['low']).replace(0, 1e-5)
        df_date['upper_band'] = df_date['high'].rolling(20).max()
        df_date['lower_band'] = df_date['low'].rolling(20).min()
        df_date['channel_position'] = (df_date['mid_price'] - df_date['lower_band']) / (df_date['upper_band'] - df_date['lower_band']).replace(0, 1e-5)
        df_date['ret_5min'] = df_date['mid_price'] / df_date['mid_price'].shift(5) - 1
        df_date['volume_zscore'] = (df_date['vol'] - df_date['vol'].rolling(30).mean()) / df_date['vol'].rolling(30).std()
        df_date['volume_cluster'] = (df_date['vol'] > df_date['vol'].rolling(30).mean() * 1.5).astype(int)
        df_date['spread_volatility'] = df_date['relative_spread'].rolling(10).std()
        df_date['volume_oi_ratio'] = df_date['vol'] / df_date['openinterest'].replace(0, 1e-5)
        df_date['oi_support'] = (df_date['openinterest'] > df_date['openinterest'].rolling(20).mean()).astype(int)
        df_date['dollar_volume'] = df_date['mid_price'] * df_date['vol']
        df_date['dollar_volume'] = df_date['dollar_volume'].replace(0, np.nan)
        df_date['effective_depth_5min'] = df_date['effective_depth_min'].rolling(5,min_periods = 1).mean()
        df_date['effective_depth_10min'] = df_date['effective_depth_min'].rolling(10,min_periods = 1).mean()
        df_date['effective_depth_30min'] = df_date['effective_depth_min'].rolling(30,min_periods = 1).mean()
        df_date['bigorder_vol_pct_1min'] = df_date['bigorder_vol_1min'] / df_date['vol']
        df_date['bigorder_vol_pct_5min'] = df_date['bigorder_vol_1min'].rolling(5,min_periods=1).sum() / df_date['vol'].rolling(5,min_periods=1).sum()
        df_date['bigorder_vol_pct_10min'] = df_date['bigorder_vol_1min'].rolling(10,min_periods=1).sum() / df_date['vol'].rolling(10,min_periods=1).sum()
        df_date['bigorder_vol_pct_30min'] = df_date['bigorder_vol_1min'].rolling(30,min_periods=1).sum() / df_date['vol'].rolling(30,min_periods=1).sum()
        df_date['ret_kurt_10'] = df_date['ret_min'].rolling(10).kurt()
        df_date['ret_kurt_30'] = df_date['ret_min'].rolling(30,min_periods = 10).kurt()
        df_date['ret_kurt_120'] = df_date['ret_min'].rolling(120,min_periods = 10).kurt()
        df_date['pvol_oi_ratio'] = (df_date['vwap']*df_date['vol'])/((df_date['openinterest'].map(lambda x: np.log(x) if x > 0 else np.nan))*df_date['rs_vol_30']*100)
        df_date['buyvolpct_1min'] = df_date['buyvol_1min'] / df_date['vol']
        df_date['buyvolpct_5min'] = df_date['buyvol_1min'].rolling(5,min_periods=1).sum() / df_date['vol'].rolling(5,min_periods=1).sum()
        df_date['buyvolpct_10min'] = df_date['buyvol_1min'].rolling(10,min_periods=1).sum() / df_date['vol'].rolling(10,min_periods=1).sum()
        df_date['buyvolpct_30min'] = df_date['buyvol_1min'].rolling(30,min_periods=1).sum() / df_date['vol'].rolling(30,min_periods=1).sum()
        df_date['sellvolpct_1min'] = df_date['sellvol_1min'] / df_date['vol']
        df_date['sellvolpct_5min'] = df_date['sellvol_1min'].rolling(5,min_periods=1).sum() / df_date['vol'].rolling(5,min_periods=1).sum()
        df_date['sellvolpct_10min'] = df_date['sellvol_1min'].rolling(10,min_periods=1).sum() / df_date['vol'].rolling(10,min_periods=1).sum()
        df_date['sellvolpct_30min'] = df_date['sellvol_1min'].rolling(30,min_periods=1).sum() / df_date['vol'].rolling(30,min_periods=1).sum()
        df_date['bigbuyvolpct_1min'] = df_date['bigbuy_volume_min'] / df_date['vol']
        df_date['bigbuyvolpct_5min'] = df_date['bigbuy_volume_min'].rolling(5,min_periods=1).sum() / df_date['vol'].rolling(5,min_periods=1).sum()
        df_date['bigbuyvolpct_10min'] = df_date['bigbuy_volume_min'].rolling(10,min_periods=1).sum() / df_date['vol'].rolling(10,min_periods=1).sum()
        df_date['bigbuyvolpct_30min'] = df_date['bigbuy_volume_min'].rolling(30,min_periods=1).sum() / df_date['vol'].rolling(30,min_periods=1).sum()
        df_date['bigsellvolpct_1min'] = df_date['bigsell_volume_min'] / df_date['vol']
        df_date['bigsellvolpct_5min'] = df_date['bigsell_volume_min'].rolling(5,min_periods=1).sum() / df_date['vol'].rolling(5,min_periods=1).sum()
        df_date['bigsellvolpct_10min'] = df_date['bigsell_volume_min'].rolling(10,min_periods=1).sum() / df_date['vol'].rolling(10,min_periods=1).sum()
        df_date['bigsellvolpct_30min'] = df_date['bigsell_volume_min'].rolling(30,min_periods=1).sum() / df_date['vol'].rolling(30,min_periods=1).sum()
        df_date['weight_buysell_10'] = (df_date['buyvol_1min']/(df_date['sellvol_1min']+1e-9)).ewm(span=10, min_periods=5).std()
        df_date['weight_buysell_30'] = (df_date['buyvol_1min']/(df_date['sellvol_1min']+1e-9)).ewm(span=30, min_periods=10).std()
        df_date['CV_10'] = df_date['ret_min'].rolling(10,min_periods=5).var() / np.abs(df_date['ret_min'].rolling(10,min_periods=5).mean())
        df_date['CV_30'] = df_date['ret_min'].rolling(30,min_periods=5).var() / np.abs(df_date['ret_min'].rolling(30,min_periods=5).mean())
        df_date['cvar_10'] = df_date['ret_min'].rolling(window=10).apply(_calculate_cvar_window, raw=True, args=(0.05,))
        df_date['cvar_30'] = df_date['ret_min'].rolling(window=30).apply(_calculate_cvar_window, raw=True, args=(0.05,))
        df_date['illiq_ratio_minute'] = df_date['ret_min'].abs() / df_date['dollar_volume'] * 1e6
        df_date[f'amihud_ratio_30'] = df_date['illiq_ratio_minute'].rolling(window=30,min_periods=10).mean()
        df_date[f'amihud_ratio_10'] = df_date['illiq_ratio_minute'].rolling(window=10).mean()
        df_date[f'amihud_ratio_60'] = df_date['illiq_ratio_minute'].rolling(window=60,min_periods=10).mean()
        abs_return = df_date['ret_min'].abs()
        abs_return.replace(0, np.nan, inplace=True)
        df_date['lr_minute'] = df_date['dollar_volume'] / abs_return
        factor_name = f'amivest_lr_10'
        df_date[factor_name] = df_date['lr_minute'].rolling(window=10,min_periods=5).mean()
        df_date['amivest_lr_30'] = df_date['lr_minute'].rolling(window=30,min_periods=5).mean()
        df_date['typical_price'] = (df_date['high'] + df_date['low'] + df_date['mid_price']) / 3
        df_date['hlc_ma_10'] = df_date['typical_price'].rolling(window=10, min_periods=1).mean()
        avedev_func = lambda x: np.mean(np.abs(x - np.mean(x)))
        df_date['avedev_10'] = df_date['typical_price'].rolling(window=10, min_periods=1).apply(avedev_func, raw=True)
        open_prev = df_date['open'].shift(1)
        dtm_cond = df_date['open'] > open_prev
        dtm_val = np.maximum(df_date['high'] - df_date['open'], df_date['open'] - open_prev)
        df_date['dtm'] = np.where(dtm_cond, dtm_val, 0)
        dbm_cond = df_date['open'] < open_prev
        dbm_val = np.maximum(df_date['open'] - df_date['low'], open_prev - df_date['open'])
        df_date['dbm'] = np.where(dbm_cond, dbm_val, 0)
        stm = df_date['dtm'].rolling(window=10).sum()
        sbm = df_date['dbm'].rolling(window=10).sum()
        max_sm = np.maximum(stm, sbm)
        adtm_factor = (stm - sbm) / max_sm.replace(0, np.nan)
        adtm_factor.fillna(0, inplace=True) # 當總動能為0時，可視為中性
        factor_name = f'adtm_{10}m'
        df_date[factor_name] = adtm_factor
        stm = df_date['dtm'].rolling(window=30).sum()
        sbm = df_date['dbm'].rolling(window=30).sum()
        max_sm = np.maximum(stm, sbm)
        adtm_factor = (stm - sbm) / max_sm.replace(0, np.nan)
        adtm_factor.fillna(0, inplace=True) # 當總動能為0時，可視為中性
        factor_name = f'adtm_{30}m'
        df_date[factor_name] = adtm_factor
        df_date[f'skew_overall_{10}m'] = df_date['ret_min'].rolling(window=10).skew()
        df_date['TrendStrenth_10'] = (df_date['mid_price']-df_date['mid_price'].shift(10))/((df_date['mid_price'] - df_date['mid_price'].shift(1))).abs().rolling(9).sum()
        df_date['TrendStrenth_30'] = (df_date['mid_price']-df_date['mid_price'].shift(30))/((df_date['mid_price'] - df_date['mid_price'].shift(1))).abs().rolling(29).sum()
        df_date['cls_abs_var_30'] = df_date['mid_price'].rolling(30,min_periods = 10).apply(lambda x : np.abs(x - x.mean()).mean())
        df_date['cls_abs_var_10'] = df_date['mid_price'].rolling(10,min_periods = 10).apply(lambda x : np.abs(x - x.mean()).mean())
        df_date['money_flow'] = df_date['typical_price'] * df_date['vol']
        price_direction = df_date['typical_price'].diff(1)
        positive_flow = np.where(price_direction > 0, df_date['money_flow'], 0)
        negative_flow = np.where(price_direction <= 0, df_date['money_flow'], 0)
        positive_flow_series = pd.Series(positive_flow, index=df_date.index)
        negative_flow_series = pd.Series(negative_flow, index=df_date.index)
        rolling_pf = positive_flow_series.rolling(window=10).sum()
        rolling_nf = negative_flow_series.rolling(window=10).sum()
        money_ratio = rolling_pf / rolling_nf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_ratio))
        mfi.fillna(100, inplace=True)
        mfi[ (rolling_pf == 0) & (rolling_nf == 0) ] = 50
        df_date[f'mfi_{10}m'] = mfi
        rolling_pf = positive_flow_series.rolling(window=30).sum()
        rolling_nf = negative_flow_series.rolling(window=30).sum()
        money_ratio = rolling_pf / rolling_nf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_ratio))
        mfi.fillna(100, inplace=True)
        mfi[ (rolling_pf == 0) & (rolling_nf == 0) ] = 50
        df_date[f'mfi_{30}m'] = mfi
        df_date['pelas_5min'] = df_date['elascity_min'].rolling(5,min_periods = 1).mean()
        df_date['pelas_10min'] = df_date['elascity_min'].rolling(10,min_periods = 1).mean()
        df_date['pelas_30min'] = df_date['elascity_min'].rolling(30,min_periods = 1).mean()
        denominator = 0.015 * df_date['avedev_10']
        df_date[f'cci_10'] = np.where(
            denominator > 1e-9,  # 避免除以一個非常小的數
            (df_date['typical_price'] - df_date['hlc_ma_10']) / denominator,
            0.0
        )
        df_date['hlc_ma_30'] = df_date['typical_price'].rolling(window=30, min_periods=1).mean()
        df_date['avedev_30'] = df_date['typical_price'].rolling(window=30, min_periods=1).apply(avedev_func, raw=True)
        denominator = 0.015 * df_date['avedev_30']
        df_date[f'cci_30'] = np.where(
            denominator > 1e-9,  # 避免除以一個非常小的數
            (df_date['typical_price'] - df_date['hlc_ma_30']) / denominator,
            0.0
        )
        factor_series = df_date['last'].rolling(
            window=10, 
            min_periods=10
        ).apply(_calculate_regression_factor_window_robust, raw=True)
        
        df_date[f'regression_factor_10'] = factor_series
        factor_series = df_date['last'].rolling(
            window=30, 
            min_periods=30
        ).apply(_calculate_regression_factor_window_robust, raw=True)
        volatility_series = df_date['ret_min'].rolling(window=10).std()
        fuzziness_series = volatility_series.rolling(window=10).std()
        df_date['fuzziness'] = fuzziness_series
        df_date['fuzzy_corr'] = fuzziness_series.rolling(
            window=30
        ).corr(
            df_date['dollar_volume']
        )
        df_date['oi_vol_corr_10'] = df_date['openinterest'].rolling(window = 10).corr(df_date['vol'])
        df_date['oi_vol_corr_30'] = df_date['openinterest'].rolling(window = 30).corr(df_date['vol'])
        df_date['oi_ret_corr_10'] = df_date['openinterest'].rolling(window = 10).corr(df_date['ret_min'])
        df_date['oi_ret_corr_30'] = df_date['openinterest'].rolling(window = 30).corr(df_date['ret_min'])
        price_change_sign = np.sign(df_date['ret_min'])
        signed_dollar_volume = df_date['dollar_volume'] * price_change_sign
        rolling_net_flow = signed_dollar_volume.rolling(window=10).sum()
        rolling_total_volume = df_date['dollar_volume'].rolling(window=10).sum()
        df_date[f'flow_in_ratio_{10}m'] = (rolling_net_flow / rolling_total_volume.replace(0, np.nan)).fillna(0, inplace=True) # 當總成交額為0時，可視為中性
        rolling_net_flow = signed_dollar_volume.rolling(window=30,min_periods = 10).sum()
        rolling_total_volume = df_date['dollar_volume'].rolling(window=30,min_periods = 10).sum()
        df_date[f'flow_in_ratio_{30}m'] = (rolling_net_flow / rolling_total_volume.replace(0, np.nan)).fillna(0, inplace=True) # 當總成交額為0時，可視為中性
        rolling_std = df_date['ret_min'].rolling(window=10).std()
        rolling_mean = df_date['ret_min'].rolling(window=10).mean()
        df_date['volret_ratio_10'] = rolling_std / rolling_mean.replace(0, np.nan)
        rolling_std = df_date['ret_min'].rolling(window=30).std()
        rolling_mean = df_date['ret_min'].rolling(window=30).mean()
        df_date['volret_ratio_30'] = rolling_std / rolling_mean.replace(0, np.nan)
        up_volume = np.where(df_date['ret_min'] > 0, df_date['vol'], 0)
        down_volume = np.where(df_date['ret_min'] <= 0, df_date['vol'], 0)
        
        # 將輔助序列轉為 pandas Series 以便使用 rolling 方法
        up_volume_series = pd.Series(up_volume, index=df_date.index)
        down_volume_series = pd.Series(down_volume, index=df_date.index)
        df_date['vr_10'] = up_volume_series.rolling(window=10).sum() / down_volume_series.rolling(window=10).sum().replace(0, np.nan)
        df_date['vr_30'] = up_volume_series.rolling(window=30).sum() / down_volume_series.rolling(window=30).sum().replace(0, np.nan)
        df_date['netflow_5min'] = df_date['net_inflow'].rolling(5).mean()
        df_date['netflow_10min'] = df_date['net_inflow'].rolling(10).mean()
        df_date['netflow_30min'] = df_date['net_inflow'].rolling(30).mean()
        df_date['netflow_10min_std'] = df_date['net_inflow'].rolling(10).std()
        df_date['netflow_30min_std'] = df_date['net_inflow'].rolling(30).std()
        gains = df_date['ret_min'].clip(lower=0)
        # CZ2: 下跌的幅度（取正值），上漲時為0
        losses = -df_date['ret_min'].clip(upper=0)
        sum_up = gains.rolling(window=10, min_periods=1).sum()
        sum_down = losses.rolling(window=10, min_periods=1).sum()
        total_momentum = sum_up + sum_down 
        cmo_factor = ((sum_up - sum_down) / total_momentum.replace(0, np.nan)) * 100
        cmo_factor.fillna(0, inplace=True) # 填充 NaN 值為 0
        factor_name = f'cmo_10'
        df_date[factor_name] = cmo_factor
        sum_up = gains.rolling(window=30, min_periods=1).sum()
        sum_down = losses.rolling(window=30, min_periods=1).sum()
        total_momentum = sum_up + sum_down 
        cmo_factor = ((sum_up - sum_down) / total_momentum.replace(0, np.nan)) * 100
        cmo_factor.fillna(0, inplace=True) # 填充 NaN 值為 0
        factor_name = f'cmo_30'
        df_date[factor_name] = cmo_factor
        df_date['sq_return'] = df_date['ret_min']**2
        df_date['up_sq_return'] = np.where(df_date['ret_min'] > 0, df_date['sq_return'], 0)
        factor_series = (df_date['up_sq_return'].rolling(window=10).sum() / df_date['sq_return'].rolling(window=10).sum().replace(0, np.nan)).fillna(0.5, inplace=True)
        df_date['up_vol_ratio_10'] = factor_series
        factor_series = (df_date['up_sq_return'].rolling(window=30).sum() / df_date['sq_return'].rolling(window=30).sum().replace(0, np.nan)).fillna(0.5, inplace=True)
        df_date['up_vol_ratio_30'] = factor_series
        df_date['up_variance_term'] = np.where(df_date['ret_min'] > 0, df_date['sq_return'], 0)
        df_date['down_variance_term'] = np.where(df_date['ret_min'] < 0, df_date['sq_return'], 0)
        rolling_up_variance = df_date['up_variance_term'].rolling(window=10).sum()
        rolling_down_variance = df_date['down_variance_term'].rolling(window=10).sum()
        factor_series = (rolling_up_variance - rolling_down_variance) * 1e4
        factor_name = f'variance_diff_{10}m'
        df_date[factor_name] = factor_series
        rolling_up_variance = df_date['up_variance_term'].rolling(window=30).sum()
        rolling_down_variance = df_date['down_variance_term'].rolling(window=30).sum()
        factor_series = (rolling_up_variance - rolling_down_variance) * 1e4
        factor_name = f'variance_diff_{30}m'
        df_date[factor_name] = factor_series
        roc_n1 = df_date['mid_price'].pct_change(periods=5) * 100
        roc_n2 = df_date['mid_price'].pct_change(periods=10) * 100
        rc_series = roc_n1 + roc_n2
        factor_name = f'coppock_{5}_{10}_{30}m'
        df_date[factor_name] = rc_series.rolling(window=30).mean()
        roc_n1 = df_date['mid_price'].pct_change(periods=10) * 100
        roc_n2 = df_date['mid_price'].pct_change(periods=15) * 100
        rc_series = roc_n1 + roc_n2
        factor_name = f'coppock_{10}_{15}_{30}m'
        df_date[factor_name] = rc_series.rolling(window=30).mean()
        reversal_indicator = (df_date['ret_min'].shift(1) > 0) & (df_date['ret_min'] < 0)
        df_date['is_reversal'] = reversal_indicator.astype(int)
        nr_series = df_date['is_reversal'].rolling(window=10).sum()
        df_date[f'NR_10'] = nr_series
        nr_sma = nr_series.rolling(window=5).mean()
        ab_nr_factor = nr_series / nr_sma.replace(0, np.nan)
        factor_name = f'ab_nr_{10}m'
        df_date[factor_name] = ab_nr_factor
        nr_series = df_date['is_reversal'].rolling(window=30).sum()
        df_date[f'NR_10'] = nr_series
        nr_sma = nr_series.rolling(window=12).mean()
        ab_nr_factor = nr_series / nr_sma.replace(0, np.nan)
        factor_name = f'ab_nr_{30}m'
        df_date[factor_name] = ab_nr_factor
        df_date['regr_r2_ret_10'] = fast_rolling_quadratic_r2(df_date['ret_min'].fillna(0.0).values,window=10)
        df_date['regr_r2_ret_30'] = fast_rolling_quadratic_r2(df_date['ret_min'].fillna(0.0).values,window=30)
        df_date['sq_return'] = df_date['ret_min']**2
        df_date['sq_up_return'] = np.where(df_date['ret_min'] > 0, df_date['sq_return'], 0)
        df_date['sq_down_return'] = np.where(df_date['ret_min'] < 0, df_date['sq_return'], 0)
        rolling_variance = df_date['sq_return'].rolling(window=10).sum()
        rolling_up_variance = df_date['sq_up_return'].rolling(window=10).sum()
        rolling_down_variance = df_date['sq_down_return'].rolling(window=10).sum()
        rv = np.sqrt(rolling_variance)
        rv_up = np.sqrt(rolling_up_variance)
        rv_down = np.sqrt(rolling_down_variance)
        rsj_factor = (rv_up - rv_down) / rv.replace(0, np.nan)
        df_date[f'rsj_10'] = rsj_factor
        rolling_variance = df_date['sq_return'].rolling(window=30).sum()
        rolling_up_variance = df_date['sq_up_return'].rolling(window=30).sum()
        rolling_down_variance = df_date['sq_down_return'].rolling(window=30).sum()
        rv = np.sqrt(rolling_variance)
        rv_up = np.sqrt(rolling_up_variance)
        rv_down = np.sqrt(rolling_down_variance)
        rsj_factor = (rv_up - rv_down) / rv.replace(0, np.nan)
        df_date[f'rsj_30'] = rsj_factor
        df_date['ar_30'] = (df_date['high'] - df_date['open']).rolling(30,min_periods=10).sum()/(df_date['open'] - df_date['low']).rolling(30,min_periods=10).sum()
        df_date['ar_10'] = (df_date['high'] - df_date['open']).rolling(10,min_periods=5).sum()/(df_date['open'] - df_date['low']).rolling(10,min_periods=5).sum()
        close_prev = df_date['mid_price'].shift(1)
        ewm_volatility = df_date['ret_min'].ewm(span=10, min_periods=5).std()
        df_date['ewm_vol_10'] = ewm_volatility
        ewm_volatility = df_date['ret_min'].ewm(span=30, min_periods=5).std()
        df_date['ewm_vol_30'] = ewm_volatility
        # 步驟 2: 計算每分鐘的上攻力量和下探力量
        up_power = (df_date['high'] - close_prev).clip(lower=0)
        down_power = (close_prev - df_date['low']).clip(lower=0)

        # 步驟 3: 計算分子和分母的滾動加總
        rolling_sum_up = up_power.rolling(window=10, min_periods=1).sum()
        rolling_sum_down = down_power.rolling(window=10, min_periods=1).sum()
        br_factor = (rolling_sum_up / rolling_sum_down.replace(0, np.nan)) * 100
        # 這裡我們用一個較大的值（如400）來填充這種極端情況
        br_factor.fillna(400, inplace=True) 
        df_date[f'br_10'] = br_factor
        rolling_sum_up = up_power.rolling(window=30, min_periods=1).sum()
        rolling_sum_down = down_power.rolling(window=30, min_periods=1).sum()
        br_factor = (rolling_sum_up / rolling_sum_down.replace(0, np.nan)) * 100
        # 這裡我們用一個較大的值（如400）來填充這種極端情況
        br_factor.fillna(400, inplace=True) 
        df_date[f'br_30'] = br_factor
        df_date['is_up'] = np.where(df_date['ret_min']>0,1,0)
        df_date['PSY_10'] = df_date['is_up'].rolling(10).mean()
        df_date['PSY_30'] = df_date['is_up'].rolling(30).mean()
        df_date['PSY_60'] = df_date['is_up'].rolling(60).mean()
        df_date['amplitude_30'] = fast_rolling_ideal_amplitude(df_date['high'].values,df_date['low'].values,df_date['mid_price'].values,window=30,quantile=0.8)
        df_date['amplitude_10'] = fast_rolling_ideal_amplitude(df_date['high'].values,df_date['low'].values,df_date['mid_price'].values,window=10,quantile=0.8)
        df_date['struct_rev_10'] = fast_rolling_structural_reversal(df_date['ret_min'].values,df_date['vol'].values,window=10,quantile_threshold=0.8)
        df_date['struct_rev_30'] = fast_rolling_structural_reversal(df_date['ret_min'].values,df_date['vol'].values,window=30,quantile_threshold=0.8)
        df_date['volume_threshold'] = df_date['vol'].rolling(
        window=240, min_periods=10).quantile(0.8)
        df_date['trend_10'] = fast_rolling_trend_fund_factor(df_date['mid_price'].values,df_date['vol'].values,vol_threshold_arr=df_date['volume_threshold'].values,window=10)
        df_date['trend_30'] = fast_rolling_trend_fund_factor(df_date['mid_price'].values,df_date['vol'].values,vol_threshold_arr=df_date['volume_threshold'].values,window=30)
        high_low = df_date['high'] - df_date['low']
        high_mid_price = np.abs(df_date['high'] - df_date['mid_price'].shift())
        low_mid_price = np.abs(df_date['low'] - df_date['mid_price'].shift())
        true_range = pd.concat([high_low, high_mid_price, low_mid_price], axis=1).max(axis=1)
        df_date['atr'] = true_range.rolling(14).mean() / df_date['mid_price']
        df_date['ma20'] = df_date['mid_price'].rolling(20).mean()
        df_date['upper_bb'] = (df_date['ma20'] + 2 * df_date['mid_price'].rolling(20).std() - df_date['mid_price'])/df_date['mid_price']
        df_date['lower_bb'] = (df_date['ma20'] - 2 * df_date['mid_price'].rolling(20).std() - df_date['mid_price']) / df_date['mid_price']
        df_date['turnover_5min'] = df_date['delta_turnover'].rolling(5).mean()
        df_date['turnover_10min'] = df_date['delta_turnover'].rolling(10).mean()
        return df_date

def calc_final_spread_rolling(df, is_main, 
                              multiplier=1.0, 
                              lambda_adj=0.3, 
                              rolling_window=60,  # 新增：滚动窗口大小
                              depth_mode='L1'):   # 新增：'L1' 或 'L1_5'
    """
    计算最终价差 (包含基于滚动窗口的补充流修正)
    
    参数:
    rolling_window: 计算补充速度的时间窗口 (例如 60 ticks)
    depth_mode: 'L1' (仅基于最优档) 或 'L1_5' (基于前5档总和)
    """
    
    # 1. 流量准备
    raw_buy_flow = df['dynamic_buy_orderflow'] * multiplier
    raw_sell_flow = df['dynamic_sell_orderflow'] * multiplier
    
    # 准备价格和量列名
    ask_prices = [f'ap{i}' for i in range(1, 6)]
    ask_vols = [f'av{i}' for i in range(1, 6)]
    bid_prices = [f'bp{i}' for i in range(1, 6)]
    bid_vols = [f'bv{i}' for i in range(1, 6)]
    
    # ================= 2. OFI 补充流计算 (Rolling Logic) =================
    
    act_buy_col = f'active_buy_vol'
    act_sell_col = f'active_sell_vol'
    
    # 初始化修正流
    df['effective_buy_flow'] = raw_buy_flow
    df['effective_sell_flow'] = raw_sell_flow
    
    if act_buy_col in df.columns:
        # --- 2.1 确定深度 (Depth) 和 价格基准 (Price Basis) ---
        if depth_mode == 'L1':
            # 模式 A: 仅使用 L1
            ask_depth = df['av1']
            bid_depth = df['bv1']
            # L1 模式下，严格使用 ap1/bp1 变化判断 OFI
            ask_p = df['ap1']
            bid_p = df['bp1']
            
        elif depth_mode == 'L1_5':
            # 模式 B: 使用 L1-L5 总和
            # 聚合 5 档挂单量
            ask_depth = df[ask_vols].sum(axis=1)
            bid_depth = df[bid_vols].sum(axis=1)
            # L1-L5 模式下，通常使用 Mid Price 或 ap1/bp1 作为整体移动的锚点
            # 这里依然使用 ap1/bp1，假设如果最优价变了，整个 Orderbook 都在移动
            ask_p = df['ap1']
            bid_p = df['bp1']
        else:
            raise ValueError("depth_mode must be 'L1' or 'L1_5'")
            
        # --- 2.2 计算瞬时 OFI (Cont's Method) ---
        # 这里的 delta_depth 代表：排除价格变动干扰后的"净挂单增量"
        
        # Ask Side
        ask_p_prev = ask_p.shift(1)
        ask_d_curr = ask_depth
        ask_d_prev = ask_depth.shift(1)
        
        delta_ask_depth = pd.Series(0.0, index=df.index)
        # 价格不变：直接看量的变化
        delta_ask_depth[ask_p == ask_p_prev] = ask_d_curr - ask_d_prev
        # 价格涨了 (Ask被攻破/撤退)：旧的深度视为全部损失
        delta_ask_depth[ask_p > ask_p_prev] = -ask_d_prev 
        # 价格跌了 (Ask压进/补充)：新的深度视为全部增量
        delta_ask_depth[ask_p < ask_p_prev] = ask_d_curr 
        
        # 瞬时补充 = 深度净变化 + 被吃掉的量 (Active Buy)
        # 注意：如果用 L1_5 模式，理论上应该加回 "L1-5 范围内所有的成交"，
        # 但我们只有 total active buy，这里近似认为 active buy 消耗的是整体深度。
        replenishment_ask_inst = delta_ask_depth + df[act_buy_col]

        # Bid Side
        bid_p_prev = bid_p.shift(1)
        bid_d_curr = bid_depth
        bid_d_prev = bid_depth.shift(1)
        
        delta_bid_depth = pd.Series(0.0, index=df.index)
        delta_bid_depth[bid_p == bid_p_prev] = bid_d_curr - bid_d_prev
        delta_bid_depth[bid_p < bid_p_prev] = -bid_d_prev # 跌了，Bid被攻破
        delta_bid_depth[bid_p > bid_p_prev] = bid_d_curr # 涨了，Bid顶上
        replenishment_bid_inst = delta_bid_depth + df[act_sell_col]
        rep_ask_rolling = replenishment_ask_inst.rolling(window=rolling_window, min_periods=1).mean()
        rep_bid_rolling = replenishment_bid_inst.rolling(window=rolling_window, min_periods=1).mean()
        vol_change_rolling = df['volume'].diff().abs().rolling(rolling_window, min_periods=1).mean()
        scale = vol_change_rolling.replace(0, 1) * 5 # *5 是一个经验放缩系数
        
        # 计算弹性系数
        resilience_ask = np.tanh(rep_ask_rolling / scale) * lambda_adj
        resilience_bid = np.tanh(rep_bid_rolling / scale) * lambda_adj
        
        # --- 2.5 应用修正 ---
        df['effective_buy_flow'] = raw_buy_flow * (1 - resilience_ask)
        df['effective_sell_flow'] = raw_sell_flow * (1 - resilience_bid)
        
        # 裁剪
        df['effective_buy_flow'] = df['effective_buy_flow'].clip(lower=0)
        df['effective_sell_flow'] = df['effective_sell_flow'].clip(lower=0)
        
    else:
        print("Warning: Active volume columns not found.")

    # ================= 3. 计算冲击成本 & 最终指标 =================
    
    # 计算 Impact
    df['dynamic_ask_price'] = df.apply(
        lambda x: get_weighted_price_robust(x, ask_prices, ask_vols, x['effective_buy_flow']), 
        axis=1
    )
    df['dynamic_bid_price'] = df.apply(
        lambda x: get_weighted_price_robust(x, bid_prices, bid_vols, x['effective_sell_flow']), 
        axis=1
    )
    
    col_spread = f'dynamic_spread_robust'
    df[col_spread] = df['dynamic_ask_price'] - df['dynamic_bid_price']
    # df[f'{col_spread}_mean'] = df[col_spread].rolling(240).mean()
    df[f'{col_spread}_mean'] = agg_ewma(df,col_spread,span = 240)
    col_skew = f'liquidity_skewness'
    df[col_skew] = (df['dynamic_ask_price'] - df['ap1']) - (df['bp1'] - df['dynamic_bid_price'])
    # df[f'liquidity_skew_mean{suffix}'] = df[col_skew].rolling(240).mean()
    df[f'liquidity_skew_mean'] = agg_ewma(df,col_skew,span = 240)
    df['spread'] = df['ap1'] - df['bp1']
    df['spread_mean'] = df['spread'].rolling(240).mean()
    df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1))
    df['delta_volume'] = df['volume'].diff().fillna(0.0)
    df['delta_turnover'] = df['turnover'].diff().fillna(0.0) / 15
    epsilon = 1e-8
    df['tick_illiq'] = df['log_ret'].abs() / (df['delta_turnover'] + epsilon)
    trade_ticks = df[df['volume'] > 0]
    df['rolling_amihud'] = trade_ticks['tick_illiq'].rolling(240).sum()
    df = calc_wbas_5_level(df)
    df['deep_spread_rolling'] = df['wbas'].rolling(240).mean()
    df[f'liquidity_skew_mean_600'] = agg_ewma(df,col_skew,span = 600)
    df[f'{col_spread}_mean_600'] = agg_ewma(df,col_spread,span = 600)
    return df
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
    market_data_dir = '/home/zyyuan/project1/try/market_data'
    market_files_map = {}
    
    # 假设 market 文件名也包含日期，用同样的逻辑提取
    market_candidates = glob.glob(os.path.join(market_data_dir, "*.csv"))
    print(f"扫描 Market Data 目录... 发现 {len(market_candidates)} 个文件")
    for mf in market_candidates:
        d_str = get_trade_day_from_filename(mf) # 假设这个函数通用，或者你需要写一个针对 market 文件名的解析函数
        if d_str:
            market_files_map[d_str] = mf
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

            market_file_path = market_files_map[trade_day_str]
            df_market = pd.read_csv(market_file_path)
            df_market['hms'] = pd.to_datetime(df_market['hms'])
            ms_delta = pd.to_timedelta(df_market['ms'], unit='ms')
            df_market['timestamp'] = df_market['hms'] + ms_delta
            df = pd.merge(
                    df, 
                    df_market, 
                    on='timestamp', 
                    how='left', 
                    suffixes=('', '_market')
                )
            df = process_single_file_basic(df)
            df = calc_final_spread_rolling(df, is_main=True, multiplier=1.0,lambda_adj=0.7,depth_mode='L1_5')
            calculate_features(df)
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

            # 添加均值 (所有特征)
            s_mean = feat_mean.iloc[indices].copy()
            s_mean.columns = [f"{c}_mean" for c in raw_feature_cols]
            concat_list.append(s_mean)
            

            meta_cols = [CONFIG['label_col'], 'timestamp', 'trade_day','triggerInst.volume','OBI_cum120','rolling_amp','LABEL_CAL_DQ_inst1_60','is_pm','is_am','is_night','gate_session_decay','gate_open_impulse','gate_rv','gate_downside_ratio','gate_rv_log','gate_range_log','liquidity_skew_mean','liquidity_skewness','dynamic_spread_robust','dynamic_spread_robust_mean','spread_mean','rolling_amihud','deep_spread_rolling','liquidity_skew_mean_600','dynamic_spread_robust_mean_600']
            s_meta = df[meta_cols].iloc[indices].copy()
            concat_list.append(s_meta)
            
            # 添加市场特征
            market_cols = [x for x in df.columns if x[:2]!='f_']
            market_slice = df[market_cols].iloc[indices].copy()
            market_factors = calculate_minute_factors(market_slice)
            market_factors = market_factors[[x for x in market_factors.columns if x not in meta_cols]]
            concat_list.append(market_factors)

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
    liquid_factors = ['liquidity_skew_mean','liquidity_skewness','dynamic_spread_robust','dynamic_spread_robust_mean','spread_mean','rolling_amihud','deep_spread_rolling']
    train_df[[x+'_std' for x in liquid_factors]] = train_df[liquid_factors]
    valid_df[[x+'_std' for x in liquid_factors]] = valid_df[liquid_factors]
    print(f"Train Shape: {train_df.shape}")

    # 4. 标准化
    all_cols = train_df.columns
    exclude_final = [CONFIG['label_col'], 'timestamp', 'trade_day','LABEL_CAL_DQ_inst1_60','is_pm','is_am','is_night','gate_session_decay','gate_open_impulse']
    final_feature_cols = [c for c in all_cols if c[:2]=='f_']+['OBI_cum120','rolling_amp','gate_rv','gate_downside_ratio','gate_rv_log','gate_range_log','delta_vol_1','delta_vol_2','delta_vol_4','delta_vol_8','delta_vol_16']+[x+'_std' for x in liquid_factors]
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
        train_df.to_pickle(f"traindata_augmented_{CONFIG['window_size']}_market.pkl")
        valid_df.to_pickle(f"validdata_augmented_{CONFIG['window_size']}_market.pkl")