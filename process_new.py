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
            meta_cols = [CONFIG['label_col'], 'timestamp', 'trade_day','triggerInst.volume','OBI_cum120','rolling_amp','LABEL_CAL_DQ_inst1_60','is_pm','is_am','is_night','gate_session_decay','gate_open_impulse','gate_rv','gate_downside_ratio','gate_rv_log','gate_range_log','liquidity_skew_mean','liquidity_skewness','dynamic_spread_robust','dynamic_spread_robust_mean','spread_mean','rolling_amihud','deep_spread_rolling','liquidity_skew_mean_600','dynamic_spread_robust_mean_600']
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
    # train_df = pd.read_pickle(os.path.join(CONFIG['save_dir'], "traindata_origin_new.pkl"))
    # valid_df = pd.read_pickle(os.path.join(CONFIG['save_dir'], "validdata_origin_new.pkl"))
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
        train_df.to_pickle(f"traindata_augmented_{CONFIG['window_size']}_newc.pkl")
        valid_df.to_pickle(f"validdata_augmented_{CONFIG['window_size']}_newc.pkl")