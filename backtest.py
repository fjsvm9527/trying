# backtester.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def analyze_and_plot_results(daily_stats_df, initial_capital=1000000):
    """
    Analyzes the aggregated daily backtest results and plots the equity curve.

    Args:
        daily_stats_df (pd.DataFrame): DataFrame with daily PnL and trade counts.
        initial_capital (float): The starting capital for calculating returns.
    """
    if daily_stats_df.empty:
        print("No data to analyze.")
        return

    # --- Performance Calculation ---
    stats = daily_stats_df.copy()
    stats['daily_return'] = stats['pnl'] / initial_capital
    stats['equity_curve'] = (stats['daily_return'] + 1).cumprod() * initial_capital

    # Total Return
    total_return = (stats['equity_curve'].iloc[-1] / initial_capital) - 1

    # Sharpe Ratio (annualized)
    # Assuming 252 trading days in a year
    daily_returns = stats['daily_return']
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

    # Max Drawdown
    cumulative_max = stats['equity_curve'].cummax()
    drawdown = (stats['equity_curve'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    # --- Print Statistics ---
    print("\n" + "="*50)
    print("Backtest Performance Analysis")
    print("="*50)
    stats.index = pd.to_datetime(stats.index)
    print(f"Period: {stats.index.min().strftime('%Y-%m-%d')} to {stats.index.max().strftime('%Y-%m-%d')}")
    print(f"Initial Capital: {initial_capital:,.2f}")
    print(f"Final Net Value: {stats['equity_curve'].iloc[-1]:,.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Total Long Trades: {stats['long_trades'].sum()}")
    print(f"Total Short Trades: {stats['short_trades'].sum()}")
    print(f"Average Daily Trades: {stats['long_trades'].sum() + stats['short_trades'].sum() / len(stats):.2f}")
    print("="*50 + "\n")

    # --- Plotting Equity Curve ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(stats.index, stats['equity_curve'], label='Strategy Net Value', color='royalblue')
    
    # Formatting
    ax.set_title('Strategy Equity Curve', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Net Value (RMB)', fontsize=12)
    ax.legend()
    fig.autofmt_xdate()
    
    # Use a more readable format for y-axis
    from matplotlib.ticker import FuncFormatter
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    plt.show()


def backtest_single_day(daily_data: pd.DataFrame, threshold: float, commission_rate: float):
    """
    Runs a high-performance backtest for a single day's data.

    Args:
        daily_data (pd.DataFrame): DataFrame with columns ['time', 'askp1', 'bidp1', 'signal'].
        threshold (float): The signal value threshold to trigger a trade.
        commission_rate (float): The transaction fee rate.

    Returns:
        dict: A dictionary containing the total PnL, and counts of long and short trades.
    """
    # Convert to NumPy arrays for speed
    # print(daily_data)
    ask_prices = daily_data['askp1'].values
    bid_prices = daily_data['bidp1'].values
    signals = daily_data['signal'].values
    
    n_ticks = len(signals)
    if n_ticks == 0:
        return {'pnl': 0, 'long_trades': 0, 'short_trades': 0}

    # State variables
    position = 0  # 0: flat, 1: long, -1: short
    pnl = 0.0
    entry_price = 0.0
    long_trades = 0
    short_trades = 0

    for i in range(n_ticks):
        signal = signals[i]
        ask_price = ask_prices[i]
        bid_price = bid_prices[i]

        # --- State: No Position ---
        if position == 0:
            if signal > threshold:  # Open Long
                position = 1
                entry_price = ask_price
                pnl -= entry_price * commission_rate  # Entry fee
                long_trades += 1
            elif signal < -threshold:  # Open Short
                position = -1
                entry_price = bid_price
                pnl -= entry_price * commission_rate  # Entry fee
                short_trades += 1

        # --- State: Long Position ---
        elif position == 1:
            if signal <= 0:  # Close Long signal
                position = 0
                exit_price = bid_price
                pnl += (exit_price - entry_price)
                pnl -= exit_price * commission_rate  # Exit fee
                entry_price = 0.0

        # --- State: Short Position ---
        elif position == -1:
            if signal >= 0:  # Close Short signal
                position = 0
                exit_price = ask_price
                pnl += (entry_price - exit_price)  # PnL for short is entry - exit
                pnl -= exit_price * commission_rate  # Exit fee
                entry_price = 0.0

    # --- End of Day Settlement ---
    if position != 0:
        if position == 1:  # Force close long position
            exit_price = bid_prices[-1] # Use last available bid price
            pnl += (exit_price - entry_price)
            pnl -= exit_price * commission_rate
        elif position == -1:  # Force close short position
            exit_price = ask_prices[-1] # Use last available ask price
            pnl += (entry_price - exit_price)
            pnl -= exit_price * commission_rate
            
    return {'pnl': pnl, 'long_trades': long_trades, 'short_trades': short_trades}


def run_backtest(dates: list, data_folder: str, threshold: float, commission_rate: float, initial_capital: float):
    """
    Main function to run the backtest over a series of dates.

    Args:
        dates (list): A list of pd.Timestamp objects for the backtest period.
        data_folder (str): Path to the folder containing daily prediction .pkl files.
        threshold (float): Signal threshold to trigger trades.
        commission_rate (float): Transaction fee rate (e.g., 0.0001 for 万分之一).
        initial_capital (float): The starting capital for calculating returns.
    """
    all_daily_stats = []
    
    print(f"Starting backtest for {len(dates)} days...")
    for date in tqdm(dates, desc="Backtesting Progress"):
        # file_name = f"{date.strftime('%Y%m%d')}.pkl"
        # file_path = os.path.join(data_folder, file_name)

        # if not os.path.exists(file_path):
        #     # print(f"Warning: Data for {date.strftime('%Y-%m-%d')} not found. Skipping.")
        #     continue
            
        # daily_data = pd.read_pickle(file_path)
        global data
        daily_data = data[data.date == date]
        
        # Ensure data is sorted by time, just in case
        daily_data = daily_data.sort_values(by='time').reset_index(drop=True)
        
        # Run the backtest for the day
        daily_result = backtest_single_day(daily_data, threshold, commission_rate)
        
        # Store results
        daily_result['date'] = date
        all_daily_stats.append(daily_result)
        
    if not all_daily_stats:
        print("Error: No data was found for the given dates. Backtest cannot proceed.")
        return
        
    # --- Aggregate and Analyze ---
    results_df = pd.DataFrame(all_daily_stats).set_index('date')
    
    # Calculate daily return and display summary
    analyze_and_plot_results(results_df, initial_capital)
    
    return results_df


if __name__ == '__main__':
    # =================================================================
    #  EXAMPLE USAGE
    # =================================================================
    
    # --- 1. Define Parameters ---
    DATA_FOLDER = "models/InceptionLSTM_Incremental_v1/results" # Folder where your daily results are saved
    THRESHOLD = 0.0001  # Example: trade if signal > 1.2 or < -1.2
    COMMISSION_RATE = 0  # 万分之一
    INITIAL_CAPITAL = 1_000_000 # 1 million starting capital

    # --- 2. Generate Date Range for Backtest ---
    # This should match the test dates from your training script
    data = pd.read_pickle("D:\JT_Summer\models\SAGRU_v1_20250722\\test_predictions.pkl")
    DATES_TO_TEST = data['date'].unique()
    # --- 4. Run the Backtest ---
    backtest_results = run_backtest(
        dates=DATES_TO_TEST,
        data_folder=DATA_FOLDER,
        threshold=THRESHOLD,
        commission_rate=COMMISSION_RATE,
        initial_capital=INITIAL_CAPITAL
    )

    # You can inspect the final results DataFrame if needed
    if backtest_results is not None:
        print("\nDaily PnL and Trade Counts:")
        print(backtest_results.head())