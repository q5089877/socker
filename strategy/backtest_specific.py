import pandas as pd
import vectorbt as vbt
import numpy as np
from dynamic_delay import trade

# --- 1. 設定回測參數 ---
STOCK_ID = "2330"  # 想回測的股票代碼 (例如: "2330", "2603")
START_DATE = "2025-05-01"  # 回測開始日期
END_DATE = "2025-06-30"   # 回測結束日期

STRATEGY_PARAMS = {
    "delay": 15,
    "initial_money": 1000000,  # 初始資金建議提高以符合台股股價
    "max_buy": 1,             # 假設一次買一張 (1000股)，但此策略是以股為單位
    "max_sell": 1,
    "print_log": False        # 回測時建議關閉，保持輸出乾淨
}

# 交易成本設定 (手續費買賣各 0.1425% + 證交稅賣出 0.3%)
# vectorbt 的 fees 會雙向收取，這裡用一個簡化值，更精確的設定需要自訂 Signal-level fees
FEES = 0.001425


def run_backtest():
    """執行特定股票和日期區間的回測"""

    # --- 2. 載入並篩選資料 ---
    # 從 GitHub repo 讀取特定股票的資料
    url = f"https://raw.githubusercontent.com/voidful/tw_stocker/main/data/{STOCK_ID}.csv"
    print(f"載入股票 {STOCK_ID} 資料中...")

    try:
        # 讀取資料，並將 Datetime 欄位設為 index，同時轉換為 datetime 物件
        df = pd.read_csv(url, index_col='Datetime', parse_dates=True)
    except Exception as e:
        print(f"讀取資料失敗: {e}")
        return

    # 確保索引已排序且唯一，以處理日期切片 (slicing)
    # 由於資料可能是分批更新附加的，先排序索引確保時間序列正確
    df.sort_index(inplace=True)
    # 移除重複的索引（可能因重複抓取產生），保留第一個
    df = df[~df.index.duplicated(keep='first')]

    # 篩選出指定日期的資料
    df_filtered = df.loc[START_DATE:END_DATE].copy()

    if df_filtered.empty:
        print(f"錯誤：在 {START_DATE} 到 {END_DATE} 的區間內找不到 {STOCK_ID} 的資料。")
        return

    print(f"資料已篩選，區間為 {df_filtered.index.min()} 至 {df_filtered.index.max()}")

    # --- 3. 執行交易策略 ---
    # 直接將篩選後的 DataFrame 傳入 trade 函式
    states_buy, states_sell, states_entry, states_exit, total_gains, invest = trade(
        df_filtered, **STRATEGY_PARAMS)

    # --- 4. 使用 vectorbt 進行回測 ---
    # trade 函式回傳的 states_entry/exit 是 list，長度為 N-1
    # 需要轉換成與價格序列對齊的 Pandas Series 才能給 vectorbt 使用
    # 這是為了在不修改 dynamic_delay.py 的情況下修正訊號對齊問題
    entry_signals = pd.Series(states_entry, index=df_filtered.index[1:]).reindex(
        df_filtered.index, fill_value=False)
    exit_signals = pd.Series(states_exit, index=df_filtered.index[1:]).reindex(
        df_filtered.index, fill_value=False)

    # 準備 portfolio 的參數
    portfolio_kwargs = dict(
        size=np.inf,
        fees=FEES,
        freq='5m'  # 根據資料頻率設定
    )

    # 使用 vectorbt 的 Portfolio 進行回測
    portfolio = vbt.Portfolio.from_signals(
        df_filtered['close'],
        entries=entry_signals,
        exits=exit_signals,
        **portfolio_kwargs
    )

    # --- 5. 顯示回測結果 ---
    print("\n--- 回測統計數據 ---")
    print(portfolio.stats())

    print("\n--- 繪製回測圖表 ---")
    # 繪製結果圖表 (在腳本執行時會自動開啟)
    fig = portfolio.plot()
    fig.show()


if __name__ == "__main__":
    run_backtest()
