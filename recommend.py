from typing import Any, Dict, List, Tuple
import nlp2
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import os
from datetime import datetime

from strategy.grid import trade

# 定義近期訊號的時間窗口（以資料點數量計算）
# 台股交易時間為 4.5 小時 (270分鐘)，資料頻率為 5 分鐘。
# 270 / 5 = 54 個資料點/天。這裡設定 27 (約半天) 為近期訊號的閾值。
RECENT_SIGNAL_THRESHOLD_PERIODS = 27


def recommend_stock(url: str, parameters: Dict[str, Any]) -> Tuple[bool, bool, float, float, str, str]:
    """
    根據給定的 URL 和策略參數分析股票數據，判斷是否應該買入或賣出。

    Args:
        url (str): 股票數據 CSV 檔案的路徑或 URL。
        parameters (Dict[str, Any]): 用於交易策略的參數字典。

    Returns:
        Tuple[bool, bool, float, float, str, str]: 一個元組，包含：
        - should_buy (bool): 如果最近有買入信號，則為 True。
        - should_sell (bool): 如果最近有賣出信號，則為 True。
        - today_close_price (float): 最後的收盤價。
        - total_gains (float): 回測的總收益。
        - start_date (str): 數據的開始日期。
        - end_date (str): 數據的結束日期。

    Raises:
        FileNotFoundError: 如果 CSV 檔案不存在。
        Exception: 如果數據處理或策略計算中發生其他錯誤。
    """
    df = pd.read_csv(url, index_col='Datetime')
    # 為了處理混合時區數據並消除警告，先將所有時間戳轉換為統一的 UTC 時間，
    # 然後再轉換回 'Asia/Taipei' 時區，以確保後續處理的時區一致性。
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Asia/Taipei')
    # 確保索引已排序，以處理分批更新附加的資料
    df.sort_index(inplace=True)
    df.columns = map(str.lower, df.columns)
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    states_buy, states_sell, states_entry, states_exit, total_gains, invest = trade(
        df, **parameters)

    today = len(df)
    today_close_price = df.close.iloc[-1]
    start_date = df.index[0].strftime('%Y-%m-%d')
    end_date = df.index[-1].strftime('%Y-%m-%d')

    # 處理策略未產生任何買入/賣出信號的情況，避免 IndexError
    should_buy = False
    if states_buy:  # 檢查列表是否為空
        should_buy = abs(
            today - states_buy[-1]) < RECENT_SIGNAL_THRESHOLD_PERIODS

    should_sell = False
    if states_sell:  # 檢查列表是否為空
        should_sell = abs(
            today - states_sell[-1]) < RECENT_SIGNAL_THRESHOLD_PERIODS

    return should_buy, should_sell, today_close_price, total_gains, start_date, end_date


def generate_report(urls: List[str], parameters: Dict[str, Any], limit: int = 10):
    """
    生成推薦股票的 HTML 報告。

    Args:
        urls (List[str]): 股票數據 CSV 檔案的路徑或 URL 列表。
        parameters (Dict[str, Any]): 用於交易策略的參數字典。
        limit (int, optional): 報告中包含的最大股票數量。預設為 10。
    """
    results = []
    backtest_start_date, backtest_end_date = None, None
    print("--- 正在生成報告 ---")
    for url in urls:
        try:
            should_buy, should_sell, today_close_price, total_gains, start_date, end_date = recommend_stock(
                url, parameters)
            if should_sell or should_buy:
                results.append({
                    "Stock": os.path.splitext(os.path.basename(url))[0],
                    "Should_Buy": should_buy,
                    "Should_Sell": should_sell,
                    "Recommended_Price": today_close_price,
                    "Total_Gains": total_gains,
                })
                # 記錄第一個成功處理的檔案的日期作為報告的日期區間
                if not backtest_start_date:
                    backtest_start_date = start_date
                    backtest_end_date = end_date
        except Exception as e:
            # 打印錯誤而不是靜默忽略，以便於調試
            print(f"處理 {url} 時發生錯誤: {e}")

    # 根據總收益排序並選出前 limit 檔股票
    sorted_results = sorted(
        results, key=lambda x: x['Total_Gains'], reverse=True)[:limit]
    df = pd.DataFrame(sorted_results)

    # 準備要傳遞到模板的參數，分門別類讓模板更清晰
    backtest_params = {
        "初始本金": parameters.get("initial_money"),
        "回測開始日期": backtest_start_date,
        "回測結束日期": backtest_end_date,
    }
    strategy_params = {
        "RSI 週期": parameters.get("rsi_period"),
        "低 RSI 閾值": parameters.get("low_rsi"),
        "高 RSI 閾值": parameters.get("high_rsi"),
    }

    # 建立模板環境，使用相對於此腳本檔案的絕對路徑
    # 這樣可以確保無論從哪裡執行腳本，都能找到模板檔案
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('stock_report_template.html')
    html_output = template.render(
        stocks=df.to_dict(orient='records'),
        strategy_params=strategy_params,
        backtest_params=backtest_params
    )

    output_path = os.path.join(script_dir, 'stock_report.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    print(f"報告已生成: {output_path}")


def main():
    """
    主執行函數，用於運行股票推薦和報告生成。
    """
    parameters = {
        "initial_money": 10000,
        "rsi_period": 14,
        "low_rsi": 30,
        "high_rsi": 70,
        "ema_period": 26,
    }

    stock_files = list(nlp2.get_files_from_dir("data"))

    print("--- 開始進行個股分析 ---")
    for file_path in stock_files:
        try:
            # recommend_stock 現在回傳 6 個值
            should_buy, should_sell, today_close_price, total_gains, _, _ = recommend_stock(
                file_path, parameters)
            if should_sell or should_buy:
                stock_id = os.path.splitext(os.path.basename(file_path))[0]
                print(
                    f"股票代號: {stock_id:<10} | 建議買入: {str(should_buy):<5} | 建議賣出: {str(should_sell):<5} | 目前價格: {today_close_price:<8.2f} | 策略收益: {total_gains:.2f}"
                )
        except Exception as e:
            print(f"分析 {file_path} 時發生錯誤: {e}")

    generate_report(stock_files, parameters)


if __name__ == "__main__":
    main()
