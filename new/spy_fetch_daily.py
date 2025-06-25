import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def fetch_and_save_spy():
    today = datetime.today().strftime('%Y-%m-%d')
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    try:
        print("📡 正在从 yfinance 获取 SPY 数据...")
        spy = yf.download("SPY", period="5y", interval="1d", progress=False)
        if spy.empty:
            print("❌ SPY 数据为空，可能是被限制了访问频率。")
            return
        
        spy["SP500_Close"] = spy["Close"]
        output_file = f"{output_dir}/spy_{today}.csv"
        spy[["SP500_Close"]].to_csv(output_file)

        print(f"✅ SPY 数据已保存到 {output_file}")
    except Exception as e:
        print("❌ 获取 SPY 数据失败:", e)

if __name__ == "__main__":
    fetch_and_save_spy()

