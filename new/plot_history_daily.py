# plot_history_daily.py
import os, json
import pandas as pd
import matplotlib.pyplot as plt

records = []
for fname in sorted(os.listdir("daily_logs")):
    if fname.endswith("_prediction.json"):
        with open(os.path.join("daily_logs", fname)) as f:
            entry = json.load(f)
            records.append(entry)
df = pd.DataFrame(records)
df["date"] = pd.to_datetime(df["date"]).dt.date

sp500 = pd.read_csv("https://query1.finance.yahoo.com/v7/finance/download/SPY?period1=1600000000&period2=9999999999&interval=1d&events=history")
sp500["Date"] = pd.to_datetime(sp500["Date"])
sp500 = sp500.set_index("Date")["Close"].rename("SP500_Close")
df = df.set_index("date").join(sp500).dropna().reset_index()
df["SP500_Change"] = df["SP500_Close"].pct_change()
df = df.dropna(subset=["SP500_Change"])
df["Actual_Up"] = df["SP500_Change"] > 0
df["Pred_Up"] = df["prediction_probability"] > 50
df["Correct"] = df["Actual_Up"] == df["Pred_Up"]

plt.figure(figsize=(10, 4))
plt.plot(df["date"], df["prediction_probability"], label="预测上涨概率")
plt.plot(df["date"], df["SP500_Change"] * 100, label="实际涨跌 (%)")
plt.axhline(50, color="gray", linestyle="--")
plt.legend()
plt.title("每日预测 vs 实际对比")
plt.tight_layout()
plt.savefig("static/daily_history_plot.png")
plt.close()
print("✅ 图表已保存")

