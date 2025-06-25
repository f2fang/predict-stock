import sys
sys.modules["torch"] = None  # 屏蔽 PyTorch GPU 错误

import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

# ==== 参数 ====
symbol = "MSTR"
btc_symbol = "BTC"
output_file = f"{symbol}_shap_force_cpu.png"

# 多个 Alpha Vantage 免费 key
ALPHA_KEYS = [
    "ANS1QUJWSV2FB3KH",
    "7T58L9TI90UXYIV6"
]
current_key_index = 0

def get_next_key():
    global current_key_index
    key = ALPHA_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(ALPHA_KEYS)
    return key

def fetch_with_key_rotation(url, base_params, try_limit=5):
    for _ in range(try_limit):
        key = get_next_key()
        params = base_params.copy()
        params["apikey"] = key
        r = requests.get(url, params=params)
        json_data = r.json()
        if "Note" in json_data or "Information" in json_data or json_data == {}:
            print(f"⚠️ key {key} 超额或无效，切换...")
            continue
        return json_data
    raise RuntimeError("❌ 所有 Alpha Vantage key 均已超额")

# ==== 获取 BTC 数据 ====
btc_url = "https://www.alphavantage.co/query"
btc_params = {
    "function": "DIGITAL_CURRENCY_DAILY",
    "symbol": btc_symbol,
    "market": "USD"
}
btc_json = fetch_with_key_rotation(btc_url, btc_params)
btc_data = btc_json.get("Time Series (Digital Currency Daily)", {})
btc_df = pd.DataFrame.from_dict(btc_data, orient="index")
btc_df.index = pd.to_datetime(btc_df.index)
btc_df = btc_df.sort_index()

# 自动识别 BTC 收盘字段
btc_close_col = None
for col in btc_df.columns:
    if "close" in col.lower():
        btc_close_col = col
        break
if not btc_close_col:
    raise ValueError(f"❌ BTC 收盘字段不存在，字段有：{btc_df.columns.tolist()}")
btc_df = btc_df.rename(columns={btc_close_col: "BTC_Close"})
btc_df["BTC_Close"] = btc_df["BTC_Close"].astype(float)

# ==== 获取 MSTR 股票数据 ====
stock_url = "https://www.alphavantage.co/query"
stock_params = {
    "function": "TIME_SERIES_DAILY",
    "symbol": symbol,
    "outputsize": "full"
}
stock_json = fetch_with_key_rotation(stock_url, stock_params)
stock_data = stock_json.get("Time Series (Daily)", {})
df = pd.DataFrame.from_dict(stock_data, orient="index")
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df = df.rename(columns={"4. close": "Close", "5. volume": "Volume"})
if "Close" not in df.columns or "Volume" not in df.columns:
    raise RuntimeError("❌ 缺少 Close 或 Volume 字段")
df["Close"] = df["Close"].astype(float)
df["Volume"] = df["Volume"].astype(float)

# ==== 合并 BTC 并构建特征 ====
df = df.merge(btc_df[["BTC_Close"]], left_index=True, right_index=True, how="left")
df["BTC_Change"] = df["BTC_Close"].pct_change()
df["Return_1d"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_20"] = df["Close"].rolling(20).mean()
df["Volatility"] = df["Close"].rolling(10).std()
df["Volume_Change"] = df["Volume"].pct_change()
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()

features = ["Return_1d", "MA_5", "MA_20", "Volatility", "Volume_Change", "BTC_Change"]
X = df[features]
y = df["Target"]

# ==== 模型训练 ====
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
model.fit(X_train, y_train)

# ==== 明日预测 ====
latest = X.iloc[[-1]]
proba = model.predict_proba(latest)[0][1]
print(f"📈 {symbol} 明日上涨概率：{proba * 100:.2f}%")

# ==== SHAP 可视化 ====
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(latest)
shap.initjs()
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    latest.iloc[0],
    matplotlib=True,
    show=False
)
plt.title(f"{symbol} SHAP Force Plot: Tomorrow Prediction")
plt.savefig(output_file)
plt.close()
print(f"🖼️ 图表已保存为 {output_file}")

# ==== 模型评估 ====
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ 测试集准确率：{accuracy * 100:.2f}%")

