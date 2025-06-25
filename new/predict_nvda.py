import sys
sys.modules["torch"] = None  # ✅ 屏蔽 PyTorch 以避免 GPU 依赖错误

import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

# === 参数设置 ===
symbol = "NVDA"
api_key = "7T58L9TI90UXYIV6"
output_file = f"{symbol}_shap_force_cpu.png"

# === 下载 Alpha Vantage 免费数据 ===
url = "https://www.alphavantage.co/query"
params = {
    "function": "TIME_SERIES_DAILY",
    "symbol": symbol,
    "outputsize": "full",
    "apikey": api_key
}
r = requests.get(url, params=params)
json_data = r.json()

# === 处理返回结构 ===
if "Time Series (Daily)" not in json_data:
    raise RuntimeError(f"❌ 数据获取失败: {json_data.get('Note') or json_data.get('Error Message') or json_data}")

data = json_data["Time Series (Daily)"]
df = pd.DataFrame.from_dict(data, orient="index")
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# ✅ 重命名列为通用格式
df = df.rename(columns={"4. close": "Close", "5. volume": "Volume"})
if "Close" not in df.columns or "Volume" not in df.columns:
    raise RuntimeError("❌ 缺少 Close 或 Volume 字段，请确认 API 是否限流")

df["Close"] = df["Close"].astype(float)
df["Volume"] = df["Volume"].astype(float)

# === 构造特征和标签 ===
df["Return_1d"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_20"] = df["Close"].rolling(20).mean()
df["Volatility"] = df["Close"].rolling(10).std()
df["Volume_Change"] = df["Volume"].pct_change()
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)  # 明天涨 = 1

df = df.dropna()
features = ["Return_1d", "MA_5", "MA_20", "Volatility", "Volume_Change"]
X = df[features]
y = df["Target"]

# === 模型训练 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
model.fit(X_train, y_train)

# === 预测明日上涨概率 ===
latest = X.iloc[[-1]]
proba = model.predict_proba(latest)[0][1]
print(f"📈 {symbol} 明日上涨概率：{proba * 100:.2f}%")

# === SHAP 解释图（CPU 专用） ===
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

# === 准确率评估 ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ 测试集准确率：{accuracy * 100:.2f}%")

