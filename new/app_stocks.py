import pandas as pd
import numpy as np
import requests
import shap
import joblib
import json
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from fredapi import Fred
from datetime import datetime, timedelta
import os
import pandas as pd
import requests
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from flask import Flask, render_template, request


os.makedirs("stock_logs", exist_ok=True)

def stock_log_path(symbol):
    today_str = datetime.today().strftime("%Y-%m-%d")
    return f"stock_logs/{symbol}_{today_str}.json"

app = Flask(__name__)

# API Keys
FRED_API_KEY = "a5a1f88198b66ee4c30a2874db599cd6"

# 多个 Alpha Vantage API Keys
ALPHA_KEYS = [
    "SM5968YULPLQ5Z45",
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


# Directories
os.makedirs("daily_logs", exist_ok=True)
os.makedirs("monthly_logs", exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.route("/")
def home():
    # 获取最新的 daily 和 monthly json 文件
    latest_daily = sorted([f for f in os.listdir("daily_logs") if f.endswith(".json")])[-1]
    latest_monthly = sorted([f for f in os.listdir("monthly_logs") if f.endswith(".json")])[-1]

    with open(os.path.join("daily_logs", latest_daily)) as f:
        daily_prediction = json.load(f)

    with open(os.path.join("monthly_logs", latest_monthly)) as f:
        monthly_prediction = json.load(f)

    return render_template("index.html",
                           daily_prediction=daily_prediction,
                           monthly_prediction=monthly_prediction)
@app.route("/stocks")
def stock_predictions():
    today_str = datetime.today().strftime("%Y-%m-%d")
    symbols = ["NVDA", "MSTR", "IAU"]
    predictions = []

    for symbol in symbols:
        path = f"stock_logs/{symbol}_{today_str}.json"
        if os.path.exists(path):
            with open(path) as f:
                predictions.append(json.load(f))
        else:
            predictions.append({
                "symbol": symbol,
                "date": today_str,
                "probability": "N/A",
                "accuracy": "N/A",
                "shap_image": "none.png"
            })

    return render_template("stocks.html", predictions=predictions)


#def stock_predictions():

 #   def load_prediction(symbol):
  #      filename = f"{symbol}_prediction.json"
   #     if os.path.exists(filename):
    #        with open(filename) as f:
     #           return json.load(f)
      #  else:
       #     return {"probability": "N/A", "shap_image": "none.png", "symbol": symbol}

    #symbols = ["NVDA", "MSTR", "IAU"]
    #predictions = [load_prediction(sym) for sym in symbols]
    #return render_template("stocks.html", predictions=predictions)

def predict_generic(symbol, api_key= get_next_key()):
    output_file = f"static/{symbol}_shap_force_cpu.png"
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": api_key
    }
    r = requests.get(url, params=params)
    json_data = r.json()

    if "Time Series (Daily)" not in json_data:
        print(f"❌ 无法获取 {symbol} 数据: {json_data}")
        return

    data = json_data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={"4. close": "Close", "5. volume": "Volume"})

    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)

    df["Return_1d"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Volatility"] = df["Close"].rolling(10).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna()
    features = ["Return_1d", "MA_5", "MA_20", "Volatility", "Volume_Change"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train, y_train)

    latest = X.iloc[[-1]]
    proba = model.predict_proba(latest)[0][1]

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

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    log = {
        "symbol": symbol,
        "date": str(df.index[-1].date()),
        "probability": float(round(proba * 100, 2)),
        "accuracy": float(round(accuracy * 100, 2)),
        "shap_image": output_file.replace("static/", "")
    }

    log_path = stock_log_path(symbol)
    if os.path.exists(log_path):
        print(f"✅ {symbol} 今天已预测，跳过")
        return
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"✅ {symbol} 预测完成：{log['probability']}% 上涨概率，准确率 {log['accuracy']}%")

def predict_nvda():
    predict_generic("NVDA")

def predict_mstr():
    symbol = "MSTR"
    btc_url = "https://www.alphavantage.co/query"
    btc_params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": "BTC",
        "market": "USD"
    }
    btc_json = fetch_with_key_rotation(btc_url, btc_params)
    btc_data = btc_json.get("Time Series (Digital Currency Daily)", {})
    btc_df = pd.DataFrame.from_dict(btc_data, orient="index")
    btc_df.index = pd.to_datetime(btc_df.index)
    btc_df = btc_df.sort_index()

    # 自动识别 BTC close 字段
    btc_close_col = None
    for col in btc_df.columns:
        if "close" in col.lower():
            btc_close_col = col
            break
    if not btc_close_col:
        raise ValueError(f"❌ BTC 收盘价字段不存在，字段有：{btc_df.columns.tolist()}")
    btc_df = btc_df.rename(columns={btc_close_col: "BTC_Close"})
    btc_df["BTC_Close"] = btc_df["BTC_Close"].astype(float)

    # 获取 MSTR 股票数据
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full"
    }
    stock_json = fetch_with_key_rotation(url, params)
    data = stock_json.get("Time Series (Daily)", {})
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={"4. close": "Close", "5. volume": "Volume"})
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)

    # 合并 BTC
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train, y_train)

    latest = X.iloc[[-1]]
    proba = model.predict_proba(latest)[0][1]

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
    output_file = "static/MSTR_shap_force_cpu.png"
    plt.title("MSTR SHAP Force Plot: Tomorrow Prediction")
    plt.savefig(output_file)
    plt.close()

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    log = {
        "symbol": symbol,
        "date": str(df.index[-1].date()),
        "probability": float(round(proba * 100, 2)),
        "accuracy": float(round(accuracy * 100, 2)),
        "shap_image": output_file.replace("static/", "")
    }
    log_path = stock_log_path(symbol)
    if os.path.exists(log_path):
        print(f"✅ {symbol} 今天已预测，跳过")
        return
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


    print(f"✅ {symbol}（含 BTC 特征）预测完成：{log['probability']}% 上涨概率，准确率 {log['accuracy']}%")



def predict_iau():
    predict_generic("IAU")

@app.route("/stocks/history")
def stocks_history():
    history = {}
    for fname in os.listdir("stock_logs"):
        if fname.endswith(".json"):
            symbol = fname.split("_")[0]
            history.setdefault(symbol, []).append(fname)
    return render_template("stocks_history.html", history=history)

@app.route("/stocks/view")
def view_stock_log():
    filename = request.args.get("file")
    path = os.path.join("stock_logs", filename)
    if not os.path.exists(path):
        return f"找不到日志文件: {filename}", 404
    with open(path) as f:
        data = json.load(f)
    return render_template("view_stock_log.html", data=data)


@app.route("/history")
def history():
    daily_files = sorted([f for f in os.listdir("daily_logs") if f.endswith("_prediction.json")])
    monthly_files = sorted([f for f in os.listdir("monthly_logs") if f.endswith("_prediction.json")])
    return render_template("history_overview.html", daily_files=daily_files, monthly_files=monthly_files)

@app.route("/history/daily")
def history_daily():
    records = []
    for fname in sorted(os.listdir("daily_logs")):
        if fname.endswith("_prediction.json"):
            with open(os.path.join("daily_logs", fname)) as f:
                entry = json.load(f)
                records.append(entry)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    try:
        sp500 = pd.read_csv("https://query1.finance.yahoo.com/v7/finance/download/SPY?period1=1600000000&period2=9999999999&interval=1d&events=history")
        sp500["Date"] = pd.to_datetime(sp500["Date"])
        sp500 = sp500.set_index("Date")["Close"].rename("SP500_Close")
        df = df.set_index("date").join(sp500).dropna().reset_index()
        df["Actual_Up"] = df["SP500_Close"].pct_change().gt(0)
        df["Pred_Up"] = df["prediction_probability"] > 50
        df["Correct"] = df["Actual_Up"] == df["Pred_Up"]
    except Exception as e:
        print("⚠️ Failed to fetch actual SPY data", e)
        df["Actual_Up"] = None
        df["Correct"] = None

    # Save plot
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(df["date"], df["prediction_probability"], label="预测上涨概率")
        plt.plot(df["date"], df["SP500_Close"].pct_change()*100, label="实际涨跌 (%)")
        plt.axhline(50, color="gray", linestyle="--")
        plt.legend()
        plt.title("Daily Prediction vs Actural Result")
        plt.tight_layout()
        plt.savefig("static/daily_history_plot.png")
        plt.close()
    except Exception as e:
        print("⚠️ Failed to generate plot:", e)


    return render_template("history_daily.html", records=df.to_dict(orient="records"), plot_url="/static/daily_history_plot.png")

@app.route("/history/monthly")
def history_monthly():
    records = []
    for fname in sorted(os.listdir("monthly_logs")):
        if fname.endswith("_prediction.json"):
            with open(os.path.join("monthly_logs", fname)) as f:
                entry = json.load(f)
                records.append(entry)
    df = pd.DataFrame(records)
    df["month"] = pd.to_datetime(df["month"])

    try:
        sp500 = pd.read_csv("https://query1.finance.yahoo.com/v7/finance/download/SPY?period1=1600000000&period2=9999999999&interval=1mo&events=history")
        sp500["Date"] = pd.to_datetime(sp500["Date"])
        sp500 = sp500.set_index("Date")["Close"].rename("SP500_Close")
        sp500 = sp500.resample("M").last().pct_change()*100
        df = df.set_index("month").join(sp500.rename("Actual_Return")).dropna().reset_index()
        df["Error"] = df["predicted_monthly_return"] - df["Actual_Return"]
    except Exception as e:
        print("⚠️ Failed to fetch SPY monthly", e)
        df["Actual_Return"] = None
        df["Error"] = None

    # Save plot
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(df["month"], df["predicted_monthly_return"], label="预测月涨幅")
        plt.plot(df["month"], df["Actual_Return"], label="实际涨幅")
        plt.axhline(0, color="gray", linestyle="--")
        plt.legend()
        plt.title("每月预测 vs 实际对比")
        plt.tight_layout()
        plt.savefig("static/monthly_history_plot.png")
        plt.close()
    except:
        pass

    return render_template("history_monthly.html", records=df.to_dict(orient="records"), plot_url="/static/monthly_history_plot.png")


@app.route("/history/daily/view")
def view_daily():
    filename = request.args.get("file")
    with open(os.path.join("daily_logs", filename)) as f:
        data = json.load(f)
    return render_template("view_daily.html", data=data)

@app.route("/history/monthly/view")
def view_monthly():
    filename = request.args.get("file")
    with open(os.path.join("monthly_logs", filename)) as f:
        data = json.load(f)
    return render_template("view_monthly.html", data=data)

# ========= DAILY PREDICTION =========
def predict_once():
    today_str = datetime.today().strftime("%Y-%m-%d")
    pred_path = f"daily_logs/{today_str}_prediction.json"
    shap_path = f"static/{today_str}_shap_force_plot.png"

    if os.path.exists(pred_path) and os.path.exists(shap_path):
        print("✅ Today's prediction already exists. Skipping re-computation.")
        return


    spy_url = "https://www.alphavantage.co/query"
    spy_params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": "SPY",
        "outputsize": "full",
        "datatype": "json",
    }

    spy_json = fetch_with_key_rotation(spy_url, spy_params)
    ts = spy_json.get("Time Series (Daily)", {})
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={"4. close": "SP500_Close"})
    df["SP500_Close"] = df["SP500_Close"].astype(float)


    fred = Fred(api_key=FRED_API_KEY)
    indicators = {
        'MoneySupply_M2': 'M2SL',
        'CPI': 'CPIAUCSL',
        'FedFundsRate': 'FEDFUNDS',
        'FedBalanceSheet': 'WALCL',
        'ReverseRepo': 'RRPONTSYD',
        'TenYearYield': 'GS10'
    }
    macro_df = pd.DataFrame()
    for name, code in indicators.items():
        s = fred.get_series(code, observation_start=df.index.min())
        s = s.to_frame(name)
        s.index = pd.to_datetime(s.index)
        s = s.resample("D").ffill()
        macro_df = pd.concat([macro_df, s], axis=1) if not macro_df.empty else s

    merged = pd.concat([macro_df, df], axis=1)
    merged["SP500_Change"] = merged["SP500_Close"].pct_change()
    merged["SP500_Up"] = (merged["SP500_Change"] > 0).astype(int)

    for col in macro_df.columns:
        merged[f"{col}_lag"] = merged[col].shift(1)

    merged = merged.dropna()
    features = [f"{col}_lag" for col in macro_df.columns]
    X = merged[features]
    y = merged["SP500_Up"]

    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X, y)
    joblib.dump(model, f"daily_logs/{today_str}_model.pkl")

    latest = X.iloc[[-1]]
    proba = model.predict_proba(latest)[0][1]
    percentage = round(float(proba) * 100, 2)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(latest)
    shap.initjs()
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values[0],
        features=latest.iloc[0],
        feature_names=latest.columns,
        matplotlib=True,
        show=False
    )
    plt.title("SHAP Force Plot: 明天 S&P 500 涨跌预测解释")
    plt.tight_layout()
    plt.savefig(shap_path)
    plt.close()

    macro_data = {col: float(val) for col, val in zip(latest.columns, latest.values[0])}
    log = {
        "date": today_str,
        "prediction_probability": float(percentage),
        "shap_plot": shap_path,
        "top_features": macro_data
    }
    with open(pred_path + ".tmp", "w") as f:
        json.dump(log, f, indent=2)
    os.replace(pred_path + ".tmp", pred_path)

    print(f"✅ Prediction saved for {today_str}: {percentage}%")

# ========= MONTHLY PREDICTION =========
def fetch_monthly_spy():
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_MONTHLY",
        "symbol": "SPY",
        "apikey": get_next_key()
    }
    r = requests.get(url, params=params)
    ts = r.json().get("Monthly Time Series", {})
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={"4. close": "SP500_Close"})
    df["SP500_Close"] = df["SP500_Close"].astype(float)
    return df[["SP500_Close"]]

def fetch_monthly_macro(start_date):
    fred = Fred(api_key=FRED_API_KEY)
    indicators = {
        'MoneySupply_M2': 'M2SL',
        'CPI': 'CPIAUCSL',
        'FedFundsRate': 'FEDFUNDS',
        'FedBalanceSheet': 'WALCL',
        'ReverseRepo': 'RRPONTSYD',
        'TenYearYield': 'GS10'
    }
    macro_df = pd.DataFrame()
    for name, code in indicators.items():
        s = fred.get_series(code, observation_start=start_date)
        s = s.to_frame(name)
        s.index = pd.to_datetime(s.index)
        s = s.resample("M").last()
        macro_df = pd.concat([macro_df, s], axis=1) if not macro_df.empty else s
    return macro_df

def predict_monthly():
    month_str = datetime.today().strftime("%Y-%m")
    pred_path = f"monthly_logs/{month_str}_prediction.json"
    shap_path = f"static/{month_str}_shap_bar.png"

    if os.path.exists(pred_path) and os.path.exists(shap_path):
        print("✅ This month's prediction already exists.")
        return

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 5)

    spy_df = fetch_monthly_spy()
    macro_df = fetch_monthly_macro(start_date)

    df = pd.concat([macro_df, spy_df], axis=1)
    df["SP500_Return"] = df["SP500_Close"].pct_change(fill_method=None)

    lag_features = list(macro_df.columns)
    for f in lag_features:
        df[f"{f}_lag"] = df[f].shift(1)

    df = df.dropna()
    X = df[[f"{f}_lag" for f in lag_features]]
    y = df["SP500_Return"]

    model = XGBRegressor()
    model.fit(X, y)
    joblib.dump(model, f"monthly_logs/{month_str}_model.pkl")

    latest = X.iloc[[-1]]
    predicted_return = model.predict(latest)[0]
    percentage = round(float(predicted_return) * 100, 2)

    explainer = shap.Explainer(model)
    shap_values = explainer(latest)
    shap.plots.bar(shap_values[0], show=False)
    plt.title("SHAP Bar Plot: Next Month S&P 500 Prediction Explanation")
    plt.tight_layout()
    plt.savefig(shap_path)
    plt.close()

    log = {
        "month": month_str,
        "predicted_monthly_return": float(percentage),
        "shap_bar_path": shap_path,
        "macro_features": {col: float(val) for col, val in zip(latest.columns, latest.values[0])}
    }
    with open(pred_path + ".tmp", "w") as f:
        json.dump(log, f, indent=2)
    os.replace(pred_path + ".tmp", pred_path)

    print(f"✅ Monthly prediction for {month_str}: {percentage}%")

if __name__ == '__main__':
    predict_once()
    predict_monthly()

    predict_mstr()
    predict_nvda()
    predict_iau()
    app.run(host="0.0.0.0", port=5000, debug=True)

