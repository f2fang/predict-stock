import pandas as pd
import numpy as np
import requests
import shap
import joblib
import json
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from fredapi import Fred
from datetime import datetime, timedelta
import os

# API Keys
ALPHA_API_KEY = "7T58L9TI90UXYIV6"
FRED_API_KEY = "a5a1f88198b66ee4c30a2874db599cd6"

# Directories
os.makedirs("daily_logs", exist_ok=True)
os.makedirs("static", exist_ok=True)

def fetch_spy():
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": "SPY",
        "outputsize": "full",
        "datatype": "json",
        "apikey": ALPHA_API_KEY
    }
    r = requests.get(url, params=params)
    ts = r.json().get("Time Series (Daily)", {})
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={"4. close": "SP500_Close"})
    df["SP500_Close"] = df["SP500_Close"].astype(float)
    return df[["SP500_Close"]]

def fetch_macro(start_date):
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
        s = s.resample("D").ffill()
        macro_df = pd.concat([macro_df, s], axis=1) if not macro_df.empty else s
    return macro_df

def predict_once():
    today_str = datetime.today().strftime("%Y-%m-%d")
    pred_path = f"daily_logs/{today_str}_prediction.json"
    shap_path = f"static/{today_str}_shap_force_plot.png"

    if os.path.exists(pred_path) and os.path.exists(shap_path):
        print("✅ Today's prediction already exists. Skipping re-computation.")
        return

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 5)

    spy = fetch_spy()
    macro = fetch_macro(start_date)

    df = pd.concat([macro, spy], axis=1)
    df["SP500_Change"] = df["SP500_Close"].pct_change(fill_method=None)
    df["SP500_Up"] = (df["SP500_Change"] > 0).astype(int)

    for col in macro.columns:
        df[f"{col}_lag"] = df[col].shift(1)

    df = df.dropna()
    features = [f"{col}_lag" for col in macro.columns]
    X = df[features]
    y = df["SP500_Up"]

    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X, y)
    joblib.dump(model, f"daily_logs/{today_str}_model.pkl")

    latest = X.iloc[[-1]]
    proba = model.predict_proba(latest)[0][1]
    percentage = round(float(proba) * 100, 2)

    # SHAP
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

    # Save prediction log safely
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

if __name__ == '__main__':
    predict_once()
