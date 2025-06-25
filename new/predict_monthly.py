import pandas as pd
import numpy as np
import requests
import shap
import joblib
import json
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from fredapi import Fred
from datetime import datetime, timedelta
import os

# API Keys
ALPHA_API_KEY = "7T58L9TI90UXYIV6"
FRED_API_KEY = "a5a1f88198b66ee4c30a2874db599cd6"

# Directories
os.makedirs("monthly_logs", exist_ok=True)
os.makedirs("static", exist_ok=True)

def fetch_monthly_spy():
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_MONTHLY",
        "symbol": "SPY",
        "apikey": ALPHA_API_KEY
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

    # SHAP bar chart
    explainer = shap.Explainer(model)
    shap_values = explainer(latest)

    shap.plots.bar(shap_values[0], show=False)
    plt.title("SHAP Bar Plot: Next Month S&P 500 Prediction Explanation")
    plt.tight_layout()
    plt.savefig(shap_path)
    plt.close()

    # Save log
    log = {
        "month": month_str,
        "predicted_monthly_return": percentage,
        "shap_bar_path": shap_path,
        "macro_features": {col: float(val) for col, val in zip(latest.columns, latest.values[0])}
    }
    with open(pred_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"✅ Monthly prediction for {month_str}: {percentage}%")

if __name__ == '__main__':
    predict_monthly()

