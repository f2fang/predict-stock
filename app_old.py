from flask import Flask, render_template
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import os

app = Flask(__name__)

# Function to fetch the required data
def fetch_data():
    # Time range: 20 years
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 20)
    
    # FRED API key (replace with yours if needed)
    FRED_API_KEY = "a5a1f88198b66ee4c30a2874db599cd6"
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
    
    # FRED indicators
    fred_series = {
        'FedBalanceSheet': 'WALCL',
        'MoneySupply_M2': 'M2SL',
        'ReverseRepo': 'RRPONTSYD',
        'FedFundsRate': 'FEDFUNDS',
        'CPI': 'CPIAUCSL',
        'TenYearYield': 'GS10',
        'GDP_Growth': 'A191RL1Q225SBEA',
        'RetailSales': 'RSXFS',
        'PPI': 'PPIACO',
        'VIX': 'VIXCLS'
    }
    
    # Download and merge all FRED series
    df = pd.DataFrame()
    for name, code in fred_series.items():
        series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
        series = series.to_frame(name)
        series.index = pd.to_datetime(series.index.date)
        df = pd.concat([df, series], axis=1) if not df.empty else series

    # Fetch S&P 500 data from Yahoo Finance
    try:
        print("Fetching S&P 500 index...")
        spx = yf.download("^GSPC", start=start_date, end=end_date)
        spx_close = spx[["Close"]]  # Only keep the 'Close' column
        spx_close.columns = ['SP500_Close']  # Rename column to 'SP500_Close'
        spx_close.index = pd.to_datetime(spx_close.index.date)
        df = pd.concat([df, spx_close], axis=1)
    except Exception as e:
        print(f"⚠️ Failed to fetch S&P 500 data: {e}")

    # Fetch USD to EUR Exchange Rate from Yahoo Finance
    try:
        print("Fetching USD to EUR Exchange Rate...")
        usd_eur = yf.download("EURUSD=X", start=start_date, end=end_date)
        usd_eur_close = usd_eur[["Close"]].rename(columns={"Close": "USD_EUR_ExchangeRate"})
        usd_eur_close.index = pd.to_datetime(usd_eur_close.index.date)
        df = pd.concat([df, usd_eur_close], axis=1)
        print("✅  USD to EUR Exchange Rate data fetched successfully.")
    except Exception as e:
        print(f"⚠️ Failed to fetch USD to EUR Exchange Rate: {e}")

    # Fetch USD Index (DXY) data from Yahoo Finance
    try:
        print("Fetching USD Index (DXY)...")
        dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date)
        dxy_close = dxy[["Close"]]  # Only keep the 'Close' column
        dxy_close.columns = ['USD_Index']  # Rename column to 'USD_Index'
        dxy_close.index = pd.to_datetime(dxy_close.index.date)
        df = pd.concat([df, dxy_close], axis=1)
        print("✅  USD Index data fetched successfully.")
    except Exception as e:
        print(f"⚠️ Failed to fetch USD Index data: {e}")


    # Calculate Net Liquidity
    if "FedBalanceSheet" in df.columns and "ReverseRepo" in df.columns:
        df["NetLiquidity"] = df["FedBalanceSheet"] - df["ReverseRepo"]

    # Clean the data
    df = df.ffill()  # Forward fill missing data

    return df, start_date, end_date


# Function to train and predict S&P 500 for tomorrow (using all factors)
def predict_tomorrow(df, model_lagged):
    selected_features = ["RetailSales", "MoneySupply_M2", "CPI", "NetLiquidity", "FedBalanceSheet","PPI"]#,"USD_Index"]

    # Get the most recent data for the selected features
    latest_data = df.iloc[-1][selected_features]

    # Create lagged features for prediction (using the previous day)
    latest_data_lagged = latest_data.shift(1).values.reshape(1, -1)

    # Predict the S&P 500 for tomorrow
    predicted_sp500_tomorrow = model_lagged.predict(latest_data_lagged)[0]

    return predicted_sp500_tomorrow

# Function to prepare data for linear regression prediction (using only S&P 500 values)
def prepare_sp500_data(df, n_days=5):
    X = []
    y = []

    # Create lagged features (use past 'n_days' to predict next day's S&P500 value
    for i in range(n_days, len(df)):
        X.append(df['SP500_Close'].iloc[i - n_days:i].values)  # Use last 'n_days' S&P500 values
        y.append(df['SP500_Close'].iloc[i])  # The next day's value of S&P 500

    return np.array(X), np.array(y)

# Function to train the Linear Regression model on S&P 500 values
def train_sp500_model(df, n_days=5):
    X, y = prepare_sp500_data(df, n_days)

    # Check for missing values and drop if any
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print(f"Missing values found in the input data. Dropping rows with NaN values.")
        # Drop rows with NaN values in X or y
        df_clean = df.dropna(subset=['SP500_Close'])

        X_clean, y_clean = prepare_sp500_data(df_clean, n_days)
    else:
        X_clean, y_clean = X, y

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    test_score = model.score(X_test, y_test)
    print(f"Test R-squared: {test_score}")

    return model





# Function to predict the next 6 days using the trained Linear Regression model
def predict_next_6_days(df, model, predicted_sp500_tomorrow, n_days=5):
    # Use the last n_days data to start the prediction
    last_data = df['SP500_Close'].iloc[-n_days:].values.reshape(1, -1)

    # Predict for the first day (tomorrow)
    predicted_sp500_week = [predicted_sp500_tomorrow]  # Store tomorrow's prediction

    # Predict the next 6 days based on S&P 500 values only
    for i in range(6):
        # Update the last n_days data with the predicted value
        last_data = np.roll(last_data, shift=-1, axis=1)  # Shift the data to the left
        last_data[0, -1] = predicted_sp500_tomorrow  # Add the predicted value for the next day

        # Predict the next day's value using the updated data
        predicted_sp500_tomorrow = model.predict(last_data)[0]
        predicted_sp500_week.append(predicted_sp500_tomorrow)  # Append the predicted value for the next day

    return predicted_sp500_week



 #Function to train the model
def train_model(df):
    # List of factors to include for the model
    selected_features = ["RetailSales", "MoneySupply_M2", "CPI", "NetLiquidity", "FedBalanceSheet","PPI"]#,"USD_Index"]

    # Create lagged features by shifting the data by one day
    df_lagged = df.copy()
    for feature in selected_features:
        df_lagged[f'{feature}_lag'] = df_lagged[feature].shift(1)

    # Drop rows with NaN values after shifting
    df_lagged = df_lagged.dropna()

    # Prepare the data with lagged features
    X_lagged = df_lagged[[f'{feature}_lag' for feature in selected_features]]  # Use lagged features
    y_lagged = df_lagged["SP500_Close"]

    # Split the data into training and testing sets
    X_train_lagged, X_test_lagged, y_train_lagged, y_test_lagged = train_test_split(X_lagged, y_lagged, test_size=0.2, random_state=42)

    # Train an XGBoost regressor
    model_lagged = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5)
    model_lagged.fit(X_train_lagged, y_train_lagged)

    # You can also check the accuracy of the model here, if needed
    # y_pred_lagged = model_lagged.predict(X_test_lagged)
    # mse_lagged = mean_squared_error(y_test_lagged, y_pred_lagged)
    # r2_lagged = r2_score(y_test_lagged, y_pred_lagged)
    # print(f"Mean Squared Error: {mse_lagged}")
    # print(f"R-squared: {r2_lagged}")

    return model_lagged

# Function to create the subplots of factors vs. S&P 500 and save the plot
def plot_factors_vs_sp500(df):
    # Define the factors you want to plot
    factors = [
        "NetLiquidity", "RetailSales", "FedBalanceSheet", "MoneySupply_M2",
        "ReverseRepo", "FedFundsRate", "CPI", "TenYearYield", "GDP_Growth",
        "PPI", "VIX", "USD_EUR_ExchangeRate", "USD_Index"
    ]

    # Calculate the number of rows and columns dynamically based on the number of factors
    num_factors = len(factors)
    num_columns = 2  # Fixed columns
    num_rows = (num_factors // num_columns) + (1 if num_factors % num_columns != 0 else 0)

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(14, 18), sharex=True)

    # Flatten axes array if more than one row
    axes = axes.flatten()


    # Loop through each factor and plot in a subplot
    for factor_idx, factor in enumerate(factors):
        ax = axes[factor_idx]  # Get the axis for the current subplot
        if factor in df.columns:
            df[factor].plot(ax=ax, label=factor, color="blue")
            df["SP500_Close"].plot(ax=ax, secondary_y=True, label="S&P 500 Close", color="orange")
            ax.set_title(f"{factor} vs. S&P 500 Close")
            ax.set_xlabel("Date")
            ax.set_ylabel(f"{factor} Value")
            ax.right_ax.set_ylabel("S&P 500 Close")
            ax.legend(loc="upper left")
            ax.right_ax.legend(loc="upper right")
            ax.grid(True)
        else:
            ax.set_title(f"⚠️ {factor} Missing")
            ax.axis("off")  # Hide the empty plot if the factor is missing

    # # Loop through each factor and plot in a subplot
    #for row in range(6):
     #   for col in range(2):
      #      if factor_idx < len(factors):
       #         factor = factors[factor_idx]
        #        ax = axes[row, col]
         #       if factor in df.columns:
          #          print(f"Plotting factor: {factor}")
#
 #                   df[factor].plot(ax=ax, label=factor, color="blue")
  #                  df["SP500_Close"].plot(ax=ax, secondary_y=True, label="S&P 500 Close", color="orange")
   #                 ax.set_title(f"{factor} vs. S&P 500 Close")
    #                ax.set_xlabel("Date")
     #               ax.set_ylabel(f"{factor} Value")
      #              ax.right_ax.set_ylabel("S&P 500 Close")
       #             ax.legend(loc="upper left")
        #            ax.right_ax.legend(loc="upper right")
         #           ax.grid(True)
          #      else:
           #         ax.set_title(f"⚠️ {factor} Missing")
            #        ax.axis("off")  # Hide the empty plot if the factor is missing
             #   factor_idx += 1

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot to a file in the 'static' directory
    plot_file_path = os.path.join('static', 'factors_vs_sp500.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path  # Return the path to the saved plot file

def create_actual_vs_predicted_plot(df, predicted_sp500_tomorrow, tomorrow_date):
    plt.figure(figsize=(12, 6))

    # Plot the actual S&P 500 data
    plt.plot(df.index, df['SP500_Close'], label="Actual S&P 500", color="blue")

    # Plot the predicted S&P 500 for tomorrow
    plt.axvline(x=tomorrow_date, color='red', linestyle='--', label="Predicted S&P 500 (Tomorrow)")
    plt.scatter(tomorrow_date, predicted_sp500_tomorrow, color='red', label="Predicted Value")

    plt.title("Actual vs Predicted S&P 500 (Tomorrow)")
    plt.xlabel("Date")
    plt.ylabel("S&P 500 Close")
    plt.legend()
    plt.grid(True)

    # Save the plot in the static folder
    output_plot_file = os.path.join('static', 'actual_vs_predicted_sp500_tomorrow.png')
    plt.tight_layout()
    plt.savefig(output_plot_file)
    plt.close()

    return output_plot_file  # Return the path for the template



# Route to display the prediction and data on the webpage
@app.route('/')
def home():
    # Fetch data
    df, start_date, end_date = fetch_data()
    #print(df.columns)  # Check the columns of the DataFrame


    # Train the model for predicting tomorrow (model_lagged)
    model_lagged = train_model(df)
    # Get the last values for each factor and SP500_Close
    factors_values = {}
    factors = ["NetLiquidity", "RetailSales", "FedBalanceSheet", "MoneySupply_M2",
               "ReverseRepo", "FedFundsRate", "CPI", "TenYearYield", "GDP_Growth",
               "PPI", "VIX", "USD_Index","USD_EUR_ExchangeRate", "SP500_Close"]

    for factor in factors:
        if factor in df.columns:
            factors_values[factor] = df[factor].iloc[-1]  # Get the last value of the factor


    # Predict tomorrow's S&P 500 value using the existing model
    predicted_sp500_tomorrow = predict_tomorrow(df, model_lagged)

    # Train the linear regression model on S&P 500 values (using only the historical S&P 500 data)
    model_sp500 = train_sp500_model(df)

    # Predict the next 6 days using the linear regression model
    predicted_sp500_week = predict_next_6_days(df, model_sp500, predicted_sp500_tomorrow)

    tomorrow_prediction = predicted_sp500_tomorrow

# Get the date for tomorrow's prediction
    tomorrow_date = df.index[-1] + timedelta(days=1)

    # Inside the home function where you call the plotting function
    a_plot_url = create_actual_vs_predicted_plot(df, predicted_sp500_tomorrow, tomorrow_date)

    # Create plot for the next week's prediction
    predicted_dates_week = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7, freq='D')
    plt.figure(figsize=(12, 6))
    plt.plot(predicted_dates_week, predicted_sp500_week, label="Predicted S&P 500 (Next Week)", color="orange", linestyle="--")
    plt.title("Predicted S&P 500 (Next Week)")
    plt.xlabel("Date")
    plt.ylabel("S&P 500 Close")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_plot_file = "static/predicted_sp500_next_week.png"
    plt.savefig(output_plot_file)

  # Generate the plot for factors vs S&P 500
    plot_url_factors = plot_factors_vs_sp500(df)  # This generates and saves the plot

    # Return the HTML page with the prediction and plot links
    return render_template('index.html', actual_vs_predicted_plot_url=a_plot_url, plot_factors_url=plot_url_factors,  tomorrow_prediction=tomorrow_prediction,factors_values=factors_values,predicted_week=predicted_sp500_week, predicted_dates_week=predicted_dates_week, plot_url=output_plot_file, tomorrow_date=tomorrow_date)


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
