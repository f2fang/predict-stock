# Predict Stock System

This project provides a simple AI-powered web system to predict S&P 500 direction and other stock-related predictions.

## Features

- Daily S&P 500 direction prediction
- SHAP visual explanation of model results
- Historical prediction logging
- Simple web interface (Flask based)

## Folder Structure

- `new/` : The latest working version of the project
- `app_old.py` : Legacy version for reference only

## How to Run

**Make sure you're in the project root directory.**

Start the system using:

```bash
new/restart_flask.sh
```

This will:

Start the Flask web app in the background

Log output to flask.log

## Access
Once running, visit:
```
http://<your-server-ip>:5000/
http://<your-server-ip>:5000/stocks
```
## Notes

The code is structured for personal research and testing purposes.

Requires Python 3.8+ and typical machine learning libraries (sklearn, pandas, etc.).

© 2025 Fang Fang
