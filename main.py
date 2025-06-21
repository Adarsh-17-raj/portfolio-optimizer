# main.py
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns
if not os.path.exists("plots"):
    os.makedirs("plots")

# ------------------ USER INPUT ------------------ #
tickers = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL']  # You can change these
start_date = "2020-01-01"
end_date = "2024-12-31"

# ------------------ DATA DOWNLOAD ------------------ #
print("Downloading stock data...")
raw_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
data = pd.DataFrame({ticker: raw_data[ticker]['Adj Close'] for ticker in tickers})
data.to_csv("stock_data.csv")
print("Data downloaded.\n")

# ------------------ RETURNS CALCULATION ------------------ #
returns = data.pct_change().dropna()

# ------------------ EDA: CORRELATION ------------------ #
plt.figure(figsize=(10, 6))
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm")
plt.title("Stock Return Correlation")
plt.savefig("plots/correlation_matrix.png")
plt.close()

# ------------------ MONTE CARLO SIMULATION ------------------ #
print("Running portfolio simulation...")

num_portfolios = 10000
all_weights = np.zeros((num_portfolios, len(tickers)))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    all_weights[i] = weights
    ret_arr[i] = np.sum(returns.mean() * weights * 252)
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_arr[i] = ret_arr[i] / vol_arr[i]

plt.figure(figsize=(12, 6))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Monte Carlo Portfolio Simulation")
plt.savefig("plots/monte_carlo.png")
plt.close()

# ------------------ PORTFOLIO OPTIMIZATION ------------------ #
print("Optimizing portfolio...")

mu = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("\nOptimized Weights:")
for stock, weight in cleaned_weights.items():
    print(f"{stock}: {weight:.2%}")

expected_perf = ef.portfolio_performance(verbose=True)

# ------------------ VISUALIZE ALLOCATION ------------------ #
plt.figure(figsize=(10, 6))
plt.bar(cleaned_weights.keys(), cleaned_weights.values(), color='teal')
plt.title("Optimized Portfolio Allocation")
plt.ylabel("Allocation %")
plt.savefig("plots/optimized_allocation.png")
plt.close()

print("\nAll plots saved in 'plots/' folder.")
