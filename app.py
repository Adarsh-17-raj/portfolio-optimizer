import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("📊 Portfolio Optimizer")
st.write("An interactive tool to build and optimize a stock portfolio using Modern Portfolio Theory.")

# ---- USER INPUTS ----
tickers_input = st.text_input("Enter stock tickers (comma separated)", value="AAPL,MSFT,GOOG,AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date(2020, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=date.today())

if st.button("Optimize Portfolio"):

    # ---- FETCH DATA ----
    st.subheader("📈 Fetching Data...")
    raw_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
    data = pd.DataFrame({ticker: raw_data[ticker]['Adj Close'] for ticker in tickers})

    returns = data.pct_change().dropna()

    st.write(f"Loaded data for {len(tickers)} stocks.")
    st.line_chart(data)

    # ---- SIMULATION ----
    st.subheader("⚙️ Simulating Portfolios...")
    num_portfolios = 10000
    all_weights = []
    ret_arr = []
    vol_arr = []
    sharpe_arr = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        all_weights.append(weights)
        ret_arr.append(np.sum(returns.mean() * weights * 252))
        vol_arr.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))))
        sharpe_arr.append(ret_arr[-1] / vol_arr[-1])

    # ---- BEST PORTFOLIO ----
    max_sharpe_idx = np.argmax(sharpe_arr)
    optimal_weights = all_weights[max_sharpe_idx]
    optimal_return = ret_arr[max_sharpe_idx]
    optimal_volatility = vol_arr[max_sharpe_idx]
    optimal_sharpe = sharpe_arr[max_sharpe_idx]

    st.subheader("✅ Optimal Portfolio")
    st.write("### Weights:")
    opt_df = pd.DataFrame({'Stock': tickers, 'Weight': optimal_weights})
    st.dataframe(opt_df)

    st.metric("Expected Return", f"{optimal_return:.2%}")
    st.metric("Volatility (Risk)", f"{optimal_volatility:.2%}")
    st.metric("Sharpe Ratio", f"{optimal_sharpe:.2f}")

    # ---- PLOT ----
    st.subheader("📊 Efficient Frontier")
    fig, ax = plt.subplots()
    scatter = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
    ax.scatter(optimal_volatility, optimal_return, c='red', s=50)  # optimal
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.set_title("Efficient Frontier")
    fig.colorbar(scatter, label='Sharpe Ratio')
    st.pyplot(fig)
