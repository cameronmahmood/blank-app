import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Momentum Strategy Backtest")

# --- Parameters ---
start_date = "2019-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
lookback_months = 3
holding_period_months = 1
top_n = 5
transaction_cost = 0.002
risk_free_rate = 0.02 / 12

main_tickers = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'NVDA', 'META', 'AMZN', 'NFLX', 'JPM', 'UNH',
                'V', 'MA', 'HD', 'BAC', 'XOM', 'WMT', 'PEP', 'KO', 'CSCO', 'INTC']

with st.spinner("Downloading data..."):
    price_data = yf.download(main_tickers, start=start_date, end=end_date, auto_adjust=True)[['Close']]
    price_data.columns = price_data.columns.droplevel(0)
    spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)['Close']
    price_data['SPY'] = spy_data

monthly_prices = price_data.resample('ME').last()
momentum = monthly_prices.pct_change(periods=lookback_months)

st.subheader("Momentum Scores (Last 10 Rows)")
st.dataframe(momentum.tail(10).round(4))

last_rebalance = momentum.dropna().index[-1]
st.subheader(f"Latest Momentum Ranking on {last_rebalance.strftime('%Y-%m-%d')}")
st.dataframe(momentum.loc[last_rebalance].drop('SPY').sort_values(ascending=False).round(4))

# --- Backtest ---
portfolio_value = 1000
portfolio_values, spy_values, monthly_returns, rebalance_log = [], [], [], []
selection_matrix = pd.DataFrame(0, index=momentum.index, columns=monthly_prices.columns)
rebalance_dates = momentum.dropna().index

for date in rebalance_dates:
    try:
        past_returns = momentum.loc[date].drop('SPY')
        top_stocks = list(past_returns.nlargest(top_n).index)

        entry_prices = monthly_prices.loc[date, top_stocks]
        exit_idx = monthly_prices.index.get_loc(date) + holding_period_months
        if exit_idx >= len(monthly_prices.index):
            break
        exit_date = monthly_prices.index[exit_idx]
        exit_prices = monthly_prices.loc[exit_date, top_stocks]

        spy_entry = monthly_prices.loc[date, 'SPY']
        spy_exit = monthly_prices.loc[exit_date, 'SPY']
        spy_return = (spy_exit - spy_entry) / spy_entry

        gross_returns = (exit_prices - entry_prices) / entry_prices
        net_returns = gross_returns - transaction_cost
        avg_return = net_returns.mean()

        portfolio_value *= (1 + avg_return)
        spy_value = spy_values[-1] * (1 + spy_return) if spy_values else 1000

        portfolio_values.append(portfolio_value)
        spy_values.append(spy_value)
        monthly_returns.append(avg_return)

        selection_matrix.loc[date, top_stocks] = 1

        rebalance_log.append({
            'Rebalance Date': date.strftime('%Y-%m-%d'),
            'Exit Date': exit_date.strftime('%Y-%m-%d'),
            'Top Stocks': ', '.join(top_stocks),
            'Monthly Return (%)': round(avg_return * 100, 2)
        })

    except Exception as e:
        continue

result_df = pd.DataFrame({
    'Portfolio Value': portfolio_values,
    'SPY Value': spy_values,
    'Monthly Return': monthly_returns
}, index=rebalance_dates[:len(portfolio_values)])

returns = pd.Series(monthly_returns, index=result_df.index)
excess_returns = returns - risk_free_rate
sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(12)
volatility = returns.std() * np.sqrt(12)
cumulative = pd.Series(portfolio_values).pct_change().add(1).cumprod()
drawdown = 1 - cumulative / cumulative.cummax()
max_drawdown = drawdown.max()
total_years = (result_df.index[-1] - result_df.index[0]).days / 365.25
cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1 / total_years) - 1
downside_returns = returns[returns < risk_free_rate]
downside_std = downside_returns.std() * np.sqrt(12)
sortino = (returns.mean() - risk_free_rate) / downside_std if downside_std > 0 else np.nan

# --- Display Metrics ---
st.subheader("Strategy Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
col2.metric("Sortino Ratio", f"{sortino:.2f}")
col3.metric("CAGR", f"{cagr:.2%}")
st.metric("Annualized Volatility", f"{volatility:.2%}")
st.metric("Max Drawdown", f"{max_drawdown:.2%}")

st.subheader("Last 5 Rebalances")
st.dataframe(pd.DataFrame(rebalance_log).tail())

# --- Charts ---
st.subheader("📈 Strategy vs SPY")
st.line_chart(result_df[['Portfolio Value', 'SPY Value']])

st.subheader("📊 Stock Selection Heatmap")
fig, ax = plt.subplots(figsize=(14, 8))
heatmap_data = selection_matrix.loc[rebalance_dates[:len(portfolio_values)]].T
heatmap_data.columns = heatmap_data.columns.strftime('%Y-%m')
sns.heatmap(heatmap_data, cmap='YlGnBu', cbar=True, ax=ax)
st.pyplot(fig)

# --- Rolling Sharpe Ratio ---
st.subheader("📉 Rolling 12-Month Sharpe Ratio")
rolling_sharpe = (
    (returns.rolling(window=12).mean() - risk_free_rate) /
    returns.rolling(window=12).std()
) * np.sqrt(12)
st.line_chart(rolling_sharpe)

