import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA


def risk_free_rate_conversion(rate, freq):
    if freq == "1d":
        rf_adjusted = rate / 252
    elif freq == "1wk":
        rf_adjusted = rate / 52
    elif freq == "1m":
        rf_adjusted = rate / 12
    elif freq == "1h":
        rf_adjusted = rate / (252 * 6.5)
    elif freq == "1min":
        rf_adjusted = rate / (252 * 6.5 * 60)
    else:
        rf_adjusted = 0
    return rf_adjusted

def calculate_strategy_returns(df, risk_free_rate, freq, transaction_cost):
    rf_adjusted = risk_free_rate_conversion(risk_free_rate, freq)
    df['Return'] = df['Close'].pct_change().replace(np.nan, 0)
    
    df['Trade'] = df['Position'].diff().abs().replace(np.nan, 0.0)
    df['Cost'] = df['Trade'] * transaction_cost
    
    df['Strategy_Return'] = (df['Position'] * df['Return'] + (1 - df['Position']) * rf_adjusted - df["Cost"]).replace(np.nan, 0)
    
    df["Strategy"] = ((1 + df['Strategy_Return']).cumprod() - 1) * 100
    df["Asset_only"] = ((1 + df['Return']).cumprod() - 1) * 100
    return df.dropna()

def long_moving_average(df, freq,risk_free_rate = 0.05,transaction_cost=0.01, window=20  ):
    df = df.copy()
    rf_adjusted = risk_free_rate_conversion(risk_free_rate, freq)
    df['Return'] = df['Close'].pct_change().replace(np.nan, 0)

    df['MA'] = df['Close'].rolling(window).mean()
    df["MA"].replace(np.nan,np.inf, inplace=True)
    df['Position'] = (df['Close'].shift(1) > df['MA'].shift(1)).astype(int)
    df['Trade'] = df['Position'].diff().abs().replace(np.nan, 0.0)

    df['Cost'] = df['Trade'] * transaction_cost

    df['Strategy_Return'] = (df['Position'] * df['Return'] + (1 - df['Position']) * rf_adjusted - df["Cost"]).replace(np.nan, 0)


    df["Strategy"] = ((1 + df['Strategy_Return']).cumprod() - 1) * 100
    df["Asset_only"] = ((1 + df['Return']).cumprod() - 1) * 100


    return df.dropna()


def double_moving_average(df,freq,risk_free_rate = 0.05,transaction_cost=0.01, w1=20, w2 = 50):
    # We buy when the moving average 1 is above the moving average 2 and hold otherwise
    df = df.copy()
    rf_adjusted = risk_free_rate_conversion(risk_free_rate, freq)
    df['Return'] = df['Close'].pct_change()
    df['MA_short'] = df['Close'].rolling(w1).mean()
    df['MA_long'] = df['Close'].rolling(w2).mean()
    df['Position'] = (df['MA_short'].shift(1) > df['MA_long'].shift(1)).astype(int)

    df['Trade'] = df['Position'].diff().abs().replace(np.nan, 0.0)

    df['Cost'] = df['Trade'] * transaction_cost

    df['Strategy_Return'] = (df['Position'] * df['Return'] + (1 - df['Position']) * rf_adjusted - df["Cost"]).replace(np.nan, 0)

    df["Strategy"] = ((1 + df['Strategy_Return']).cumprod() - 1) * 100
    df["Asset_only"] = ((1 + df['Return']).cumprod() - 1) * 100
    return df.dropna()

def bollinger_bands_strategy(df, freq, risk_free_rate=0.05, transaction_cost=0.01, window=20, num_std=2):
    df = df.copy()
    df['MA'] = df['Close'].rolling(window).mean()
    df['STD'] = df['Close'].rolling(window).std()
    df['Upper'] = df['MA'] + (df['STD'] * num_std)
    df['Lower'] = df['MA'] - (df['STD'] * num_std)
    
    df['Signal'] = 0
    df.loc[df['Close'] < df['Lower'], 'Signal'] = 1 
    df.loc[df['Close'] > df['Upper'], 'Signal'] = -1
    
    df['Position'] = df['Signal'].replace(0, np.nan).ffill().replace(-1, 0).shift(1).fillna(0)
    
    return calculate_strategy_returns(df, risk_free_rate, freq, transaction_cost)

def rsi_strategy(df, freq, risk_free_rate=0.05, transaction_cost=0.01, window=14, lower=30, upper=70):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Signal'] = 0
    df.loc[df['RSI'] < lower, 'Signal'] = 1
    df.loc[df['RSI'] > upper, 'Signal'] = -1
    
    df['Position'] = df['Signal'].replace(0, np.nan).ffill().replace(-1, 0).shift(1).fillna(0)
    
    return calculate_strategy_returns(df, risk_free_rate, freq, transaction_cost)

def macd_strategy(df, freq, risk_free_rate=0.05, transaction_cost=0.01, slow=26, fast=12, signal=9):
    df = df.copy()
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    
    df['Position'] = (df['MACD'].shift(1) > df['Signal_Line'].shift(1)).astype(int)
    
    return calculate_strategy_returns(df, risk_free_rate, freq, transaction_cost)


def performance_metrics(df, risk_free_rate=0.0, periods_per_year=252):
    df = df.copy()

    volatility = df['Strategy_Return'].std() * np.sqrt(periods_per_year)

    sharpe = (df['Strategy_Return'].mean() * periods_per_year - risk_free_rate) / volatility

    downside_returns = df.loc[df['Strategy_Return'] < 0, 'Strategy_Return']
    downside_risk = downside_returns.std() * np.sqrt(periods_per_year)
    sortino = (df['Strategy_Return'].mean() * periods_per_year - risk_free_rate) / downside_risk

    equity_curve = (1 + df['Strategy_Return']).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_drawdown = drawdown.min()

    n_years = len(df) / periods_per_year
    CAGR = (equity_curve.iloc[-1]) ** (1 / n_years) - 1

    win_rate = (df['Strategy_Return'] > 0).mean()

    trades = (df['Position'].diff().abs() > 0).sum()

    return {
        'CAGR': CAGR,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Number of Trades': trades
    }

def predict_prophet(df, periods=30, interval="1d"):
    df_prophet = df.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

    m = Prophet()
    m.fit(df_prophet)

    if interval == "1min":
        # prédiction en minutes
        future = m.make_future_dataframe(
            periods=periods,
            freq="min",
            include_history=False
        )
    else:
        # prédiction en jours
        future = m.make_future_dataframe(
            periods=periods,
            freq="D",
            include_history=False
        )

    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def predict_arima(df, periods=30, interval="1d"):
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()

    forecast_result = model_fit.get_forecast(steps=periods)
    forecast_df = forecast_result.summary_frame()

    last_date = df.index[-1]

    if interval == "1min":
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(minutes=1),
            periods=periods,
            freq="min"
        )
    else:
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq="D"
        )

    return pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_df['mean'].values,
        'yhat_lower': forecast_df['mean_ci_lower'].values,
        'yhat_upper': forecast_df['mean_ci_upper'].values
    })



