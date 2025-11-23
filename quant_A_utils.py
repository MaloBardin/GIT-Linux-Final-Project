import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def get_data(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval)
    #df = yf.download("BNP.PA", period='3y', interval='1d')
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1)
    return df[['Close']].copy()

def risk_free_rate_conversion(rate, freq):
    if freq == "1d":
        rf_adjusted = rate / 252
    elif freq == "1wk":
        rf_adjusted = rate / 52
    elif freq == "1m":
        rf_adjusted = rate / 12
    elif freq == "1h":
        rf_adjusted = rate / (252 * 6.5)
    else:
        rf_adjusted = 0
    return rf_adjusted


def long_moving_average(df, freq,risk_free_rate = 0.05,transaction_cost=0.01, window=20  ):
    df = df.copy()
    rf_adjusted = risk_free_rate_conversion(risk_free_rate, freq)
    df['Return'] = df['Close'].pct_change()

    df['MA'] = df['Close'].rolling(window).mean()
    df['Position'] = (df['Close'] > df['MA']).astype(int)

    df['Trade'] = df['Position'].diff().abs()

    df['Cost'] = df['Trade'] * transaction_cost

    df['Strategy_Return'] = df['Position'] * (df['Return'] + rf_adjusted) - df['Cost']

    df['Strategy'] = (1 + df['Strategy_Return']).cumprod() - 1

    df['Asset_only'] = df['Close'].pct_change().add(1).cumprod() - 1

    df = df.dropna().copy()
    df['Strategy'] = (df['Strategy'] - df['Strategy'].iloc[0])
    df['Asset_only'] = df['Asset_only'] - df['Asset_only'].iloc[0]


    return df.dropna()


def double_moving_average(df, w1=20, w2 = 50):
    # We buy when the moving average 1 is above the moving average 2 and hold otherwise
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA_short'] = df['Close'].rolling(w1).mean()
    df['MA_long'] = df['Close'].rolling(w2).mean()
    df['Position'] = (df['MA_short'] > df['MA_long']).astype(int)

    df['Strategy_Return'] = df['Position'] * df['Return']
    df['Strategy'] = (1 + df['Strategy_Return']).cumprod() - 1

    df['Asset_only'] = df['Close'].pct_change().add(1).cumprod() - 1

    df = df.dropna().copy()
    df['Strategy'] = (df['Strategy'] - df['Strategy'].iloc[0])
    df['Asset_only'] = df['Asset_only'] - df['Asset_only'].iloc[0]
    return df.dropna()


def performance_metrics(df, risk_free_rate=0.0, periods_per_year=252):
    df = df.copy()

    volatility = df['Strategy_Return'].std() * np.sqrt(periods_per_year)

    sharpe = (df['Strategy_Return'].mean() * periods_per_year - risk_free_rate) / volatility

    downside_returns = df.loc[df['Strategy_Return'] < 0, 'Strategy_Return']
    downside_risk = downside_returns.std() * np.sqrt(periods_per_year)
    sortino = (df['Strategy_Return'].mean() * periods_per_year - risk_free_rate) / downside_risk

    df['Rolling_Max'] = df['Strategy'].cummax()
    drawdown = df['Strategy'] / df['Rolling_Max'] - 1
    max_drawdown = drawdown.min()
    n_years = len(df) / periods_per_year
    CAGR = (df['Strategy'].iloc[-1]) ** (1 / n_years) - 1

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




if __name__ == "__main__":
    TICKER = "GC=F"
    START = "2020-01-01"
    END = "2025-01-01"
    INTERVAL = "1d"
    WINDOW = 100
    data = get_data(TICKER, START, END, INTERVAL)


    ret = long_moving_average(data, WINDOW)

    plt.figure(figsize=(10, 5))

    plt.plot(ret['Asset_only'], label='Price (normalized)')
    plt.plot(ret['Strategy'], label='Cumulative Strategy')




    plt.title("Strategie Mean Return - Points d'achat")
    plt.legend()
    plt.show()

    metrics = performance_metrics(ret)
    print(metrics)


    # 3. Affichage des Métriques (Console)
    #sharpe_h, dd_h = calculate_metrics(results['Strat_Hold'])
    #sharpe_m, dd_m = calculate_metrics(results['Strat_Momentum'])


    print(f"MÉTRIQUES ({START} à aujourd'hui)")

    #print(f"Buy & Hold -> Sharpe: {sharpe_h:.2f} | Max Drawdown: {dd_h:.2%}")
    #print(f"Momentum   -> Sharpe: {sharpe_m:.2f} | Max Drawdown: {dd_m:.2%}")

