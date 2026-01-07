import streamlit as st
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
import json
import os


from quant_A_utils import (
    get_data,
    long_moving_average,
    double_moving_average,
    bollinger_bands_strategy,
    rsi_strategy,
    macd_strategy,
    predict_future_only,
    performance_metrics,
    predict_arima
)

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css("style.css")

def load_tickers(filename="tickers.json"):
    if not os.path.exists(filename):
        return {"AAPL": "Apple Inc.", "TSLA": "Tesla", "GC=F": "Gold", "^FCHI": "CAC 40"}
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

st.set_page_config(layout="wide")

#Header
with st.container(border=True):
    col_title, col_info = st.columns([100, 5])
    with col_title:
        st.markdown("""
            <h1 style='text-align: center; font-size: 30px; margin-bottom: 0px;'>
                Quant A : Single Asset Management
            </h1>
        """, unsafe_allow_html=True)

# Layout Principal
col_left, col_right = st.columns([3, 1], gap="medium")


with col_right:
    w1 = 20
    w2 = 50
    with st.container(border=True):
        st.write("### ⚙️ Settings")

        ticker_map = load_tickers()
        display_options = {f"{name} ({ticker})": ticker for ticker, name in ticker_map.items()}
        
        selected_label = st.selectbox("Select Asset", options=list(display_options.keys()))
        ticker = display_options[selected_label]
        start_date = st.date_input("Start", value=dt.date(2023, 1, 1))
        end_date = st.date_input("End", value=dt.date(2025, 1, 1))

        interval = st.selectbox("Interval", ["1h", "1d", "1wk", "1mo"], index=1)

        risk_free_rate = st.slider("Risk Free rate (%)", 0.0, 15.0, 6.0, 0.1) / 100
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 5.0, 0.1, 0.05) / 100

        strategy_choice = st.selectbox(
            "Strategy", 
            ["Long Moving Average", "Double Moving Average", "Bollinger Bands", "RSI", "MACD"]
        )

        if strategy_choice == "Long Moving Average":
            w = st.slider("Window", 5, 200, 20)
        elif strategy_choice == "Double Moving Average":
            w1 = st.slider("Short MA", 5, 200, 20)
            w2 = st.slider("Long MA", 5, 200, 50)
        elif strategy_choice == "Bollinger Bands":
            bb_window = st.slider("Window", 5, 100, 20)
            bb_std = st.slider("Std Dev", 1.0, 4.0, 2.0, 0.1)
        elif strategy_choice == "RSI":
            rsi_window = st.slider("Window", 5, 50, 14)
            rsi_lower = st.slider("Lower Threshold (Buy)", 10, 45, 30)
            rsi_upper = st.slider("Upper Threshold (Sell)", 55, 90, 70)
        elif strategy_choice == "MACD":
            macd_fast = st.slider("Fast EMA", 2, 50, 12)
            macd_slow = st.slider("Slow EMA", 10, 100, 26)
            macd_signal = st.slider("Signal EMA", 2, 50, 9)


with col_left:
    with st.container(border=True):
        if True:
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")

            df = get_data(ticker, start_date_str, end_date_str, interval)

            if df.empty:
                st.error("No data found for this ticker.")
            else:
                if strategy_choice == "Long Moving Average":
                    df_strategy = long_moving_average(df, interval, risk_free_rate, transaction_cost, w)
                elif strategy_choice == "Double Moving Average":
                    df_strategy = double_moving_average(df, interval, risk_free_rate, transaction_cost, w1, w2)
                elif strategy_choice == "Bollinger Bands":
                    df_strategy = bollinger_bands_strategy(df, interval, risk_free_rate, transaction_cost, bb_window, bb_std)
                elif strategy_choice == "RSI":
                    df_strategy = rsi_strategy(df, interval, risk_free_rate, transaction_cost, rsi_window, rsi_lower, rsi_upper)
                elif strategy_choice == "MACD":
                    df_strategy = macd_strategy(df, interval, risk_free_rate, transaction_cost, macd_slow, macd_fast, macd_signal)

                metrics = performance_metrics(df_strategy, risk_free_rate=risk_free_rate)
                last_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                price_change = last_price - prev_price
                pct_change = (price_change / prev_price) * 100

                color = "#2EBD85" if price_change >= 0 else "#F6465D"
                arrow = "↑" if price_change >= 0 else "↓"

                st.markdown(f"<h3 style='margin-bottom: 10px;'>{selected_label}</h3>", unsafe_allow_html=True)

                # Price Display
                with st.container(border = True):
                    st.markdown(f"""

                        <span style="font-size:30px; font-weight: bold; color: white;">
                            ${last_price:,.2f}
                        </span>
                        <span style="color: {color}; font-size: 25px; margin-left: 10px; font-weight: 500;">
                            {arrow} {price_change:+.2f} ({pct_change:+.2f}%)
                        </span>
                        <br>
                        <span style="color: #787B86; font-size: 15px; text-transform: uppercase;">
                            Current Price
                        </span>

                    """, unsafe_allow_html=True)

                # Chart
                with st.container(border = True):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_strategy.index, y=df_strategy['Asset_only'], 
                        mode='lines', name='Asset Only', line=dict(color='#FF5252')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_strategy.index, y=df_strategy['Strategy'], 
                        mode='lines', name='Strategy', line=dict(color='#00E676')
                    ))
                    fig.update_layout(
                        title=f'{ticker} — Asset vs Strategy',
                        xaxis_title='Time', yaxis_title='Return (%)',
                        hovermode='x unified', template='plotly_dark',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Metrics
                with st.container(border=True):
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
            
                    c1.metric("Volatility", f"{metrics['Volatility']:.2%}")
                    c2.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                    c3.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
                    c4.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                    c5.metric("Total Trades", f"{metrics['Number of Trades']}")
                    c6.metric("Win Rate", f"{metrics['Win Rate']:.2%}")




st.subheader("Price Prediction")

col_pred_left, col_pred_right = st.columns([1, 4], gap="medium")

with col_pred_left:
    with st.container(border=True):
        st.write("**Configuration**")
        pred_model = st.selectbox("Model", ["Prophet", "ARIMA"])
        pred_days = st.slider("Forecast Period (days)", 7, 365, 45)


with col_pred_right:
    with st.container(border=True):
        with st.spinner("Generating prediction..."):
            if pred_model == "Prophet":
                forecast = predict_future_only(df, days=pred_days)
            else:
                forecast = predict_arima(df, days=pred_days)

            last_hist_date = df.index[-1]
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=df.index, y=df['Close'], mode='lines', 
                name='Historical', line=dict(color='#2962FF', width=2)
            ))
            fig_pred.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'], mode='lines', 
                name=f'{pred_model} Prediction', line=dict(color='#00E676', width=2, dash='dot')
            ))
            fig_pred.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_upper'], mode='lines',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig_pred.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
                line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 230, 118, 0.2)',
                showlegend=False, name='Confidence Interval'
            ))
            fig_pred.add_vline(x=last_hist_date, line_width=1, line_dash="dash", line_color="white")
            
            fig_pred.update_layout(
                title=f"Forecast ({pred_days} days)",
                xaxis_title="Date", yaxis_title="Price", 
                template="plotly_dark", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            st.plotly_chart(fig_pred, use_container_width=True)


