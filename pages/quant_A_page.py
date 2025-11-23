import streamlit as st
import pandas as pd
import datetime as dt
import plotly.graph_objects as go

from quant_A_utils import (
    get_data,
    long_moving_average,
    double_moving_average,
    performance_metrics
)


st.set_page_config(layout="wide")


# Info
@st.dialog("📌 About the Strategies")
def show_strategy_info():
    st.write("### 📌 Available Strategies")
    st.write("""
    **1️⃣ Long Moving Average (LMA)**  
    ➤ Buy when the price > MA  
    ➤ Sell when price < MA  

    **2️⃣ Double Moving Average (DMA)**  
    ➤ Buy when MA_courte > MA_longue  
    ➤ Sell when MA_courte < MA_longue  

    **Parameters:**  
    • `Risk Free Rate` = Annual risk-free return
    • `Transaction Cost` = Included at each buy/sell
    """)


col_title, col_info = st.columns([100, 5])
with col_title:
    st.markdown("""
        <h1 style='text-align: center; font-size: 50px;'>
            Quant A Strategy Backtest
        </h1>
    """, unsafe_allow_html=True)

with col_info:
    st.write("")
    st.write("")
    if st.button("ℹ️", help="About the Black-Litterman Model"):
        show_strategy_info()



#parameters bar
col_left, col_right = st.columns([3, 1], gap="medium")

with col_right:
    with st.container(border=True):
        st.write("**Strategy parameters**")

        ticker = st.text_input("Ticker", "AAPL")
        start_date = st.date_input("Start", value=dt.date(2020, 1, 1))
        end_date = st.date_input("End", value=dt.date(2025, 1, 1))

        interval = st.selectbox("Interval", ["1h", "1d", "1wk", "1mo"], index=1)

        risk_free_rate = st.slider(
            "Risk Free rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=6.0,
            step=0.1
        ) / 100

        transaction_cost = st.slider(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.1,
            step=0.05
        ) / 100

        strategy_choice = st.selectbox("Strategy", ["long_moving_average", "double_moving_average"])

        if strategy_choice == "long_moving_average":
            w = st.slider(
                "Window (MA period)",
                min_value=5,
                max_value=200,
                value=20
            )
        else:
            w1 = st.slider("MA Courte", 5, 200, value=20)
            w2 = st.slider("MA Longue", 5, 200, value=50)

    with st.container(border=False): # pas utile 
        submitted = st.button(
            "Run backtest",
            type="primary",
            use_container_width=True
        )



#Chart + Metrics
with col_left:
    if submitted or True:
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        df = get_data(ticker, start_date, end_date, interval)

        if df.empty:
            st.error("No data found for this ticker.")
        else:
            if strategy_choice == "long_moving_average":
                df_strategy = long_moving_average(
                    df, freq=interval,
                    risk_free_rate=risk_free_rate,
                    transaction_cost=transaction_cost,
                    window=w
                )
            else:
                df_strategy = double_moving_average(df, w1, w2)



            # Chart

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df_strategy['Asset_only'], mode='lines',
                name='Asset Only', line=dict(color='red')
            ))
            fig.add_trace(go.Scatter(
                y=df_strategy['Strategy'], mode='lines',
                name='Strategy', line=dict(color='green')
            ))
            fig.update_layout(
                title=f'{ticker} — Asset vs Strategy',
                xaxis_title='Time', yaxis_title='Value',
                hovermode='x unified',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Performance Metrics
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame(performance_metrics(df_strategy, risk_free_rate=risk_free_rate), index=[ticker])
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
