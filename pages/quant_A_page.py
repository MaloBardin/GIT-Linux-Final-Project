import streamlit as st
import pandas as pd
import datetime as dt

from quant_A_utils import get_data, long_moving_average, double_moving_average, performance_metrics
st.set_page_config(layout="wide")
st.title("Quant A Strategy")
st.sidebar.header("Parameters")


with st.sidebar.form("param_form", clear_on_submit=False):
    ticker = st.text_input("Ticker", "AAPL")
    start_date = st.date_input("Start", value=dt.date(2020, 1, 1))
    end_date = st.date_input("End", value=dt.date(2025, 1, 1))

    interval = st.selectbox("Interval", ["1h", "1d", "1wk", "1mo"], index=1)
    risk_free_rate = st.number_input("Risk Free rate (%)", 0.0, 100.0, value=6.0, step=0.1 ) / 100
    transaction_cost = st.number_input("Transaction Cost (%)", 0.0, 100.0, value=0.1, step=0.1 ) / 100

    strategy_choice = st.selectbox("Strategies", ["long_moving_average", "double_moving_average"])

    if strategy_choice == "long_moving_average":
        w = st.number_input("Window (MA period)", 5, 200, value=20)
    elif strategy_choice == "double_moving_average":
        w1 = st.number_input("MA courte", 5, 200, value=20)
        w2 = st.number_input("MA longue", 5, 200, value=50)

    submitted = st.form_submit_button("Update Backtest")


if submitted or True:
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    df = get_data(ticker, start_date, end_date, interval)

    if df.empty:
        st.error("Erreur : Aucune donnée trouvée.")
    else:
        if strategy_choice == "long_moving_average":
            df_strategy = long_moving_average(df, freq=interval, risk_free_rate=risk_free_rate, transaction_cost=transaction_cost, window=w)
        else:
            df_strategy = double_moving_average(df, w1, w2)

        st.subheader("Performance metrics")
        st.write(pd.DataFrame(performance_metrics(df_strategy), index=[ticker]))

        st.subheader("Graph")
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df_strategy['Asset_only'], mode='lines', name='Asset Only',
                                 line=dict(color='yellow')))
        fig.add_trace(go.Scatter(y=df_strategy['Strategy'], mode='lines', name='Strategy',
                                 line=dict(color='green')))

        fig.update_layout(title='Asset vs Strategy',
                          xaxis_title='Time',
                          yaxis_title='Value',
                          template='plotly_dark')

        st.plotly_chart(fig, use_container_width=True)
