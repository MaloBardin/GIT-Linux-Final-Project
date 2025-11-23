import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from grabbing_dataframe import getInfoperTicker2

st.title("📊 Market Dashboard")
query_params = st.query_params
ticker = query_params.get("ticker", "")

if ticker == "":
    ticker = st.sidebar.text_input("Ticker", value="AAPL")

short_df, sevendays_data, onemonth_data, isoverbuying = getInfoperTicker2(ticker)

period_choice = st.sidebar.radio(
    "Choisir la période",
    options=["Intraday (1m)", "7 jours (1h)", "30 jours (1h)"]
)

# --- Choix du bon df ---
if period_choice == "Intraday (1m)":
    df = short_df
elif period_choice == "7 jours (1h)":
    df = sevendays_data
else:
    df = onemonth_data

# Flatten si MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", yaxis="y2", opacity=0.6))

fig.update_layout(
    title=f"{ticker} — {period_choice}",
    xaxis_title="Date / Time",
    yaxis_title="Close",
    yaxis2=dict(title="Volume", overlaying="y", side="right"),
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

st.write(f"🔍 Overbuying indicator: **{isoverbuying:.2f}%**")
