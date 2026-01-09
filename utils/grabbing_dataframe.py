import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import os
import json
import traceback

FILE_NAME = "cac40_history.csv"

DATA_DIR = "data"
FILE_MAX = os.path.join(DATA_DIR, "cac40_history.csv")
FILE_3Y = os.path.join(DATA_DIR, "data3y.csv")
FILE_7D = os.path.join(DATA_DIR, "data7d.csv")

TICKERJSON = os.path.join(DATA_DIR, "tickers.json")
try:
    with open(TICKERJSON, "r", encoding="utf-8") as f:
        name_map = json.load(f)
except Exception as e:
    st.error(f"Failed to load ticker mapping: {e}")
    traceback.print_exc()
    name_map = {}

cac40 = [
    '^FCHI', 'AI.PA', 'AIR.PA', 'ALO.PA', 'BN.PA', 'BNP.PA', 'CA.PA',
    'CAP.PA', 'CS.PA', 'DG.PA', 'DSY.PA', 'EL.PA', 'EN.PA', 'ENGI.PA',
    'ERF.PA', 'GLE.PA', 'HO.PA', 'KER.PA', 'LR.PA', 'MC.PA', 'ML.PA',
    'OR.PA', 'ORA.PA', 'PUB.PA', 'RCO.PA', 'RI.PA', 'RMS.PA', 'SAF.PA',
    'SAN.PA', 'SGO.PA', 'STMPA.PA', 'SU.PA', 'TEP.PA',
    'TTE.PA', 'VIE.PA', 'VIV.PA', 'WLN.PA'
]

@st.cache_data
def ReadDfMax():
    try:
        df = pd.read_csv(FILE_MAX)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Failed to read max daily data: {e}")
        traceback.print_exc()
        return pd.DataFrame()

@st.cache_data
def ReadDf1Min():
    try:
        df = pd.read_csv(FILE_7D)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Failed to read 1-minute data: {e}")
        traceback.print_exc()
        return pd.DataFrame()

@st.cache_data
def ReadDf():
    try:
        df = pd.read_csv(FILE_3Y)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Failed to read 3-year data: {e}")
        traceback.print_exc()
        return pd.DataFrame()

@st.cache_data
def Dfcleaning(df):
    try:
        df = df.ffill()
        df["Date"] = pd.to_datetime(df['Date'])
        df = df.rename(columns=name_map)
        return df
    except Exception as e:
        st.error(f"Failed to clean dataframe: {e}")
        traceback.print_exc()
        return df

@st.cache_data
def Dfclean(df):
    try:
        df = df.ffill()
        df["Date"] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Failed to forward fill dataframe: {e}")
        traceback.print_exc()
        return df

def GetReturnSinceLookBack(df, lookback_days):
    try:
        latest_index = df.index[-1]
        todaysdate = df["Date"].iloc[latest_index]
        lookback_date = todaysdate - pd.Timedelta(days=lookback_days)
        mask = df["Date"] >= lookback_date
        morning_index = df.index[mask][0] if mask.any() else df.index[0]
        return (df.iloc[latest_index, 1:] - df.iloc[morning_index, 1:]) / df.iloc[morning_index, 1:] * 100
    except Exception as e:
        st.error(f"Failed to calculate returns: {e}")
        traceback.print_exc()
        return pd.Series(dtype=float)

@st.cache_data
def GetDfForDashboard(df):
    try:
        df_dashboard = pd.DataFrame()
        df_dashboard["Ticker"] = df.columns[1:]
        df_dashboard["Price"] = df.iloc[-1, 1:].values
        df_dashboard["Return_1d"] = GetReturnSinceLookBack(df, 1).values
        df_dashboard["Return_7d"] = GetReturnSinceLookBack(df, 7).values
        df_dashboard["Return_30d"] = GetReturnSinceLookBack(df, 30).values
        return df_dashboard
    except Exception as e:
        st.error(f"Failed to generate dashboard dataframe: {e}")
        traceback.print_exc()
        return pd.DataFrame()

df_dash = GetDfForDashboard(Dfcleaning(ReadDf()))

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def getInfoperTicker2(ticker):
    try:
        key_ticker = [a for a, b in name_map.items() if b == ticker]
        if not key_ticker:
            st.warning(f"No symbol found for ticker {ticker}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        symbol = key_ticker[0]

        intraday_data = yf.download(symbol, period='1d', interval="1m")[['Close', 'Volume']]
        onemonth_data = yf.download(symbol, period='30d', interval="1h")[['Close', 'Volume']]
        fiveyear_data = yf.download(symbol, period='5y', interval="1wk")[['Close', 'Volume']]

        return flatten_columns(intraday_data), flatten_columns(onemonth_data), flatten_columns(fiveyear_data)
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def getDfForGraph(df, dayforgraphlookback=30):
    try:
        graph_df = pd.DataFrame()
        latest_index = df.index[-1]
        todaysdate = df['Date'].iloc[latest_index]
        lookback_date = todaysdate - pd.Timedelta(days=dayforgraphlookback)
        mask = df['Date'] >= lookback_date
        morning_index = df.index[mask][0] if mask.any() else df.index[0]
        for ticker in df.columns[1:]:
            graph_df[ticker] = df[ticker]
        graph_df = graph_df[morning_index:-1:1]
        return graph_df
    except Exception as e:
        st.error(f"Failed to prepare graph dataframe: {e}")
        traceback.print_exc()
        return pd.DataFrame()

@st.cache_data
def get_data(ticker, start, end, interval='1d'):
    try:
        if interval == '1min':
            df = ReadDf1Min()
        else:
            df = ReadDfMax()

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df[(df.index >= start) & (df.index < end)]
        df = df[[ticker]].copy()
        df.columns = ['Close']
        return df
    except Exception as e:
        st.error(f"Failed to get data for {ticker}: {e}")
        traceback.print_exc()
        return pd.DataFrame()
