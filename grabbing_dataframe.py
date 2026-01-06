import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def GetDf():
    # tickers cac40
    cac40 = [
        'AI.PA', 'AIR.PA', 'ALO.PA', 'ATO.PA', 'BN.PA', 'BNP.PA', 'CA.PA',
        'CAP.PA', 'CS.PA', 'DG.PA', 'DSY.PA', 'EL.PA', 'EN.PA', 'ENGI.PA',
        'ERF.PA', 'GLE.PA', 'HO.PA', 'KER.PA', 'LR.PA', 'MC.PA', 'ML.PA',
        'OR.PA', 'ORA.PA', 'PUB.PA', 'RCO.PA', 'RI.PA', 'RMS.PA', 'SAF.PA',
        'SAN.PA', 'SGO.PA', 'STMPA.PA', 'SU.PA', 'TEP.PA',
        'TTE.PA', 'URW.PA', 'VIE.PA', 'VIV.PA', 'WLN.PA'
    ]

    df = yf.download(cac40, period='3y', interval='1d')[['Close', 'Volume']]




def ReadDf():
    df = pd.read_csv('cac40_data.csv')
    return df


df=ReadDf()

name_map = {
    'AI.PA': 'Air Liquide',
    'AIR.PA': 'Airbus',
    'ALO.PA': 'Alstom',
    'ATO.PA': 'Atos',
    'BN.PA': 'Danone',
    'BNP.PA': 'BNP Paribas',
    'CA.PA': 'Crédit Agricole',
    'CAP.PA': 'Capgemini',
    'CS.PA': 'AXA',
    'DG.PA': 'Vinci',
    'DSY.PA': 'Dassault Systèmes',
    'EL.PA': 'EssilorLuxottica',
    'EN.PA': 'Bouygues',
    'ENGI.PA': 'Engie',
    'ERF.PA': 'Eurofins Scientific',
    'GLE.PA': 'Société Générale',
    'HO.PA': 'Thales',
    'KER.PA': 'Kering',
    'LR.PA': 'Legrand',
    'MC.PA': 'LVMH',
    'ML.PA': 'Michelin',
    'OR.PA': "L'Oréal",
    'ORA.PA': 'Orange',
    'PUB.PA': 'Publicis',
    'RCO.PA': 'Rémy Cointreau',
    'RI.PA': 'Pernod Ricard',
    'RMS.PA': 'Hermès',
    'SAF.PA': 'Safran',
    'SAN.PA': 'Sanofi',
    'SGO.PA': 'Saint-Gobain',
    'STMPA.PA': 'STMicroelectronics',
    'SU.PA': 'Schneider Electric',
    'TEP.PA': 'Téléperformance',
    'TTE.PA': 'TotalEnergies',
    'URW.PA': 'UR-Westfield',
    'VIE.PA': 'Veolia',
    'VIV.PA': 'Vivendi',
    'WLN.PA': 'Worldline'
}


def Dfcleaning(df):
    df=df.fillna(method='ffill')
    df["Datetime"]=pd.to_datetime(df['Datetime'])

    df = df.rename(columns=name_map)

    return df


df=Dfcleaning(df)


def GetReturnSinceLookBack(df,lookback_days):
    latest_index=df.index[-1]
    todaysdate=df['Datetime'].dt.date.iloc[latest_index]
    lookback_date=todaysdate-pd.Timedelta(days=lookback_days)
    morningtime=pd.Timestamp(f"{lookback_date} 08:00:00+00:00")
    morning_index=df.index[df['Datetime']==morningtime][0]

    return (df.iloc[latest_index,1:]-df.iloc[morning_index,1:])/df.iloc[morning_index,1:]*100




def GetDfForDashboard(df):
    df_dashboard=pd.DataFrame()
    df_dashboard["Ticker"]=df.columns[1:]
    df_dashboard["Price"]=df.iloc[-1,1:].values
    df_dashboard["Return_1d"]=GetReturnSinceLookBack(df,1).values
    df_dashboard["Return_7d"]=GetReturnSinceLookBack(df,7).values
    df_dashboard["Return_30d"]=GetReturnSinceLookBack(df,30).values
    return df_dashboard

df_dash=GetDfForDashboard(Dfcleaning(ReadDf()))




def getInfoperTicker(df,ticker):
    longtimedata=df[["Datetime",ticker]]
    latestdate=df['Datetime'].dt.date.iloc[df.index[-1]]

    #request on yfinance to get the 5min data for today
    key_ticker=[ a for a, b in name_map.items() if b == ticker]

    intraday_data=yf.download(key_ticker[0], period='1d', interval="1m")[['Close', 'Volume']]
    intraday_data=intraday_data.reset_index()

    time_list=[]
    price_list=[]
    volume_list=[]


    timespan = pd.date_range(start="08:00",end="16:29",freq='1min')
    time_list = timespan.strftime('%H:%M').tolist()

    for i in range(len(time_list)):
        volume_list.append(np.nan)
        price_list.append(np.nan) # a fix for later decalage issue

    volume_list[0]=(intraday_data.iloc[0,2]) #first vol

    for i in range(1,len(intraday_data)):
        volume_list[i]=volume_list[i-1]+intraday_data.iloc[i,2]
        price_list[i]=intraday_data.iloc[i,1]



    for i in range(len(intraday_data)):

        price_list.append(intraday_data.iloc[i,1])
    short_df=pd.DataFrame()
    short_df["Time"]=time_list
    short_df["Volume"]=volume_list


    sevendays_data=yf.download(key_ticker[0], period='7d', interval="1h")[['Close', 'Volume']]
    onemonth_data=yf.download(key_ticker[0], period='30d', interval="1h")[['Close', 'Volume']]

    isoverbuying=(100*intraday_data.iloc[-1,1])/onemonth_data['Close'].mean()-100




    return short_df, sevendays_data, onemonth_data, isoverbuying

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def getInfoperTicker2(df,ticker):
    longtimedata=df[["Datetime",ticker]]
    latestdate=df['Datetime'].dt.date.iloc[df.index[-1]]

    #request on yfinance to get the 5min data for today
    key_ticker=[ a for a, b in name_map.items() if b == ticker]

    short_df=yf.download(key_ticker[0], period='1d', interval="1m")[['Close', 'Volume']]
    sevendays_data=yf.download(key_ticker[0], period='7d', interval="1h")[['Close', 'Volume']]
    onemonth_data=yf.download(key_ticker[0], period='30d', interval="1h")[['Close', 'Volume']]

    isoverbuying=(100*short_df.iloc[-1,1])/onemonth_data['Close'].mean()-100

    return flatten_columns(short_df), flatten_columns(sevendays_data), flatten_columns(onemonth_data), isoverbuying


def getInfoperTicker2(df,ticker):
    longtimedata=df[["Datetime",ticker]]
    latestdate=df['Datetime'].dt.date.iloc[df.index[-1]]

    #request on yfinance to get the 5min data for today
    key_ticker=[ a for a, b in name_map.items() if b == ticker]
    symbol = key_ticker[0]

    intraday_data=yf.download(symbol, period='1d', interval="1m")[['Close', 'Volume']]
    sevendays_data=yf.download(symbol, period='7d', interval="1h")[['Close', 'Volume']]
    onemonth_data=yf.download(symbol, period='30d', interval="1h")[['Close', 'Volume']]
    oneyear_data=yf.download(symbol, period='1y', interval="1d")[['Close', 'Volume']]
    fiveyear_data = yf.download(symbol, period='5y', interval="1wk")[['Close', 'Volume']]

    return intraday_data, sevendays_data, onemonth_data, oneyear_data, fiveyear_data



short_df,sevendays_data,onemonth_data,isoverbuying=getInfoperTicker(df,'AXA')

print(short_df)
print(sevendays_data)
print(onemonth_data)
print(isoverbuying)
