import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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
def Dfcleaning(df):
    df=df.fillna(method='ffill')
    df["Datetime"]=pd.to_datetime(df['Datetime'])
    return df

df=Dfcleaning(df)
def GetReturnSinceLookBack(df,lookback_days):
    latest_index=df.index[-1]
    todaysdate=df['Datetime'].dt.date.iloc[latest_index]
    print(todaysdate)
    lookback_date=todaysdate-pd.Timedelta(days=lookback_days)
    print(lookback_date)
    morningtime=pd.Timestamp(f"{lookback_date} 08:00:00+00:00")
    morning_index=df.index[df['Datetime']==morningtime][0]

    return (df.iloc[latest_index,1:]-df.iloc[morning_index,1:])/df.iloc[morning_index,1:]*100


returns_today=GetReturnSinceLookBack(df,0)
returns_7days=GetReturnSinceLookBack(df,7)
returns_30days=GetReturnSinceLookBack(df,30)

print("Returns today:")
print(returns_today)
print("Returns last 7 days:")
print(returns_7days)
print("Returns last 30 days:")
print(returns_30days)