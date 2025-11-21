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
    df = df.rename(columns=name_map)

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



def GetDfForDashboard(df):
    df_dashboard=pd.DataFrame()
    df_dashboard["Ticker"]=df.columns[1:]
    df_dashboard["Price"]=df.iloc[-1,1:].values
    df_dashboard["Return_1d"]=GetReturnSinceLookBack(df,1).values
    df_dashboard["Return_7d"]=GetReturnSinceLookBack(df,7).values
    df_dashboard["Return_30d"]=GetReturnSinceLookBack(df,30).values
    return df_dashboard

df_dash=GetDfForDashboard(Dfcleaning(ReadDf()))
print(df_dash)



def getInfoperTicker(ticker):
    longtimedata=df[ticker]
    print(longtimedata)


