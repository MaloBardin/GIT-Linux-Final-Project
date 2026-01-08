from VVE import runEveryDay
from mailsending import callforlinux
import pandas as pd
import yfinance as yf

def GetQuickDf():

    cac40 = [
        '^FCHI','AI.PA', 'AIR.PA', 'ALO.PA', 'BN.PA', 'BNP.PA', 'CA.PA',
        'CAP.PA', 'CS.PA', 'DG.PA', 'DSY.PA', 'EL.PA', 'EN.PA', 'ENGI.PA',
        'ERF.PA', 'GLE.PA', 'HO.PA', 'KER.PA', 'LR.PA', 'MC.PA', 'ML.PA',
        'OR.PA', 'ORA.PA', 'PUB.PA', 'RCO.PA', 'RI.PA', 'RMS.PA', 'SAF.PA',
        'SAN.PA', 'SGO.PA', 'STMPA.PA', 'SU.PA', 'TEP.PA',
        'TTE.PA', 'VIE.PA', 'VIV.PA', 'WLN.PA'
    ]

    df = yf.download(cac40, period='1d', interval='1d')['Close']
    df = df.reset_index()
    df.columns = df.columns.str.replace('^FCHI', 'Cac40')

    autres_colonnes = [col for col in df.columns if col not in ['Date', 'Cac40']]
    df = df[['Date', 'Cac40'] + autres_colonnes]

    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"].dt.strftime('%Y-%m-%d')
    return df


def LinuxEveryDayAtNight():
    runEveryDay()
    callforlinux()


def LinuxRunEveryFiveMin():

    quickdf=GetQuickDf()
    histdf=pd.read_csv("cac40_history.csv")
    histdf["Date"] = pd.to_datetime(histdf["Date"])
    histdf["Date"].dt.strftime('%Y-%m-%d')
    if quickdf["Date"].iloc[-1]==histdf["Date"].iloc[-1]:
        print("not added no new data")
        pass
    else :
        print("added new data")
        histdf = pd.concat([histdf, quickdf])
        histdf["Date"] = pd.to_datetime(histdf["Date"])
        histdf["Date"].dt.strftime('%Y-%m-%d')
        histdf.to_csv("cac40_history.csv",index=False)


