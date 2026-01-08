
import yfinance as yf
import pandas as pd
import os
from datetime import datetime


DATA_DIR = "data"
FILE_MAX = os.path.join(DATA_DIR, "cac40_history.csv")
FILE_3Y = os.path.join(DATA_DIR, "data3y.csv")

os.makedirs(DATA_DIR, exist_ok=True)

CAC40_MAX = [
    '^FCHI', 'AI.PA', 'AIR.PA', 'ALO.PA', 'BN.PA', 'BNP.PA', 'CA.PA',
    'CAP.PA', 'CS.PA', 'DG.PA', 'DSY.PA', 'EL.PA', 'EN.PA', 'ENGI.PA',
    'ERF.PA', 'GLE.PA', 'HO.PA', 'KER.PA', 'LR.PA', 'MC.PA', 'ML.PA',
    'OR.PA', 'ORA.PA', 'PUB.PA', 'RCO.PA', 'RI.PA', 'RMS.PA', 'SAF.PA',
    'SAN.PA', 'SGO.PA', 'STMPA.PA', 'SU.PA', 'TEP.PA',
    'TTE.PA', 'VIE.PA', 'VIV.PA', 'WLN.PA'
]

CAC40_3Y = [
    'AI.PA', 'AIR.PA', 'ALO.PA', 'ATO.PA', 'BN.PA', 'BNP.PA', 'CA.PA',
    'CAP.PA', 'CS.PA', 'DG.PA', 'DSY.PA', 'EL.PA', 'EN.PA', 'ENGI.PA',
    'ERF.PA', 'GLE.PA', 'HO.PA', 'KER.PA', 'LR.PA', 'MC.PA', 'ML.PA',
    'OR.PA', 'ORA.PA', 'PUB.PA', 'RCO.PA', 'RI.PA', 'RMS.PA', 'SAF.PA',
    'SAN.PA', 'SGO.PA', 'STMPA.PA', 'SU.PA', 'TEP.PA',
    'TTE.PA', 'URW.PA', 'VIE.PA', 'VIV.PA', 'WLN.PA'
]

def update_df_max():
    print("Downloading full history...")
    df = yf.download(CAC40_MAX, period='max', interval='1d')['Close']
    df = df.reset_index()
    df.rename(columns={'^FCHI': 'Cac40'}, inplace=True)
    cols = [c for c in df.columns if c not in ['Date', 'Cac40']]
    df = df[['Date', 'Cac40'] + cols]
    df = df.drop_duplicates(subset=['Date'])
    df.to_csv(FILE_MAX, index=False)
    print(f"Saved {FILE_MAX}")

def update_df_3y():
    print("Downloading last 3 years of data...")
    df = yf.download(CAC40_3Y, period='3y', interval='1d')[['Close', 'Volume']]
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df = df.reset_index()
    df.to_csv(FILE_3Y, index=False)
    print(f"Saved {FILE_3Y}")


def main():
    update_df_max()
    update_df_3y()
    print("All datasets updated successfully!")

if __name__ == "__main__":
    main()
