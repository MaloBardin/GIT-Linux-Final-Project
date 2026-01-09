
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta


DATA_DIR = "data"
FILE_MAX = os.path.join(DATA_DIR, "cac40_history.csv")
FILE_3Y = os.path.join(DATA_DIR, "data3y.csv")
FILE_7D = os.path.join(DATA_DIR, "data7d.csv")
LAST_UPDATED_FILE = os.path.join(DATA_DIR, "last_updated.txt")

os.makedirs(DATA_DIR, exist_ok=True)

CAC40= [
    '^FCHI', 'AI.PA', 'AIR.PA', 'ALO.PA', 'BN.PA', 'BNP.PA', 'CA.PA',
    'CAP.PA', 'CS.PA', 'DG.PA', 'DSY.PA', 'EL.PA', 'EN.PA', 'ENGI.PA',
    'ERF.PA', 'GLE.PA', 'HO.PA', 'KER.PA', 'LR.PA', 'MC.PA', 'ML.PA',
    'OR.PA', 'ORA.PA', 'PUB.PA', 'RCO.PA', 'RI.PA', 'RMS.PA', 'SAF.PA',
    'SAN.PA', 'SGO.PA', 'STMPA.PA', 'SU.PA', 'TEP.PA',
    'TTE.PA', 'VIE.PA', 'VIV.PA', 'WLN.PA'
]

def update_1min_df():
    df = yf.download(CAC40, period="7d", interval="1m")["Close"]
    df = df.reset_index()
    df.columns = df.columns.str.replace('^FCHI', 'Cac40')
    df.columns = df.columns.str.replace('Datetime', 'Date')
    other_columns = [col for col in df.columns if col not in ['Date', 'Cac40']]
    df = df[['Date', 'Cac40'] + other_columns]
    

    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"].dt.strftime('%Y-%m-%d')

    df.to_csv(FILE_7D, index=False)




def update_max_df():
    df = yf.download(CAC40, period="max", interval="1d")["Close"]
    df = df.reset_index()
    df.columns = df.columns.str.replace('^FCHI', 'Cac40')
    other_columns = [col for col in df.columns if col not in ['Date', 'Cac40']]
    df = df[['Date', 'Cac40'] + other_columns]

    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"].dt.strftime('%Y-%m-%d')

    df.to_csv(FILE_MAX, index=False)
    three_years_ago = pd.Timestamp.today() - pd.DateOffset(years=3)
    df_3y = df[df['Date'] >= three_years_ago]
    df_3y.to_csv(FILE_3Y, index=False)


def update_last_updated():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LAST_UPDATED_FILE, "w", encoding="utf-8") as f:
        f.write(now)
def get_last_updated():
    if not os.path.exists(LAST_UPDATED_FILE):
        return None

    with open(LAST_UPDATED_FILE, "r", encoding="utf-8") as f:
        date_str = f.read().strip()

    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def reload_all_data():
    now = datetime.now()
    last_updated = get_last_updated()

    # First run we update everything
    if last_updated is None:
        update_max_df()
        update_1min_df()
        update_last_updated()
        return

    # Update daily data if day changed
    if now.date() != last_updated.date():
        update_max_df()

    # Update minute data more than 1 minute passed
    if now - last_updated >= timedelta(minutes=1):
        update_1min_df()

    update_last_updated()


    

if __name__ == "__main__":
    reload_all_data()
