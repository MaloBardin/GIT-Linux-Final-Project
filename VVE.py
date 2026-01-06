#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from fredapi import Fred
import warnings
import yfinance as yf
warnings.filterwarnings("ignore")
import streamlit as st

from grabbing_dataframe import GetDfForDashboard, Dfcleaning, ReadDf
#%%
def GetDf():
    # tickers cac40
    cac40 = [
        '^FCHI','AI.PA', 'AIR.PA', 'ALO.PA', 'BN.PA', 'BNP.PA', 'CA.PA',
        'CAP.PA', 'CS.PA', 'DG.PA', 'DSY.PA', 'EL.PA', 'EN.PA', 'ENGI.PA',
        'ERF.PA', 'GLE.PA', 'HO.PA', 'KER.PA', 'LR.PA', 'MC.PA', 'ML.PA',
        'OR.PA', 'ORA.PA', 'PUB.PA', 'RCO.PA', 'RI.PA', 'RMS.PA', 'SAF.PA',
        'SAN.PA', 'SGO.PA', 'STMPA.PA', 'SU.PA', 'TEP.PA',
        'TTE.PA', 'VIE.PA', 'VIV.PA', 'WLN.PA'
    ]

    df = yf.download(cac40, period='3y', interval='1d')['Close']
    return df

df = GetDf()
df = df.reset_index()
df.columns = df.columns.str.replace('^FCHI', 'Cac40')

autres_colonnes = [col for col in df.columns if col not in ['Date', 'Cac40']]
df = df[['Date', 'Cac40'] + autres_colonnes]


df["Date"] = pd.to_datetime(df["Date"])
print(df.columns)

#%%
df

#%%
def GetReturn(df, date, lookback):
    date = pd.to_datetime(date)
    if date not in df["Date"].values:
        #add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date", "AI.PA", "AIR.PA", "ALO.PA", "BN.PA", "BNP.PA", "CA.PA", "CAP.PA",
                     "CS.PA", "DG.PA", "DSY.PA", "EL.PA", "EN.PA", "ENGI.PA", "ERF.PA", "GLE.PA",
                     "HO.PA", "KER.PA", "LR.PA", "MC.PA", "ML.PA", "OR.PA", "ORA.PA", "PUB.PA",
                     "RCO.PA", "RI.PA", "RMS.PA", "SAF.PA", "SAN.PA", "SGO.PA", "STMPA.PA",
                     "SU.PA", "TEP.PA", "TTE.PA", "VIE.PA", "VIV.PA", "WLN.PA"]].copy()

    date_list = returns_df.drop(columns="Date")
    date_index = returns_df.index[returns_df["Date"] == date][0]
    returns_df = returns_df[(returns_df.index <= date_index) & (returns_df.index >= date_index - lookback)]
    returns_df.drop(columns="Date", inplace=True)

    returns_df = np.log(returns_df / returns_df.shift(1))
    returns_df.dropna(inplace=True)
    #print(returns_df.std().mean()) #verification if std is around 1% daily

    return returns_df

#return a df of size (lookback, number of sectors) with log returns


def GetReturnSPX(df,date,lookback):
    date=pd.to_datetime(date)
    if date not in df["Date"].values:#add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date","Cac40"]].copy()

    date_list=returns_df.drop(columns="Date")
    date_index = returns_df.index[returns_df["Date"] == date][0]

    returns_df=returns_df[(returns_df.index<=date_index) & (returns_df.index>=date_index-lookback) ]
    returns_df.drop(columns="Date",inplace=True)

    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)
    #print(returns_df.std().mean()) #verification if std is around 1% daily

    return returns_df

#return a df of size (lookback, 1) with log returns of SPX
#%%
def GetSigma(df,date,lookback):

    returns_df=GetReturn(df,date,lookback=lookback)
    #covariance matric from returns_df
    sigma_windowed=returns_df.cov()

    return sigma_windowed

from sklearn.covariance import LedoitWolf,OAS
import pandas as pd

def get_shrunk_covariance(df,date,lookback):

    returns=GetReturn(df,date,lookback)

    lw = OAS()
    lw.fit(returns)
    shrunk_cov = lw.covariance_

    delta = lw.shrinkage_
    if isinstance(returns, pd.DataFrame):
        shrunk_cov = pd.DataFrame(
            shrunk_cov,
            index=returns.columns,
            columns=returns.columns
        )


    return shrunk_cov


def getSigmaModified(df,date,lookback,listofbanneddays,periodison=False):

    date=pd.to_datetime(date)
    if date not in df["Date"].values:#add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date", "AI.PA", "AIR.PA", "ALO.PA", "BN.PA", "BNP.PA", "CA.PA", "CAP.PA",
                     "CS.PA", "DG.PA", "DSY.PA", "EL.PA", "EN.PA", "ENGI.PA", "ERF.PA", "GLE.PA",
                     "HO.PA", "KER.PA", "LR.PA", "MC.PA", "ML.PA", "OR.PA", "ORA.PA", "PUB.PA",
                     "RCO.PA", "RI.PA", "RMS.PA", "SAF.PA", "SAN.PA", "SGO.PA", "STMPA.PA",
                     "SU.PA", "TEP.PA", "TTE.PA", "VIE.PA", "VIV.PA", "WLN.PA"]].copy()

    date_index = returns_df.index[returns_df["Date"] == date][0]
    returns_df=returns_df[(returns_df.index<=date_index) & (returns_df.index>=date_index-lookback)]
    #days selection

    #banned days
    for banned_date in listofbanneddays:
        mask = returns_df["Date"] == banned_date
        if mask.any():
            print("got one :", banned_date)
            returns_df.loc[mask, :] = np.nan

    returns_df.drop(columns="Date",inplace=True)
    returns_df.dropna(inplace=True)

    #calculation of returns
    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)

    #covaraicne matrix using shrinkage
    lw = OAS()
    lw.fit(returns_df)
    shrunk_cov = lw.covariance_

    delta = lw.shrinkage_
    if isinstance(returns_df, pd.DataFrame):
        shrunk_cov = pd.DataFrame(
            shrunk_cov,
            index=returns_df.columns,
            columns=returns_df.columns
        )


    return shrunk_cov

#return a cov matrix of size (number of sectors, number of sectors) we use lookback to have different window sizes
#%%
def GetRfDataframe(df):
    fred = Fred(api_key="5c742a53d96bd3085e9199dcdb5af60b")
    riskfree = fred.get_series('DFF')
    # riskfree = fred.get_series('DTB1MO')

    riskfree = riskfree.to_frame(name='FedFunds')
    riskfree.index.name = "Date"
    riskfree = riskfree[riskfree.index >= "2002-01-01"]
    riskfree["FedFunds"]=riskfree["FedFunds"]/100
    list_days_open = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    list_days_full = pd.to_datetime(riskfree.index, dayfirst=True, errors="coerce")

    list_days_open=[pd.to_datetime(date) for date in list_days_open]
    list_days_full=[pd.to_datetime(date) for date in list_days_full]


    list_days_open_pondered=[]
    riskfree_list=[]
    count_list=[]
    timestamp=0
    while timestamp < len(list_days_full)-1:

      if list_days_full[timestamp+1] in list_days_open:
            list_days_open_pondered.append(list_days_full[timestamp])
            riskfree_list.append(riskfree["FedFunds"].loc[list_days_full[timestamp]])
            count_list.append(1)
            timestamp += 1

      else:
          count = 0
          timestampbis = timestamp
          while (timestamp + 1 < len(list_days_full)) and (list_days_full[timestamp + 1] not in list_days_open):
              timestamp += 1
              count += 1

          list_days_open_pondered.append(list_days_full[timestampbis])  # jour de départ
          riskfree_list.append(riskfree["FedFunds"].loc[list_days_full[timestampbis]])
          count_list.append(count+1)
          timestamp += 1

    RfDf=pd.DataFrame({"Date":list_days_open_pondered,"Rf":riskfree_list,"Count":count_list})
    RfDf=RfDf.set_index("Date")
    return RfDf

def GetRiskFree(df,date,lookback,RfDf):
    positionOfStartDate=df.index[df["Date"]==pd.to_datetime(date)][0]-lookback
    #print(positionOfStartDate)
    startDate=pd.to_datetime(df.iloc[positionOfStartDate,0])
    endDate=pd.to_datetime(date)
    RfDf=RfDf[(RfDf.index >= startDate) & (RfDf.index <= endDate )].copy()
    CumulativeRf=[]

    for i in range(len(RfDf)):
      if i==0:
        CumulativeRf.append(pow((1+RfDf["Rf"].iloc[i]),(RfDf["Count"].iloc[i]/360)))
      else:
        CumulativeRf.append(pow((1+RfDf["Rf"].iloc[i]),(RfDf["Count"].iloc[i]/360))*CumulativeRf[i-1])

    RfDf["CumulativeRf"]=CumulativeRf
    RfDf["CumulativeRf"]= RfDf["CumulativeRf"]-1
    if RfDf.empty:
        return 0.0  # aucun taux sans risque disponible

    return RfDf["CumulativeRf"].iloc[-1]

RfDf=GetRfDataframe(df)

#compute risk free dataframe using API from FRED and get the cumulative risk free rate between two dates in a df
#%%
def GetWeight(df,date):
    #for the moment we will use the equal weight
    weight_vector=np.zeros((36,1))
    for i in range(0,36):
        weight_vector[i]=1/36

    return weight_vector
Weight=GetWeight(df,"2020-05-11")

#%%
def GetLambda(df,date,timeofcalculation,RfDf):
    returns=GetReturn(df,date,timeofcalculation) #daily returns
    weight_vector=GetWeight(df=0,date=0)

    mean_return=np.mean(np.dot(returns,weight_vector))
    mean_annual=(1+mean_return)**252-1 #annualized mean return


    rf_temps=GetRiskFree(df,date,timeofcalculation,RfDf)
    rf_annual=(1+rf_temps)**(252/timeofcalculation)-1 #annualized risk free rate


    Sigma=get_shrunk_covariance(df,date,timeofcalculation)
    Sigma_annual=252*Sigma #annualized covariance matrix
    var = float((weight_vector.T @ Sigma_annual.values @ weight_vector).item())
    lambda_value=(mean_annual - rf_annual)/var


    excess = mean_annual - rf_annual
    sigma2 = var
    sigma  = np.sqrt(var)
    lam    = excess / sigma2
    sharpe = excess / sigma
    #print("Excess:", excess, " Var:", sigma2, " Vol:", sigma, " λ:", lam, " Sharpe:", sharpe)

    return lambda_value

#compute the lambda value using the mean return, risk free rate and variance of the portfolio

Lambda=GetLambda(df,"2024-01-11",timeofcalculation=150,RfDf=RfDf)


#%%
#add the Q matrix calculation

def QMatrixCalculation(df,date,lookback,proportion,performerc_daily,dailyperf_market,historical_returns):
    Q=np.zeros((proportion,1))
    factor=1
    for i in range(proportion):
        Q[i,0]=(performerc_daily[i][0]-dailyperf_market)/2


    return Q,historical_returns
#%%
def GetPMatrix(df,date, lookback,proportion=3,historical_returns=0):
    #(date)
    #print(proportion)
    #print(lookback)
    AssetColumns = [
    "AI.PA", "AIR.PA", "ALO.PA", "BN.PA", "BNP.PA", "CA.PA", "CAP.PA",
    "CS.PA", "DG.PA", "DSY.PA", "EL.PA", "EN.PA", "ENGI.PA", "ERF.PA", "GLE.PA",
    "HO.PA", "KER.PA", "LR.PA", "MC.PA", "ML.PA", "OR.PA", "ORA.PA", "PUB.PA",
    "RCO.PA", "RI.PA", "RMS.PA", "SAF.PA", "SAN.PA", "SGO.PA", "STMPA.PA",
    "SU.PA", "TEP.PA", "TTE.PA", "VIE.PA", "VIV.PA", "WLN.PA"]
    bestperformer = []
    performerc = []
    performerc_daily=[]
    returnBestPerformer=[]
    endDateIndex=df.index[df["Date"]==pd.to_datetime(date)][0]
    startDateIndex=df.index[df["Date"]==pd.to_datetime(date)][0]-lookback

    for i in range(2, df.shape[1]):  #loop through asset columns
        performerc.append((((float(df.iloc[endDateIndex, i]) / float(df.iloc[startDateIndex, i]) - 1) * 100), i - 2,df.columns[i])) #pos of best stock in a tuple
        # with its return
        performerc_daily.append(((float(df.iloc[endDateIndex, i]) / float(df.iloc[startDateIndex, i])) ** (1/lookback) - 1, i - 2,df.columns[i])) #daily version


    performerc.sort(reverse=True)
    performerc_daily.sort(reverse=True)
    #print(performerc)
    perfMarket= (float(df.iloc[endDateIndex, 1]) / float(df.iloc[startDateIndex, 1]) - 1) * 100
    dailyperf_market = (float(df.iloc[endDateIndex, 1]) / float(df.iloc[startDateIndex, 1])) ** (1/lookback) - 1






    for i in range(proportion):
        bestperformer.append(performerc_daily[i][1])
        returnBestPerformer.append(performerc_daily[i][0])


    P=np.zeros((proportion,36))
    Q=np.zeros((proportion,1))
    for lineview in range(proportion):
        for i in range(len(AssetColumns)):
            P[lineview,i]=-1/len(AssetColumns)
        P[lineview,bestperformer[lineview]]=1-1/len(AssetColumns)
        sum=0
        for i in range(len(AssetColumns)):
            sum+=P[lineview,i]
    Q,historical_returns=QMatrixCalculation(df,date,lookback,proportion,performerc_daily,dailyperf_market,historical_returns)


    return P, Q, historical_returns
#%%
def GetOmega(PMatrix, Sigma, c=0.99):
    #Omega is the uncertainty of the views

    factorC=(1/c-1)
    Omega=factorC*PMatrix@Sigma@np.transpose(PMatrix)

    return Omega
#%%
def GetOmega(PMatrix, Sigma, c=0.99):
    #Omega is the uncertainty of the views

    factorC=(1/c-1)
    Omega=factorC*PMatrix@Sigma@np.transpose(PMatrix)

    return Omega
#%%
def LinkOmegaTau2(Omega, Sigma,P,tau):
    #Link omega to tau
    numerator= np.trace(np.linalg.inv(Sigma*tau))
    denominator= np.trace((np.transpose(P)@np.linalg.inv(Omega)@P))
    result=numerator/denominator

    return result


#%%
def BlackAndLittermanModel(backtestStartDate, rebalancingFrequency, lookbackPeriod, df,RfDf,confidence=0.75,proportion=4,tau=0.025,Lambda=3,historical_returns=0,modifiedlambda=0):
    #implement the full backtest of the black and litterman model

    #---------
    #PARAMETERS
    #---------
    datetoremove=pd.to_datetime("2018-04-06") #add date to remove
    listofbanneddays=[]
    Sigma=get_shrunk_covariance(df,backtestStartDate,lookback=60) #using 720 days to have better sigma of 2 years
    Sigma=getSigmaModified(df,backtestStartDate,lookback=60,listofbanneddays=listofbanneddays) #using 720 days to have better sigma of 2 years


    PMatrix,Q,historical_returns= GetPMatrix(df,backtestStartDate, lookback=lookbackPeriod,proportion=proportion,historical_returns=historical_returns)
    Omega=GetOmega(PMatrix, Sigma, c=confidence)
    rf=GetRiskFree(df,backtestStartDate,lookbackPeriod,RfDf)
    weights = GetWeight(df, backtestStartDate)
    weights = np.array(weights).reshape(-1, 1)

    if modifiedlambda==True:
        if pd.to_datetime(backtestStartDate) < pd.to_datetime("2006-04-06"):
            Lambda=3
        else :
            Lambda=3+0.1*GetLambda(df,backtestStartDate,timeofcalculation=60,RfDf=RfDf)
    else :
        Lambda=3

    uimplied = Lambda * (Sigma @ weights) + rf
    #BL formula
    #tau=OmegaLinked




    optimizedReturn=(np.linalg.inv(np.linalg.inv(tau*Sigma)+np.transpose(PMatrix)@np.linalg.inv(Omega)@PMatrix)) @ (np.linalg.inv(tau*Sigma)@uimplied+np.transpose(PMatrix)@np.linalg.inv(Omega)@Q)
    LambdaMarkowitz=3

    #MarkowitzAllocation
    WeightBL=np.linalg.inv(Sigma)@(optimizedReturn-rf)/LambdaMarkowitz
    WeightRF=1-np.sum(WeightBL)
    #if not np.isclose(float(np.sum(WeightBL)), 1.0, atol=1e-6):
        #print(np.sum(WeightBL))
        #raise ValueError("Weights do not sum to 1, please investigate.")

    return WeightBL,WeightRF,historical_returns


BlackAndLittermanModel("2024-06-11", rebalancingFrequency=3, lookbackPeriod=180, df=df,RfDf=RfDf)

#%%
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

console = Console()

#BACK TESTER
dfbacktest=df.copy()
dfbacktest["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
dfbacktest["MonthIndex"] = dfbacktest["Date"].dt.to_period("M")

df_length = dfbacktest.shape[1] - 2  # bcs of date and spx
last_rebalance = dfbacktest.loc[0, "Date"]  # première date
month_count = 0

# 🎨 AFFICHAGE STYLÉ (sans prompts)
hold = 1
hist = 0
proportion = 4
Lambda=3
tau=0.025
confidence=0.75

console.print(Panel.fit(
    "[bold cyan]📊 PORTFOLIO BACKTESTER[/bold cyan]\n"
    "[dim]Black-Litterman Model[/dim]",
    border_style="cyan"
))

console.print(f"\n[yellow]⚙️  Configuration :[/yellow]")
console.print(f"   • Hold period: [cyan]{hold}[/cyan] mois")
console.print(f"   • Historique: [cyan]{hist}[/cyan] mois")
console.print(f"   • Proportion: [cyan]{proportion}[/cyan]")
console.print(f"   • Lambda: [cyan]{Lambda:.4f}[/cyan]")
console.print(f"   • Confiance: [cyan]{confidence}[/cyan]")
console.print(f"   • Taux: [cyan]{tau}[/cyan]\n")

console.print("\n[yellow]⏳ Lancement du backtest...[/yellow]\n")
def Backtester(df,hold, hist, proportion,df_toBL, RfDf,confidence2,proportion2,tau2,Lambda2,start,modifiedlambda):
    #new dataframe for stock quantity

    StockQty = df.copy()
    StockQty.drop(columns="MonthIndex", inplace=True)
    historical_returns=[]

    StockQty.loc[:, :] = 0
    #starting data
    MoneyAtStart = 10000000
    month_count=0
    CurrentValue=MoneyAtStart
    spaceindays=0
    #first ligne
    StockQty.loc[start, "Money"] = MoneyAtStart
    StockQty.loc[start, "Cac40"] = df.iloc[start, 1]
    StockQty.loc[start, "Date"] = df.iloc[start, 0]
    RiskFreeAmount=0
    #start of the algorithm

    for i in tqdm(range(start,df.shape[0]), desc="Backtesting"):
      StockQty.iloc[i,0]=df.iloc[i,0]
      StockQty.iloc[i,1]=df.iloc[i,1]
      fees=0


      if df.loc[i, "Date"].month != df.loc[i-1, "Date"].month:
        month_count += 1


    # Si on atteint la période voulue
      if i>= hist and spaceindays>21*hold:
        #print(f"🔁 Rebalancement déclenché à la date : {df.loc[i, 'Date'].date()}")
        #print(str(df.iloc[i,0]))

        spaceindays=0

        BLWeight,RiskFreeAmount,historical_returns=BlackAndLittermanModel(str(df.iloc[i,0]),3,3*22,df_toBL,RfDf,confidence=confidence2,proportion=proportion2,tau=tau2,Lambda=Lambda2,historical_returns=historical_returns,modifiedlambda=modifiedlambda)
        #print(len(BLWeight))
        for index in range(len(BLWeight)):
            StockQty.iloc[i,index+2]=(BLWeight.iloc[index,0]*CurrentValue)/df.iloc[i,index+2] #qty = weight*total value/price
      else :
        spaceindays+=1
        for stocks in range(2,StockQty.shape[1]-1):
          StockQty.iloc[i,stocks]=StockQty.iloc[i-1,stocks] #same qty


      #value of pf

      GainOrLoss = 0
      for stocks in range(2, StockQty.shape[1]-1):
        qty = StockQty.iloc[i, stocks]

        if qty != 0.0:
            price_now = df.iloc[i, stocks]
            price_prev = df.iloc[i-1, stocks]
            GainOrLoss += qty * (price_now - price_prev)

      daily_rate = GetRiskFree(df, str(df.iloc[i,0]), 1, RfDf)
      interest_gain = (CurrentValue * RiskFreeAmount) * daily_rate
      CurrentValue += GainOrLoss + interest_gain - fees
      StockQty.iloc[i,-1]=CurrentValue


    StockQty = StockQty.iloc[start:].reset_index(drop=True)
    return StockQty


def RunBacktest(hold=1, hist=3, proportion=3, confidence=0.2,ModifiedLambda=False):
    print(hold, hist, proportion, confidence)
    RfDf = GetRfDataframe(df)
    final=Backtester(dfbacktest, hold=hold, hist=hist, proportion=proportion, df_toBL=df,RfDf=RfDf,confidence2=confidence,proportion2=proportion,tau2=tau,Lambda2=Lambda,start=181,modifiedlambda=ModifiedLambda)

    final.to_csv("backtest_bl.csv")

    console.print("\n[green]✅ Backtest terminé avec succès ![/green]\n")

def getPrintableDf(final):

    money_norm = (final["Money"]/10000000*100) - 100
    spx_norm = (final["Cac40"]/final["Cac40"].iloc[0]*100) - 100

    df_plot = pd.DataFrame({
        "Date": final["Date"],
        "Portfolio": money_norm,
        "Cac40": spx_norm
    }).melt(id_vars="Date", var_name="Série", value_name="Évolution en %")

    return df_plot

#%%
def calculate_historical_var_es(df, col_name='Money', confidence_level=0.95):


    returns = df[col_name].pct_change().dropna()

    cutoff = 1 - confidence_level

    var_value = returns.quantile(cutoff)

    worst_returns = returns[returns <= var_value]
    es_value = worst_returns.mean()

    return {
        "confidence_level": confidence_level,
        "VaR": -var_value,
        "ES": -es_value,
        "count_returns": len(returns),
        "count_breaches": len(worst_returns)
    }



def calculate_sharpe_ratio(df, col_name='close', risk_free_rate_annual=0.04):


    returns = (df[col_name] - df[col_name].shift(1)) / df[col_name].shift(1)
    returns = returns.dropna()
    rf_daily = risk_free_rate_annual / 252
    excess_returns = returns - rf_daily

    sharpe_daily = excess_returns.mean() / excess_returns.std()

    sharpe_annualized = sharpe_daily * np.sqrt(252)

    return sharpe_annualized

#%%
'''
print("Portfolio Risk Measures:")
print(calculate_historical_var_es(final, 'Money', 0.99))
print(f"Sharpe Ratio: {calculate_sharpe_ratio(final, 'Money', 0.03):.2f}")

print("\nSPX Risk Measures:")
print(calculate_historical_var_es(final, 'SPX', 0.99))
print(f"Sharpe Ratio: {calculate_sharpe_ratio(final, 'SPX', 0.03):.2f}")'''
#%%
#%%
def dailyvol(final):
    import pandas as pd
    import numpy as np
    import plotly.express as px

    df2 = final[["Date", "Cac40", "Money"]].copy()
    df2["Portfolio"] = df2["Money"]
    df2.drop(columns="Money", inplace=True)
    df2["Date"] = pd.to_datetime(df2["Date"], dayfirst=True)
    df2.set_index("Date", inplace=True)

    daily_returns = df2.pct_change()
    daily_returns.dropna(inplace=True)

    daily_returns["annualizedvolCac40"] = daily_returns['Cac40'].rolling(window=252).std() * np.sqrt(252)
    daily_returns["annualizedVolPf"] = daily_returns['Portfolio'].rolling(window=252).std() * np.sqrt(252)
    data_to_plot = daily_returns.dropna()
    return data_to_plot
#%%
#multiple runs on variable start date :
def multirun(final,n_simulations=5):
    all_results = []

    for i in range(n_simulations):
        current_start = 181 + i
        results_df = Backtester(dfbacktest, hold=hold, hist=hist, proportion=proportion,
                                df_toBL=df, RfDf=RfDf, confidence2=confidence,
                                proportion2=proportion, tau2=tau, Lambda2=Lambda,
                                start=current_start, modifiedlambda=0)

        money_norm = (results_df["Money"] / 10_000_000 * 100) - 100
        temp_df = pd.DataFrame({f"Iter_{i}": money_norm.values})

        dateresults = results_df["Date"]
        temp_df.index = dateresults

        all_results.append(temp_df)

    global_df = pd.concat(all_results, axis=1)

    global_df_clean = global_df.dropna()

    dfcopyfinal = final[["Date", "Cac40"]].copy()
    dfcopyfinal.index = dfcopyfinal["Date"]
    dfcopyfinal.drop(columns="Date", inplace=True)

    global_df_clean = global_df_clean.merge(dfcopyfinal, left_index=True, right_index=True, how="left")
    spx_norm = (global_df_clean["Cac40"] / global_df_clean["Cac40"].iloc[0] * 100) - 100
    global_df_clean["Cac40"] = spx_norm
    print(global_df_clean.tail(5))
    return global_df_clean

#%%
import numpy as np
import plotly.graph_objects as go


def animate_dataframe_plotly(df,step=5,speed=20,spx_col="Cac40"):

    df_anim = df.iloc[::step]
    x = df_anim.index

    final_values = df_anim.iloc[-1]
    best_col = final_values.drop(spx_col, errors="ignore").idxmax()

    traces = []
    for col in df_anim.columns:
        if col == spx_col:
            color = "red"
            width = 2
            alpha = 1
        elif col == best_col:
            color = "green"
            width = 2
            alpha = 1
        else:
            color = "rgba(250, 250, 250,1)"
            width = 1
            alpha = 0.4
        traces.append(go.Scatter(x=[],y=[],mode="lines",line=dict(color=color, width=width),opacity=alpha,name=col,showlegend=False))

    frames = []
    for t in range(1, len(df_anim)):
        frame_data = []
        for col in df_anim.columns:
            frame_data.append(go.Scatter(x=x[:t],y=df_anim[col].values[:t]))
        frames.append(go.Frame(data=frame_data, name=str(t)))

    layout = go.Layout(title="Multiple Backtest Runs",xaxis=dict(title="Date"),yaxis=dict(title="Perf (%)"),plot_bgcolor="black",paper_bgcolor="black",font=dict(color="white"),updatemenus=[dict(type="buttons",showactive=False,buttons=[dict(label="▶ Play",method="animate",args=[None,dict( frame=dict(duration=speed, redraw=True),fromcurrent=True)]),dict(label="⏸ Pause",method="animate",args=[[None],dict(frame=dict(duration=0), mode="immediate")])])])

    fig = go.Figure(data=traces,layout=layout,frames=frames)

    st.plotly_chart(fig, config={'staticPlot': True}, use_container_width=True)


def GetInfoOnBacktest(df_final):
    col_name = "Unnamed: 0"

    if col_name in df_final.columns:
        df_final = df_final.drop(columns=[col_name])

    listofmostpickedassets=[]
    for i in range(2,df_final.shape[1]-1):
        listofmostpickedassets.append((df_final.columns[i],0))


    for lines in range(df_final.shape[0]):
        if df_final.iloc[lines,3] != 0:
            for assets in range(2,df_final.shape[1]-1):
                if df_final.iloc[lines,assets] > 0 and df_final.iloc[lines,assets] != df_final.iloc[lines-1,assets]:
                    listofmostpickedassets[assets-2]=(listofmostpickedassets[assets-2][0],listofmostpickedassets[assets-2][1]+1)

    print(listofmostpickedassets)
    return listofmostpickedassets


def GetReturnwihtoutcv(df, date, lookback):

    if date not in df["Date"].values:
        #add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date", "AI.PA", "AIR.PA", "ALO.PA", "BN.PA", "BNP.PA", "CA.PA", "CAP.PA",
                     "CS.PA", "DG.PA", "DSY.PA", "EL.PA", "EN.PA", "ENGI.PA", "ERF.PA", "GLE.PA",
                     "HO.PA", "KER.PA", "LR.PA", "MC.PA", "ML.PA", "OR.PA", "ORA.PA", "PUB.PA",
                     "RCO.PA", "RI.PA", "RMS.PA", "SAF.PA", "SAN.PA", "SGO.PA", "STMPA.PA",
                     "SU.PA", "TEP.PA", "TTE.PA", "VIE.PA", "VIV.PA", "WLN.PA"]].copy()

    date_list = returns_df.drop(columns="Date")
    date_index = returns_df.index[returns_df["Date"] == date][0]
    returns_df = returns_df[(returns_df.index <= date_index) & (returns_df.index >= date_index - lookback)]
    returns_df.drop(columns="Date", inplace=True)

    returns_df = np.log(returns_df / returns_df.shift(1))
    returns_df.dropna(inplace=True)
    #print(returns_df.std().mean()) #verification if std is around 1% daily

    return returns_df


#animate_dataframe_plotly(global_df_clean,step=5,speed=20)
def getCorrelationMatrix(df, date, lookback):
    print(date)
    returns_df = GetReturnwihtoutcv(df, date, lookback=lookback)
    correlation_matrix = returns_df.corr()

    return correlation_matrix


def plot_max_drawdown(df):
    mydf=df.copy()
    mydf["Date"] = pd.to_datetime(mydf["Date"])
    mydf.set_index("Date", inplace=True)
    col_money="Money"

    mydf['Running_Max'] = mydf[col_money].cummax()
    mydf['Drawdown'] = (mydf[col_money] - mydf['Running_Max']) / mydf['Running_Max']
    max_dd_value = mydf['Drawdown'].min()
    idx_bottom = mydf['Drawdown'].idxmin()
    idx_peak = mydf.loc[:idx_bottom][mydf.loc[:idx_bottom, 'Drawdown'] == 0].index[-1]
    recovery_period = mydf.loc[idx_bottom:][mydf.loc[idx_bottom:, 'Drawdown'] == 0]

    if not recovery_period.empty:
        idx_recovery = recovery_period.index[0]
        recovery_str = f"Recovered on {idx_recovery.strftime('%Y-%m-%d')}"
        recovered = True
    else:
        idx_recovery = mydf.index[-1]
        recovery_str = "Didn't recover yet"
        recovered = False

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mydf.index, y=mydf[col_money],
        mode='lines', name='Portfolio',
        line=dict(color='green', width=1.5)
    ))
    subset = mydf.loc[idx_peak:idx_recovery]

    fig.add_trace(go.Scatter(
        x=subset.index,
        y=subset[col_money],
        mode='lines',
        name='Max Drawdown',
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='red', width=2)
    ))
    fig.add_annotation(
        x=idx_bottom, y=mydf.loc[idx_bottom, col_money],
        text=f"Max DD: {max_dd_value:.2%}",
        showarrow=True, arrowhead=1, ax=0, ay=40, bordercolor="red", borderwidth=1
    )

    fig.add_annotation(
        x=idx_peak, y=mydf.loc[idx_peak, col_money],
        text="Apex", showarrow=True, arrowhead=1, ax=0, ay=-30
    )

    if recovered:
        fig.add_annotation(
            x=idx_recovery, y=mydf.loc[idx_recovery, col_money],
            text="Recovery", showarrow=True, arrowhead=1, ax=0, ay=-30
        )

    fig.update_layout(
        title=f"Max drawdown and recovery time",
        xaxis_title="Date",
        yaxis_title="Pf Value",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig, max_dd_value, idx_peak, idx_recovery