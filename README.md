# GIT Linux Final Project - Interactive Financial Dashboard of the Cac40

LINK : http://20.160.99.10:8501/home_page


This project is a **financial analytics and portfolio management web application** focused on the **CAC 40**, combining **market monitoring, quantitative analysis, and systematic portfolio construction**.  
It is designed to provide both **high-level market insights** and **advanced quantitative tools** through an interactive interface.

---

### 1) Interactive Dashboard

The interactive dashboard provides a **global overview of the CAC 40 market dynamics**.  
It includes key indicators such as returns and prices to have a quick look of the current market state.  
The dashboard is designed for **fast decision-making**, offering a clear and intuitive visualization of market conditions.

<img width="1793" height="870" alt="image" src="https://github.com/user-attachments/assets/f87797f9-30c9-49b5-b356-42822e9f64c9" />

---

### 2) Daily report directly on your inbox

Users can subscribe to an **automated daily report**, sent directly by email.  
<img width="492" height="311" alt="image" src="https://github.com/user-attachments/assets/ea9f7a14-ed26-4064-a487-30af640bf09c" />

This report summarizes the **main market movements** of the day, allowing users to stay informed without accessing the platform manually.

<img width="1501" height="848" alt="image" src="https://github.com/user-attachments/assets/f27ae331-40ff-4b28-a803-c00789eabf25" />

---

### 3) Single Asset Management

This section focuses on the **individual analysis of a single CAC 40 component**.  
It provides detailed historical price evolution, return profiles, volatility, key metrics and machine learning features to help the user to make a decision. It also feature a backtesting strategy to try well know allocation strats.

<img width="1805" height="889" alt="image" src="https://github.com/user-attachments/assets/8a5dd3a5-a8ec-4251-8e31-4d1c7b772eed" />

---

### 4) Backtesting of a strategy : Black & Litterman with momentum views

The platform includes a*portfolio backtesting module based on the **Black-Litterman framework enhanced with momentum views**. This strategy is a dense and complicated and was developped during Malo's Research project (you can check the github here : https://github.com/MaloBardin/Research-Project and the paper is in publication process).
This allows users to backtest the strategy, compare it to benchmark portfolios and other assets, and evaluate performance using risk-adjusted metrics such as drawdowns, volatility, and Sharpe ratios. It also feature a multi run backtesting to proove the robustness of the Black and Litterman allocation.
The associated python file (VVE.py) is very large and wasn't created for this project, just modified to work with EU data.
<img width="1778" height="815" alt="image" src="https://github.com/user-attachments/assets/5e4af14f-f03d-424a-8c57-7fed3c108472" />

---

### 5) Yfinance and Bloomberg API data

THe data is collected automaticly every 5min using cron and the python librairie yfinance. Since we don't want to spam API request, we manage to have an intelligent request that will only add the data if it's missing. We did manage to get a working VBA Bloomberg code to get the csv data we need for our project but since we have hosted it on a VM, it doesn't have the access on the Bloomberg API (we can only use it on a computer school). You can find the VBA code here and we also added a little python script for the blpapi librairie

<img width="1033" height="628" alt="image" src="https://github.com/user-attachments/assets/5ca10d11-3385-4fe8-9b50-e60fb311f167" />
<img width="1475" height="878" alt="image" src="https://github.com/user-attachments/assets/47995939-fe25-4989-a6ce-10ae14f41639" />

VBA code : 

    Sub Datagrab
    Dim ws As Worksheet
    Dim ArrTickers As Variant
    Dim i As Long

    Set ws = ThisWorkbook.Sheets(1)
    ws.Cells.Clear

    ArrTickers = Array( _
        "CAC Index", _
        "AI FP Equity", "AIR FP Equity", "MT NA Equity", "CS FP Equity", "BNP FP Equity", "EN FP Equity", _
        "CAP FP Equity", "CA FP Equity", "ACA FP Equity", "BN FP Equity", "DSY FP Equity", "EDEN FP Equity", _
        "ENGI FP Equity", "EL FP Equity", "ERF FP Equity", "RMS FP Equity", "KER FP Equity", "OR FP Equity", _
        "LR FP Equity", "MC FP Equity", "ML FP Equity", "ORA FP Equity", "RI FP Equity", "PUB FP Equity", _
        "RNO FP Equity", "SAF FP Equity", "SGO FP Equity", "SAN FP Equity", "SU FP Equity", "GLE FP Equity", _
        "STLAP FP Equity", "STMPA FP Equity", "TTE FP Equity", "HO FP Equity", "URW NA Equity", "VIE FP Equity", _
        "DG FP Equity", "VIV FP Equity", "WLN FP Equity", "AC FP Equity")

 
    ws.Cells(1, 1).Value = "Date"

    For i = LBound(ArrTickers) To UBound(ArrTickers)
        ws.Cells(1, i + 2).Value = ArrTickers(i)
    Next i

    ws.Cells(2, 1).Formula = _
        "=BDH(""" & ArrTickers(0) & """,""PX_LAST"",""-3AY"","""",""Fill=P"")"

    For i = 1 To UBound(ArrTickers)
        ws.Cells(2, i + 2).Formula = _
            "=BDH(""" & ArrTickers(i) & """,""PX_LAST"",""-3AY"","""",""Dates=H"",""Fill=P"")"
    Next i
    End Sub

Python Code :

    from xbbg import blp
    TICKERS = [
        "CAC Index",
        "AI FP Equity", "AIR FP Equity", "MT NA Equity", "CS FP Equity", "BNP FP Equity",
        "EN FP Equity", "CAP FP Equity", "CA FP Equity", "ACA FP Equity", "BN FP Equity",
        "DSY FP Equity", "EDEN FP Equity", "ENGI FP Equity", "EL FP Equity", "ERF FP Equity",
        "RMS FP Equity", "KER FP Equity", "OR FP Equity", "LR FP Equity", "MC FP Equity",
        "ML FP Equity", "ORA FP Equity", "RI FP Equity", "PUB FP Equity", "RNO FP Equity",
        "SAF FP Equity", "SGO FP Equity", "SAN FP Equity", "SU FP Equity", "GLE FP Equity",
        "STLAP FP Equity", "STMPA FP Equity", "TTE FP Equity", "HO FP Equity",
        "URW NA Equity", "VIE FP Equity", "DG FP Equity", "VIV FP Equity",
        "WLN FP Equity", "AC FP Equity"
    ]
    
    FIELD = "PX_LAST"
    START_DATE = (datetime.today() - timedelta(days=3*365)).strftime("%Y%m%d")
    END_DATE = datetime.today().strftime("%Y%m%d")
    df = blp.bdh(
            tickers=tickers,
            flds="PX_LAST",
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d'),
            Per="D",
            Fill="P",   
            Days="A"
        )

    df.to_csv(data3y.csv)

As we said, since we've hosted our project on a virtual machine, theses codes aren't working and are just for educational purposes only.
    

### 6) Fully hosted on Microsoft Azure using Docker

The application is fully containerized and deployed using **Docker**, ensuring reproducibility, scalability, and ease of deployment.  
It is hosted on Microsoft Azure, providing a reliable cloud infrastructure suitable for production-level financial applications.

## Prerequisites

Before starting, ensure that **Docker** is installed and running on your machine.

- [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

## Installation

Clone the repository and navigate to the folder:

```bash
git clone [https://github.com/MaloBardin/GIT-Linux-Final-Project.git](https://github.com/MaloBardin/GIT-Linux-Final-Project.git)
cd GIT-Linux-Final-Project
```

Build the Docker image:

```bash
docker build -t streamlit-app .
```

---

## Usage

### Run the container

To run the application and see logs in the terminal:

```bash
docker run -p 8501:8501 streamlit-app
```

## Accessing the Application

Once the container is running, open your browser at the following address:

- **Local:** http://localhost:8501
- **Remote Server:** `http://<YOUR_SERVER_IP>:8501`
