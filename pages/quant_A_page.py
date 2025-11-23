import streamlit as st
import pandas as pd
import yfinance as yf
import time

from quant_A_utils import get_data, long_moving_average, double_moving_average, performance_metrics





st.title("Trading Strategy Backtester 📈")
st.sidebar.header("⚙️ Paramètres")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Date de début")
end_date = st.sidebar.date_input("Date de fin")

interval = st.sidebar.selectbox("Fréquence des données",
                                options=["1h", "1d", "1wk", "1mo"],
                                format_func=lambda x: x.replace("1", ""))  # → hour, day, week, month

strategy_choice = st.sidebar.selectbox("Stratégie", ["long_moving_average", "double_moving_average"])


if strategy_choice == "long_moving_average":
    w = st.sidebar.number_input("Window (MA period)", 5, 200, value=20)
elif strategy_choice == "double_moving_average":
    w1 = st.sidebar.number_input("MA courte", 5, 200, value=20)
    w2 = st.sidebar.number_input("MA longue", 5, 200, value=50)



if st.sidebar.button("Lancer Backtest"):
    st.subheader(f"Données récupérées pour **{ticker}**")
    start_date = str(start_date)
    end_date = str(end_date)

    print("interval : ", interval)
    print(f"get_data({ticker},{start_date},{end_date},{interval})")
    df = get_data(ticker, str(start_date), str(end_date), interval)

    if df.empty:
        st.error("Erreur : Aucune donnée trouvée.")
    else:
        st.write(df.head())

        # STRATEGIE
        if strategy_choice == "long_moving_average":
            df_strategy = long_moving_average(df, w)
        elif strategy_choice == "double_moving_average":
            df_strategy = double_moving_average(df, w1, w2)

        # METRIQUES
        metrics = performance_metrics(df_strategy)

        st.subheader("📊 Performance")
        st.write(pd.DataFrame(metrics, index=[ticker]))

        # GRAPHE
        st.line_chart(df_strategy[['Price_Normalized', 'Cumulative_Strategy']])

        st.success("Backtest terminé !")
