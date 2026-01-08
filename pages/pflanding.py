import streamlit as st
import pandas as pd
import numpy as np
from VVE import getPrintableDf,GetDf,runEveryDay
from VVE import RunBacktest,GetInfoOnBacktest,getCorrelationMatrix,dailyvol,calculate_sharpe_ratio,calculate_historical_var_es,multirun,plot_multirun_static,plot_max_drawdown
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import time
from mailsending import show_newsletter_popup
from utils import local_css, barre_menu


local_css("style.css")
barre_menu()

if "backtest_clicked" not in st.session_state:
    st.session_state.backtest_clicked = False


st.set_page_config(page_title="", layout="wide")



@st.dialog("The Black-Litterman Model")
def show_bl_info():
    st.write("### 🧠 Understanding the Model")

    st.write("""
    The **Black-Litterman** model is a sophisticated asset allocation method that combines two sources of information:

    1.  **Market Equilibrium** (based on CAPM): What the market implicitly "believes" or expects.
    2.  **Investor Views**: Your subjective opinions (e.g., "The Tech sector will outperform").

    **The Objective:**
    It addresses the limitations of the classic Markowitz model (which often yields extreme allocations) by producing more stable and diversified portfolios.
    """)

    st.write("**Key Formula:**")
    st.latex(r"E[R] = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q]")

    st.write("### 🛠️ Use the backtester tool")

    st.write("""
    1.  **Holding Time**: Duration to hold each portfolio before rebalancing""")
    st.write("2.  **History Time**: Lookback period to estimate returns and covariances")
    st.write("3.  **Number of Views**: How many subjective views to incorporate")
    st.write("4.  **Confidence Level**: Your confidence in these views (0% = no confidence, 100% = absolute confidence)")

    st.write("### 🚨 Attention, the views are computed by an momentum strategy on the historical data selected.")
    st.write("### ⚠️ Note: Running the backtest may take a few moments depending on the parameters chosen.")

col_titre, col_info = st.columns([100, 5])

with col_titre:
    st.markdown("""
        <h1 style='text-align: center; font-size: 50px;'>
            Black-Litterman Strategy Backtest
        </h1>
    """, unsafe_allow_html=True)

with col_info:
    st.write("")
    st.write("")
    if st.button("ℹ️", help="About the Black-Litterman Model"):
        show_bl_info()

col_gauche, col_droite = st.columns([3, 1], gap="medium")

df_backtest=pd.read_csv('backtest_bl.csv')
df_3ydata=pd.read_csv('data3y.csv')
#sliders
with col_droite:
    with st.container(border=True):
        st.write("**Strategy parameters**")
        hold_param = st.slider("Holding time in months", min_value=1, max_value=12, value=1)
        hist_param = st.slider("History time in months", min_value=1, max_value=12, value=3)
        numberviews_param = st.slider("Number of views", min_value=1, max_value=20, value=3)
        confidence_param = st.slider("Confidence level in %", min_value=0, max_value=100, value=75)
        DynamicLambda=st.checkbox("Use dynamic confidence level ?", value=False)
        if confidence_param == 0:
            confidence_param = 0.001
        if confidence_param == 100:
            confidence_param = 0.999
        confidence_param=confidence_param/100
    status_placeholder = st.empty()


    with st.container(border=False):
        if st.button("Run backtest", type="primary", use_container_width=True):
            with st.spinner('⏳ Running backtest, please wait...'):
                RunBacktest(hold_param, hist_param, numberviews_param, confidence_param,DynamicLambda)
                st.success("✅ Backtest finished !")
                import time
                st.session_state.mes_actifs_key = []
                time.sleep(0.5)
                st.rerun()


    with st.container(border=True):
        dfprice=pd.read_csv('data3y.csv')
        df_corr = getCorrelationMatrix(dfprice,dfprice['Date'].iloc[-1],22*hist_param)
        fig_corr = px.imshow(
            df_corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix of the Assets"
        )

        fig_corr.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_corr, width="stretch")

        #metrics infos
        st.subheader("📉 Risk Metrics")

        # Calculs
        var_es_pf = calculate_historical_var_es(df_backtest, "Money", 0.99)
        sharpe_pf = calculate_sharpe_ratio(df_backtest, "Money", 0.03)

        var_es_spx = calculate_historical_var_es(df_backtest, "Cac40", 0.99)
        sharpe_spx = calculate_sharpe_ratio(df_backtest, "Cac40", 0.03)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 💼 Portfolio")
            st.metric("VaR 99%", f"{var_es_pf['VaR']:.2%}")
            st.metric("ES 99%", f"{var_es_pf['ES']:.2%}")
            st.metric("Sharpe Ratio", f"{sharpe_pf:.2f}")

        with col2:
            st.markdown("#### 📊 &nbsp; &nbsp; Cac40")
            st.metric("VaR 99%", f"{var_es_spx['VaR']:.2%}")
            st.metric("ES 99%", f"{var_es_spx['ES']:.2%}")
            st.metric("Sharpe Ratio", f"{sharpe_spx:.2f}")
    with st.container(border=True):
        import os
        import datetime
        st.write("**Refresh the data (auto refresh every day)**")
        file_time = os.path.getmtime('data3y.csv')
        last_date = datetime.datetime.fromtimestamp(file_time).strftime('%d/%m/%Y at %H:%M')
        st.caption(f"📅 Last update : **{last_date}**")
        if st.button("Get the new data !", use_container_width=True):
            with st.spinner('⏳ Refreshing data, please wait...'):
                runEveryDay()
                st.success("✅ Data refreshed !")
                import time
                time.sleep(0.1)
                st.rerun()
#graph
with col_gauche:
    with st.container(border=True):
        oAssetColumns = [
    "AI.PA", "AIR.PA", "ALO.PA", "BN.PA", "BNP.PA", "CA.PA", "CAP.PA",
    "CS.PA", "DG.PA", "DSY.PA", "EL.PA", "EN.PA", "ENGI.PA", "ERF.PA", "GLE.PA",
    "HO.PA", "KER.PA", "LR.PA", "MC.PA", "ML.PA", "OR.PA", "ORA.PA", "PUB.PA",
    "RCO.PA", "RI.PA", "RMS.PA", "SAF.PA", "SAN.PA", "SGO.PA", "STMPA.PA",
    "SU.PA", "TEP.PA", "TTE.PA", "VIE.PA", "VIV.PA", "WLN.PA"]

        selection = st.multiselect(
            "📈 Display components of the Cac40 (⚠️ this will not have an impact on the composition of the portfolio since everything is automated)",
            options=oAssetColumns,
            default=None,
            key = "mes_actifs_key"
        )


    with st.container(border=True):
        df_chart = getPrintableDf(df_backtest,df_3ydata,selection)
        fix = px.line(
            df_chart,
            x="Date",
            y="Évolution en %",
            color="Série",
            color_discrete_map={"Cac40": "red", "Portfolio": "green"},
            title="Comparaison of the Black-Litterman Portfolio")

        fix.update_layout(hovermode="x unified")

        st.plotly_chart(fix, width="stretch")


    with st.container(border=True):
        data_tuples = GetInfoOnBacktest(df_backtest)
        df_histo = pd.DataFrame(data_tuples, columns=["Actif", "Fréquence"])
        df_histo = df_histo.sort_values(by="Fréquence", ascending=False)

        fig = px.bar(
            df_histo,
            x="Actif",
            y="Fréquence",
            title="Most picked assets",
            color="Fréquence",
            color_continuous_scale="Redor",
            text_auto=True
        )
        fig.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, width="stretch")
    with st.container(border=True):
        data_toplot=dailyvol(df_backtest)
        fig = px.line(data_toplot,
                      x=data_toplot.index,
                      y=["annualizedvolCac40", "annualizedVolPf"],
                      labels={"value": "annualized vol", "variable": "Actif", "Date": "Date"},
                      title="Annualized volatility")

        st.plotly_chart(fig, width="stretch")
    with st.container(border=True):
        fig, mdd, start, end = plot_max_drawdown(df_backtest)
        st.plotly_chart(fig, width="stretch")

        st.error(f"📉 **Maximum Drawdown :** {mdd:.2%}")
        st.info(f"⏱️ **Drop and recovery duration :** {(end - start).days} days")

st.divider()
st.subheader("🔃 Multi-Run Simulation")
with st.container(border=True):
    st.info("This section is used to run multiple backtest with mooving starting date to verify that the Black and Litterman model is not path dependant. It take the same parameters as the backtest above, please be patient.")
    col_input, col_btn = st.columns([1, 2])

    with col_input:
        nb_runs = st.number_input("Simulations number",min_value=1,max_value=50,value=5,step=1,help="The larger the number is, the longer the code will take to compute")

    with col_btn:
        st.write("")
        st.write("")
        start_multirun = st.button("🚀 Launch Multi-Run", type="primary")

    if start_multirun:
        my_bar = st.progress(0, text="Preparing simulations...")

        with st.spinner("Work in progress, the best simulation will be displayed in green, please wait..."):
            multirundf = multirun(dfprice, n_simulations=nb_runs, progress_bar=my_bar)
        st.toast("Simulations are finished !", icon="🏁")
        my_bar.empty()
        plot_multirun_static(multirundf)
