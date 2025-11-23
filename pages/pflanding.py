import streamlit as st
import pandas as pd
import numpy as np
from BLStrat import getPrintableDf
from BLStrat import RunBacktest,GetInfoOnBacktest
import plotly.express as px

st.set_page_config(layout="wide")


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


#sliders
with col_droite:
    with st.container(border=True):
        st.write("**Strategy parameters**")
        hold_param = st.slider("Holding time in months", min_value=1, max_value=12, value=1)
        hist_param = st.slider("History time in months", min_value=1, max_value=12, value=3)
        numberviews_param = st.slider("Number of views", min_value=1, max_value=20, value=3)
        confidence_param = st.slider("Confidence level in %", min_value=0, max_value=100, value=20)
        if confidence_param == 0:
            confidence_param = 0.01
        confidence_param=confidence_param/100

    with st.container(border=False):
        if st.button("Run backtest", type="primary", use_container_width=True):
            RunBacktest(hold_param, hist_param, numberviews_param, confidence_param)
#graph
with col_gauche:

    with st.container(border=False):
        df_chart = getPrintableDf(pd.read_csv('backtest_bl.csv'))

        fix = px.line(
            df_chart,
            x="Date",
            y="Évolution en %",
            color="Série",
            color_discrete_map={"Cac40": "red", "Portfolio": "green"},
            title="Comparaison of the Black-Litterman Portfolio vs Cac40 in %"
        )

        fix.update_layout(hovermode="x unified")

        st.plotly_chart(
            fix,
            use_container_width=True,
            config={'staticPlot': True}
        )


    with st.container(border=False):
        data_tuples = GetInfoOnBacktest(pd.read_csv('backtest_bl.csv'))
        df_histo = pd.DataFrame(data_tuples, columns=["Actif", "Fréquence"])
        df_histo = df_histo.sort_values(by="Fréquence", ascending=False)

        fig = px.bar(
            df_histo,
            x="Actif",
            y="Fréquence",
            title="Most picked assets",
            color="Fréquence",  # Ajoute une couleur selon la hauteur
            color_continuous_scale="Redor",  # Dégradé rouge/orange (ou "Viridis", "Bluered")
            text_auto=True  # Affiche la valeur au dessus de la barre
        )

        # Nettoyage visuel (retirer le fond, etc.)
        fig.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )


        st.plotly_chart(fig,config={'staticPlot': True}, use_container_width=True)



