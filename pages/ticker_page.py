import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from grabbing_dataframe import Dfcleaning, ReadDf, getInfoperTicker2
from mailsending import sendmail, bulknl
print("EEEN COURS")
st.set_page_config(layout="wide", page_title="Ticker Analysis")
sendmail("malo.bardin@gmail.com", "j'aime beaucoup le projet", "C'est vraiment un super projet")
print("DONNNNEEEE")
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass



local_css("style.css")


def barre_menu():
    col1, col2,col3,col4,col6= st.columns(5)
    with col1:
        st.page_link("pages/home_page.py", label="Dashboard", use_container_width=True)
    with col2:
        st.page_link("pages/quant_A_page.py", label="Single Asset", use_container_width=True)
        
    with col3:
        st.page_link("pages/pflanding.py", label="Portfolio simulation", use_container_width=True)

    with col6:
        if st.button("📩 Subscribe to the daily report !", use_container_width=True):
            show_newsletter_popup()


barre_menu()

# Recuperation of the ticker from query params
if "ticker" in st.query_params:
    selected_ticker = st.query_params["ticker"]
else:
    selected_ticker = "Air Liquide" 

col_header, col_btn = st.columns([6, 1])

# Home button
with col_btn:
    st.write("") 
    if st.button("⬅️ Back", use_container_width=True):
        st.markdown('<meta http-equiv="refresh" content="0; url=/home_page" target="_self">', unsafe_allow_html=True)


try:
    # Data retrieval from yfinance
    intraday_data, one_month, five_year = getInfoperTicker2(selected_ticker)

    current_price = float(intraday_data['Close'].iloc[-1])
    
    variation_1d = ((current_price - intraday_data['Close'].iloc[0]) / intraday_data['Close'].iloc[0]) * 100
    variation_1m = ((current_price - one_month['Close'].iloc[0]) / one_month['Close'].iloc[0]) * 100
    variation_5y = ((current_price - five_year['Close'].iloc[0]) / five_year['Close'].iloc[0]) * 100

    
    # Momentum Calculation
    mean_price_30d = one_month['Close'].mean()
    
    if mean_price_30d > 0:
        is_overbuying = ((current_price / mean_price_30d) - 1) * 100
    else:
        is_overbuying = 0.0

    color_var = "green" if variation_1d >= 0 else "red"

    # Header
    with col_header:
        st.markdown(f"""
            <h1 style='margin-bottom: 0;'>{selected_ticker}</h1>
            <h3 style='margin-top: 0; color: {color_var};'>
                {current_price:.2f} € <span style='font-size: 20px;'>({variation_1d:+.2f}%)</span>
            </h3>
        """, unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1], gap="medium")

    # CHarts
    with col_left:
        st.subheader("Price Evolution")
        
        tab1, tab2, tab3 = st.tabs(["1 Day",  "30 Days", "5 Years"])

        with tab1:
            x_col = intraday_data.index
            if 'Datetime' in intraday_data.columns:
                x_col = 'Datetime'
            
            if not intraday_data.empty:
                fig_1d = px.line(intraday_data, x=x_col, y="Close", title="Intraday Trend", markers=True)
                fig_1d.update_layout(xaxis_title=None, yaxis_title="Price (€)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
                fig_1d.update_traces(line_color='#29f075' if variation_1d >= 0 else '#CC4974')
                st.plotly_chart(fig_1d, use_container_width=True)
            else:
                st.info("No data available for 1 Day.")

        with tab2:
            if not one_month.empty:
                fig_30d = px.line(one_month, x=one_month.index, y="Close", title="30-Day Trend (Hourly)")
                fig_30d.update_layout(xaxis_title=None, yaxis_title="Price (€)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
                fig_30d.update_traces(line_color='#29f075' if variation_1m >= 0 else '#CC4974')
                st.plotly_chart(fig_30d, use_container_width=True)
            else:
                st.info("No data available for 30 Days.")


        with tab3:
            if not five_year.empty:
                fig_5y = px.line(five_year, x=five_year.index, y="Close", title="5-Year Trend (Weekly)")
                fig_5y.update_layout(xaxis_title=None, yaxis_title="Price (€)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
                fig_5y.update_traces(line_color='#29f075' if variation_5y >= 0 else '#CC4974')
                st.plotly_chart(fig_5y, use_container_width=True)
            else:
                st.info("No data available for 5 Years.")

    # Technical Analysis and Quick Stats
    with col_right:
        st.subheader("Technical Analysis")
        
        with st.container(border=True):
            st.write("**Momentum Analysis**")
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = is_overbuying,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [-10, 10], 'tickwidth': 1, 'tickcolor': "black"},
                    'bar': {'color': "rgba(0,0,0,0)"}, 
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#e0e0e0",
                    'steps' : [
                        {'range': [-10, -5], 'color': "#1b5e20"}, 
                        {'range': [-5, -2], 'color': "#29f075"},  
                        {'range': [-2, 2], 'color': "#e0e0e0"},   
                        {'range': [2, 5], 'color': "#ffa726"},     
                        {'range': [5, 10], 'color': "#b71c1c"}    
                    ],

                    'threshold': {
                        'line': {'color': "black", 'width': 10},
                        'thickness': 1,  
                        'value': is_overbuying
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if is_overbuying < -5:
                st.success("**Strong Buy**: Significantly undervalued.")
            elif -5 <= is_overbuying < -2:
                st.success("**Buy**: Moderately undervalued.")
            elif -2 <= is_overbuying <= 2:
                st.info("**Neutral**: Price is stable.")
            elif 2 < is_overbuying <= 5:
                st.warning("**Sell**: Moderately overvalued.")
            else:
                st.error("**Strong Sell**: Significantly overvalued.")

        st.markdown("### Quick Stats")
        with st.container(border=True):
            stat_col1, stat_col2 = st.columns(2)
            
            vol_mean = one_month['Volume'].mean() if 'Volume' in one_month.columns else 0
            vol_display = f"{int(vol_mean):,}" if pd.notnull(vol_mean) else "N/A"

            with stat_col1:
                st.metric("Avg Volume (30)", vol_display)
            with stat_col2:
                st.metric("Highest (30d)", f"{one_month['Close'].max():.2f} €")
                
            st.metric("Lowest (30d)", f"{one_month['Close'].min():.2f} €")

except Exception as e:
    st.error(f"Error loading data for {selected_ticker}.")
    st.info("Error details:")
    st.exception(e)