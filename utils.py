import streamlit as st
import os
import datetime
from grabbing_dataframe import UpdateDfMax

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def barre_menu():
    col1, col2, col3, col4, col6, col7 = st.columns([1, 1, 1, 3, 2, 1])
    
    with col1:
        st.page_link("pages/home_page.py", label="Dashboard", use_container_width=True)
    with col2:
        st.page_link("pages/quant_A_page.py", label="Single Asset", use_container_width=True)
    with col3:
        st.page_link("pages/pflanding.py", label="Portfolio simulation", use_container_width=True)

    with col6:
        if st.button("📩 Daily Report", use_container_width=True):
            show_newsletter_popup()
            
    with col7:
        if os.path.exists('cac40_history.csv'):
            file_time = os.path.getmtime('cac40_history.csv')
            last_date = datetime.datetime.fromtimestamp(file_time).strftime('%d/%m à %H:%M')
            last_update_msg = f"Last Update : {last_date}"
        else:
            last_update_msg = "Aucune donnée"

        if st.button("🔄 Reload Data", help=last_update_msg, use_container_width=True):
            with st.spinner(''):
                UpdateDfMax()
                st.toast("Data reloaded with success !", icon="✅")
                import time
                time.sleep(0.5)
                st.rerun()

