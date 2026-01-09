import streamlit as st
import os
import datetime
from Scripts.reload_dataset import reload_all_data, get_last_updated
from mailsending import show_newsletter_popup
import time

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def barre_menu():
    col1, col2, col3, col4, col6, col7 = st.columns([1, 1, 1, 3, 2, 1])
    
    with col1:
        st.page_link("pages/home_page.py", label="DASHBOARD", width='stretch')
    with col2:
        st.page_link("pages/quant_A_page.py", label="SINGLE ASSET", width='stretch')
    with col3:
        st.page_link("pages/pflanding.py", label="PORTFOLIO SIMULATION", width='stretch')

    with col6:
        if st.button("📩 DAILY REPORT", width='stretch'):
            show_newsletter_popup()
            
    with col7:
        last_update_msg = f"Last Update : {get_last_updated()}"
        if st.button("🔄 Reload Data", help=last_update_msg, width='stretch'):
            with st.spinner(''):
                reload_all_data()
                st.toast("Data reloaded with success !", icon="✅")
                
                time.sleep(0.5)
                st.rerun()

