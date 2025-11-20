# streamlit_app.py
import streamlit as st
st.set_page_config(layout="wide")
page_cible_url = "/home_page"

st.markdown(
    f'<meta http-equiv="refresh" content="0; url={page_cible_url}" target="_self">',
    unsafe_allow_html=True
)

st.stop()