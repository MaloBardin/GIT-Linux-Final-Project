import streamlit as st
from grabbing_dataframe import GetDfForDashboard, Dfcleaning, ReadDf

col_title, col_button = st.columns([4, 1])
with col_title:
    st.title("Ticker")

with col_button:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Retour au Dashboard"):
        url = "/home_page"
        st.markdown(

            f'<meta http-equiv="refresh" content="0; url={url}" target="_self">',
            unsafe_allow_html=True
        )

st.write("Détails du ticker...")