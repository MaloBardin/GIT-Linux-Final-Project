import streamlit as st
from grabbing_dataframe import GetDfForDashboard, Dfcleaning, ReadDf, getInfoperTicker
df_total = Dfcleaning(ReadDf())
query_params = st.experimental_get_query_params()

ticker = query_params.get("ticker", [""])[0]
print(ticker)
short_df, sevendays_data, onemonth_data, isoverbuying = getInfoperTicker(df_total, ticker)

#short_df.columns = short_df.columns.droplevel(1)
sevendays_data.columns = sevendays_data.columns.droplevel(1)
onemonth_data.columns = onemonth_data.columns.droplevel(1)
print(short_df)

st.line_chart(short_df['Price'])
st.bar_chart(short_df['Volume'])




