import streamlit as st
import pandas as pd
import math

query_params = st.query_params
st.set_page_config(layout="wide")
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")

#HEADER
message_ban = "test"

st.markdown(f"""
    <div class="ticker-wrap">
        <div class="ticker-move">
            <div class="ticker-item">{message_ban}</div>
        </div>
    </div>
""", unsafe_allow_html=True)


#TABLE
data = {
    "Ticker": ["AAPL", "TSLA", "MSFT", "BTC-USD", "ETH-USD"],
    "Prix": [182.5, 245.3, 330.8, 45000, 2400],
    "Volatilité (%)": [22.1, 48.3, 19.5, 65.2, 72.0],
    "Volume (M)": [98, 120, 76, 35, 28],
    "Beta": [1.10, 1.85, 0.95, 0.50, 0.70],
    "Market Cap (B$)": [2900, 780, 2500, 880, 360]
}

df_total = pd.DataFrame(data)
ROWS_PER_PAGE = 10

if 'page_number' not in st.session_state:
    st.session_state.page_number = 0

start_index = st.session_state.page_number * ROWS_PER_PAGE
end_index = start_index + ROWS_PER_PAGE

df_page = df_total.iloc[start_index:end_index].copy()
current_rows_on_page = len(df_page)

if current_rows_on_page < ROWS_PER_PAGE:
    rows_to_add = ROWS_PER_PAGE - current_rows_on_page
    empty_df = pd.DataFrame({col: [""] * rows_to_add for col in df_total.columns})
    df_page = pd.concat([df_page, empty_df], ignore_index=True)


def color_volatility(val):
    if not isinstance(val, (int, float)):
        return ''
    if val < 30:
        color = 'green'
    elif val > 50:
        color = 'red'
    else:
        return ''
    return f'color: {color};'

def color_price(val):
    color = 'green'
    #A modifier
    if not isinstance(val, (int, float)):
        return ''
    if val > 300:
        color = 'green'
    elif val < 300:
        color = 'red'
    else:
        return ''
    return f'color: {color}; '


def make_clickable(row):
    if not row['Ticker']:
        row['Button'] = ""
        return row
    ticker_value = row['Ticker']
    page_url = "/Ticker_Page"
    row["Button"] = (
        f'<a target="_self" href="{page_url}?ticker={ticker_value}" '
        f'style="text-decoration: none; color: inherit; font-weight: bold; display: block; padding: 10px;">'
        f'GO'
        f'</a>'
    )
    return row

df_display = df_page.copy()
df_display = df_display.apply(make_clickable, axis=1)

print(df_display)

styled_df = df_display.style
styled_df = styled_df.map(
    color_volatility,
    subset=['Volatilité (%)']
).map(
    color_price,
    subset=['Prix']
)
styled_df = styled_df.set_table_attributes('class="big-font-table"')

styled_df = styled_df.hide()
html_table = styled_df.to_html(index=False, border=0)

st.markdown(html_table, unsafe_allow_html=True)
st.write("")

total_pages = math.ceil(len(df_total) / ROWS_PER_PAGE)
current_page = st.session_state.page_number + 1

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("Previous", disabled=(st.session_state.page_number == 0), use_container_width=True):
        st.session_state.page_number -= 1
        st.rerun()

with col2:
    st.markdown(
        f"<div style='text-align: center; margin-top: 10px;'>"
        f"Page <b>{current_page}</b> sur <b>{total_pages}</b>"
        f"</div>",
        unsafe_allow_html=True
    )

with col3:
    if st.button("Next", disabled=(current_page >= total_pages), use_container_width=True):
        st.session_state.page_number += 1
        st.rerun()