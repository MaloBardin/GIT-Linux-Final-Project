import streamlit as st
import pandas as pd
import math
from grabbing_dataframe import GetDfForDashboard, Dfcleaning, ReadDf, getDfForGraph

#setup
query_params = st.query_params
st.set_page_config(layout="wide")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")


# graph generation
def generate_sparkline(data):
    if not isinstance(data, list) or len(data) < 2:
        return ""

    width = 100
    height = 30
    min_val = min(data)
    max_val = max(data)
    val_range = max_val - min_val if max_val != min_val else 1

    points = []
    for i, val in enumerate(data):
        x = (i / (len(data) - 1)) * width
        y = height - ((val - min_val) / val_range) * height
        points.append(f"{x},{y}")

    points_str = " ".join(points)
    color = "#29f075" if data[-1] >= data[0] else "#CC4974"

    svg = f"""
    <svg width="{width}" height="{height}" style="background-color:transparent;">
        <polyline points="{points_str}" fill="none" stroke="{color}" stroke-width="2" />
    </svg>
    """
    return svg


#header
df_total = GetDfForDashboard(Dfcleaning(ReadDf()))

df2 = df_total.copy()
df2['Price_Str'] = df2['Price'].apply(lambda x: f"{x:.2f}")
df2['Formatted_Ticker'] = df2['Ticker'] + ' : ' + df2['Price_Str']

message_ban_tickers = ' | '.join(df2['Formatted_Ticker'])
colors = ['#CC4974', '#29f075']  # rouge, vert

formatted_list = [
    f'<span style="color:{colors[i % len(colors)]};">{ticker}</span>'
    for i, ticker in enumerate(df2['Formatted_Ticker'])
]
message_ban_tickers = ' | '.join(formatted_list)
st.markdown(f"""
    <div class="ticker-wrap">
        <div class="ticker-move">
            <div class="ticker-item">{message_ban_tickers}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

#table setup

ROWS_PER_PAGE = 10

if 'page_number' not in st.session_state:
    st.session_state.page_number = 0

start_index = st.session_state.page_number * ROWS_PER_PAGE
end_index = start_index + ROWS_PER_PAGE

df_page = df_total.iloc[start_index:end_index].copy()



#graph data preparation
df_graph = getDfForGraph(Dfcleaning(ReadDf()))

def get_history_list(ticker_name):
    if ticker_name in df_graph.columns:
        prices = df_graph[ticker_name].tolist()
        return prices


df_page['History_List'] = df_page['Ticker'].apply(get_history_list)
df_page['Graph'] = df_page['History_List'].apply(generate_sparkline)



current_rows_on_page = len(df_page)

if current_rows_on_page < ROWS_PER_PAGE:
    rows_to_add = ROWS_PER_PAGE - current_rows_on_page
    empty_df = pd.DataFrame({col: [None] * rows_to_add for col in df_page.columns}) # to be modified bcs none is not the best for display
    df_page = pd.concat([df_page, empty_df], ignore_index=True)


def color_volatility(val):
    if not isinstance(val, (int, float)):
        return ''
    if val < 30:
        color = '#29f075'
    elif val > 50:
        color = '#CC4974'
    else:
        return ''
    return f'color: {color};'


def color_price(val):
    if not isinstance(val, (int, float)):
        return ''
    if val > 0:
        color = '#29f075'
    elif val < 0:
        color = '#CC4974'
    else:
        return ''
    return f'color: {color}; '


def make_clickable(row):
    if not row['Ticker']:
        row['Button'] = ""
        return row
    ticker_value = row['Ticker']
    page_url = "/ticker_page"
    row["Button"] = (
        f'<a target="_self" href="{page_url}?ticker={ticker_value}" '
        f'style="text-decoration: none; color: inherit; font-weight: bold; display: block; padding: 10px;">'
        f'GO'
        f'</a>'
    )
    return row


#table display
cols_order = ['Ticker', 'Price', 'Return_1d', 'Return_7d', 'Return_30d', 'Graph', 'Button']
# On filtre pour ne garder que les colonnes qui existent (sécurité si Return_30d manque)
cols_present = [c for c in cols_order if c in df_page.columns]
df_display = df_page[cols_present].copy()

df_display = df_display.apply(make_clickable, axis=1)

styled_df = df_display.style

#colors
styled_df = styled_df.map(
    color_price,
    subset=['Return_1d', 'Return_7d', 'Return_30d']
)

returnsformat = ['Return_1d', 'Return_7d', 'Return_30d']


styled_df = styled_df.format("{:.2f}%", subset=[c for c in returnsformat if c in df_display.columns], na_rep="")
styled_df = styled_df.format("{:.2f}", subset=['Price'], na_rep="")

styled_df = styled_df.set_table_attributes('class="big-font-table"')
styled_df = styled_df.hide()

html_table = styled_df.to_html(index=False, border=0, escape=False)

st.markdown(html_table, unsafe_allow_html=True)
st.write("")

#nav

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


