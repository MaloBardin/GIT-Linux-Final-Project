import streamlit as st
import pandas as pd
import math
from grabbing_dataframe import GetDfForDashboard, Dfcleaning, ReadDf, getDfForGraph, GetDf

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
formatted_list = []

for index, row in df2.iterrows():
    ticker = row['Ticker']
    price = row['Price'] 
    change = row['Return_1d']

    if change >= 0:
        color = '#4caf50' 
        arrow = '▲'
    else:
        color = '#f44336'
        arrow = '▼'

    html_str = f"""
    <span style='font-weight:bold; color:#ffffff;'>{ticker}</span> 
    <span style='color:#e0e0e0;'>${price:.2f}</span> 
    <span style='color:{color};'>{arrow} {abs(change):.2f}%</span>
    """
    formatted_list.append(html_str)

message_ban_tickers = "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(formatted_list)

st.markdown(f"""
    <div class="ticker-wrap">
        <div class="ticker-move">
            <div class="ticker-item">{message_ban_tickers}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

#table setup
best_performer = df_total.loc[df_total['Return_1d'].idxmax()]
worst_performer = df_total.loc[df_total['Return_1d'].idxmin()]

avg_return = df_total['Return_1d'].mean()
avg_color = "metric-delta-pos" if avg_return >= 0 else "metric-delta-neg"

st.markdown(f"""
<div class="metric-container">
    <div class="metric-card">
        <div class="metric-title">🔥 Top Performer</div>
        <div class="metric-value">{best_performer['Ticker']}</div>
        <div class="metric-delta-pos">+{best_performer['Return_1d']:.2f}%</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">🧊 Worst Performer</div>
        <div class="metric-value">{worst_performer['Ticker']}</div>
        <div class="metric-delta-neg">{worst_performer['Return_1d']:.2f}%</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">📊 Market Avg</div>
        <div class="metric-value">{avg_return:.2f}%</div>
        <div class="{avg_color}">Global Trend</div>
    </div>
</div>
""", unsafe_allow_html=True)


col_search, col_sort = st.columns([3, 1])

with col_search:
    search_query = st.text_input("🔍 Search Ticker", placeholder="...", label_visibility="collapsed")

with col_sort:
    sort_option = st.selectbox("Sort by", ["Price Desc", "Price Asc", "Return 1D Desc", "Return 1D Asc" ], label_visibility="collapsed")

if search_query:
    df_total = df_total[df_total['Ticker'].str.contains(search_query.upper(), na=False)]

if sort_option == "Return 1D Desc":
    df_total = df_total.sort_values(by="Return_1d", ascending=False)
elif sort_option == "Return 1D Asc":
    df_total = df_total.sort_values(by="Return_1d", ascending=True)
elif sort_option == "Price Desc":
    df_total = df_total.sort_values(by="Price", ascending=False)
else:
    df_total = df_total.sort_values(by="Price", ascending=True)

if search_query or sort_option:
    if 'last_search' not in st.session_state or st.session_state.last_search != search_query:
        st.session_state.page_number = 0
        st.session_state.last_search = search_query



ROWS_PER_PAGE = 10

if 'page_number' not in st.session_state:
    st.session_state.page_number = 0

total_pages = math.ceil(len(df_total) / ROWS_PER_PAGE)
current_page = st.session_state.page_number + 1


table_container = st.container(border=True)


col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("Previous", disabled=(st.session_state.page_number == 0), use_container_width=True):
        st.session_state.page_number -= 1
        st.rerun()

with col2:
    st.markdown(
        f"<div style='text-align: center; margin-top: 10px;'>"
        f"Page <b>{current_page}</b> of <b>{total_pages}</b>"
        f"</div>",
        unsafe_allow_html=True
    )

with col3:
    if st.button("Next", disabled=(current_page >= total_pages), use_container_width=True):
        st.session_state.page_number += 1
        st.rerun()


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
    empty_df = pd.DataFrame({col: [""] * rows_to_add for col in df_page.columns}) 
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


#table display logic
cols_order = ['Ticker', 'Price', 'Return_1d', 'Return_7d', 'Return_30d', 'Graph', 'Button']
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

#We replace NaN by empty string for better display
styled_df = styled_df.format(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else "", subset=[c for c in returnsformat if c in df_display.columns])
styled_df = styled_df.format(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "", subset=['Price'])

styled_df = styled_df.set_table_attributes('class="big-font-table"')
styled_df = styled_df.hide()

html_table = styled_df.to_html(index=False, border=0, escape=False)

with table_container:
    st.markdown(html_table, unsafe_allow_html=True)
    st.write("")