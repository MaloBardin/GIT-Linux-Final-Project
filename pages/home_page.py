import streamlit as st
import pandas as pd
import math
from grabbing_dataframe import GetDfForDashboard, Dfcleaning, ReadDf, getDfForGraph, GetDf
from pflanding import show_newsletter_popup

#setup
query_params = st.query_params
st.set_page_config(layout="wide")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")



def barre_menu(): #navigation bar
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


#newsltter popup




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


#html for the best and worst stock of the day !
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

#parameters for search and sort
col_search, col_sort = st.columns([3, 1])

with col_search:
    search_query = st.text_input("🔍 Search Ticker", placeholder="🔍 Search Ticker", label_visibility="collapsed")

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

#navigation for the pages of the table
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


def color_volatility(val): #color of the vol
    if not isinstance(val, (int, float)):
        return ''
    if val < 30:
        color = '#29f075'
    elif val > 50:
        color = '#CC4974'
    else:
        return ''
    return f'color: {color};'


def color_price(val): #swap the color if negative or positive return
    if not isinstance(val, (int, float)):
        return ''
    if val > 0:
        color = '#29f075'
    elif val < 0:
        color = '#CC4974'
    else:
        return ''
    return f'color: {color}; '


def make_clickable(row): #make the button clickable to be able to open the associated page
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


#BIG BIG TABLE DISPLAY
cols_order = ['Ticker', 'Price', 'Return_1d', 'Return_7d', 'Return_30d', 'Graph', 'Button']
cols_present = [c for c in cols_order if c in df_page.columns]
df_display = df_page[cols_present].copy()

df_display = df_display.apply(make_clickable, axis=1)

styled_df = df_display.style

styled_df = styled_df.map(
    color_price,
    subset=['Return_1d', 'Return_7d', 'Return_30d']
)
returnsformat = ['Return_1d', 'Return_7d', 'Return_30d']

styled_df = styled_df.format(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else "", subset=[c for c in returnsformat if c in df_display.columns])
styled_df = styled_df.format(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "", subset=['Price'])

styled_df = styled_df.set_table_attributes('class="big-font-table"')
styled_df = styled_df.hide()

html_table = styled_df.to_html(index=False, border=0, escape=False)

with table_container:
    st.markdown(html_table, unsafe_allow_html=True)
    st.write("")

#REPORT GENERATION, WE DID IT HERE SINCE WE HAVE ALL THE DATA LOADED ALREADY
def generatereport():
    from datetime import datetime
    #logic part
    today_date = datetime.now().strftime("%d %B %Y")

    #read csv
    df_total = pd.read_csv("data3y.csv")

    if "Date" in df_total.columns:
        df_total = df_total.set_index("Date")

    #load the same table on one page and remove go button
    tickers_to_display = df_total.columns
    table_data = []
    for ticker in tickers_to_display:
        prices = df_total[ticker].dropna()
        if len(prices) < 31: continue

        current_price = prices.iloc[-1]
        prev_price_1d = prices.iloc[-2]
        prev_price_7d = prices.iloc[-8] if len(prices) >= 8 else prices.iloc[0]
        prev_price_30d = prices.iloc[-31] if len(prices) >= 31 else prices.iloc[0]

        ret_1d = ((current_price / prev_price_1d) - 1) * 100
        ret_7d = ((current_price / prev_price_7d) - 1) * 100
        ret_30d = ((current_price / prev_price_30d) - 1) * 100

        history_30d = prices.tail(30).tolist()

        table_data.append({
            "Ticker": ticker,
            "Price": current_price,
            "Return_1d": ret_1d,
            "Return_7d": ret_7d,
            "Return_30d": ret_30d,
            "History": history_30d
        })

    #display the best and worst performer of today and a graph data
    df_returns_1d = df_total.pct_change().iloc[-1]
    best_ticker_name = df_returns_1d.idxmax()
    worst_ticker_name = df_returns_1d.idxmin()
    avg_return = df_returns_1d.mean() * 100

    best_val = df_returns_1d.max() * 100
    worst_val = df_returns_1d.min() * 100

    market_name = '^FCHI'
    if market_name not in df_total.columns: market_name = df_total.columns[0]

    df_last_21 = df_total.tail(21)
    dates_list = list(df_last_21.index.astype(str))

    def normalize(series):
        return (series / series.iloc[0] * 100) - 100

    val_market = normalize(df_last_21[market_name]).fillna(0).tolist()
    val_best = normalize(df_last_21[best_ticker_name]).fillna(0).tolist()
    val_worst = normalize(df_last_21[worst_ticker_name]).fillna(0).tolist()

    #html part, we used AI assistance to generate the html structure and style since we are not experts in web design
    def make_sparkline_svg(data):
        if not data: return ""
        width, height = 100, 30
        min_v, max_v = min(data), max(data)
        rng = max_v - min_v if max_v != min_v else 1

        points = []
        for i, val in enumerate(data):
            x = (i / (len(data) - 1)) * width
            y = height - ((val - min_v) / rng) * height
            points.append(f"{x},{y}")

        color = "#29f075" if data[-1] >= data[0] else "#CC4974"
        return f'<svg width="{width}" height="{height}" style="background:transparent"><polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/></svg>'


    table_rows_html = ""
    for row in table_data:
        c_1d = "#29f075" if row['Return_1d'] >= 0 else "#CC4974"
        c_7d = "#29f075" if row['Return_7d'] >= 0 else "#CC4974"
        c_30d = "#29f075" if row['Return_30d'] >= 0 else "#CC4974"

        sparkline_svg = make_sparkline_svg(row['History'])

        table_rows_html += f"""
        <tr>
            <td style="font-weight:bold; color: #2d2d2d;">{row['Ticker']}</td>
            <td>{row['Price']:.2f}</td>
            <td style="color:{c_1d}">{row['Return_1d']:.2f}%</td>
            <td style="color:{c_7d}">{row['Return_7d']:.2f}%</td>
            <td style="color:{c_30d}">{row['Return_30d']:.2f}%</td>
            <td>{sparkline_svg}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
    <style>
    body {{ 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        padding: 40px; 
        background-color: #ffffff;
        color: #4a4a4a;
        margin: 0 auto;
        max-width: 1400px;
        display: flex;
        flex-direction: column;
    }}

    .header {{ text-align: center; margin-bottom: 40px; }}
    .title {{ font-family: 'Playfair Display', serif; font-size: 36px; font-weight: 600; color: #2d2d2d; margin: 0; }}
    .date {{ font-size: 16px; color: #888; margin-top: 5px; }}

    .metric-container {{ display: flex; gap: 20px; justify-content: space-between; margin-bottom: 40px; }}
    .metric-card {{ background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px; padding: 25px; text-align: center; flex: 1; min-width: 200px; box-shadow: 0 2px 4px rgba(0,0,0,0.03); }}
    .metric-label {{ font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }}
    .metric-ticker {{ font-size: 26px; font-weight: 700; color: #2d2d2d; margin-bottom: 5px; }}
    .metric-pos {{ color: #29f075; font-weight: 600; font-size: 18px; }}
    .metric-neg {{ color: #CC4974; font-weight: 600; font-size: 18px; }}

    .chart-wrapper {{ width: 100%; height: 400px; margin-bottom: 60px; }}

    /* CORRECTION 3 : TABLEAU STYLE CLAIR */
    .table-section {{
        background-color: #ffffff;
        padding: 0px; /* Moins de padding pour fondre dans la page */
        border-radius: 12px;
        color: #4a4a4a;
        /* Pas de box-shadow pour un look plus plat ou léger */
        border: 1px solid #f0f0f0;
    }}

    .table-title {{ font-size: 24px; font-weight: 600; margin-bottom: 20px; color: #2d2d2d; padding: 20px 20px 0 20px; }}

    table {{ width: 100%; border-collapse: collapse; }}

    th {{ 
        text-align: left; 
        color: #888; 
        font-size: 12px; 
        text-transform: uppercase; 
        padding: 15px 20px; 
        border-bottom: 2px solid #f0f0f0; /* Bordure légère */
    }}

    td {{ 
        padding: 12px 20px; 
        border-bottom: 1px solid #f8f9fa; /* Séparation très subtile */
        font-size: 15px; 
        vertical-align: middle; 
    }}

    /* Hover effect sur les lignes pour la lisibilité */
    tr:hover {{ background-color: #fcfcfc; }}

    </style>
    </head>
    <body>

        <div class="header">
            <h1 class="title">Daily Market Report</h1>
            <div class="date">{today_date}</div>
        </div>

        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">🔥 Top Performer</div>
                <div class="metric-ticker">{best_ticker_name}</div>
                <div class="metric-pos">+{best_val:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🧊 Worst Performer</div>
                <div class="metric-ticker">{worst_ticker_name}</div>
                <div class="metric-neg">{worst_val:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">📊 Market Avg</div>
                <div class="metric-ticker">{avg_return:.2f}%</div>
                <div class="{'metric-pos' if avg_return >= 0 else 'metric-neg'}">Global Trend</div>
            </div>
        </div>

        <div class="chart-wrapper">
            <canvas id="performanceChart"></canvas>
        </div>

        <div class="table-section">
            <div class="table-title">Market Overview</div>
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Price</th>
                        <th>1 Day</th>
                        <th>7 Days</th>
                        <th>30 Days</th>
                        <th>Trend (30d)</th>
                        </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            const ctx = document.getElementById('performanceChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {dates_list}, 
                    datasets: [
                        {{ label: '{best_ticker_name}', data: {val_best}, borderColor: '#29f075', borderWidth: 2, pointRadius: 0, tension: 0.3 }},
                        {{ label: '{market_name}', data: {val_market}, borderColor: '#4a4a4a', borderWidth: 2, borderDash: [5, 5], pointRadius: 0, tension: 0.3 }},
                        {{ label: '{worst_ticker_name}', data: {val_worst}, borderColor: '#CC4974', borderWidth: 2, pointRadius: 0, tension: 0.3 }}
                    ]
                }},
                options: {{
                    animation: false,
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{ mode: 'index', intersect: false }},
                    plugins: {{ 
                        title: {{ display: true, text: 'Relative Performance (21 Days)', padding: 20, font: {{size: 16}} }},
                        legend: {{ position: 'top', align: 'end' }}
                    }},
                    scales: {{ 
                        x: {{ display: false }}, 
                        y: {{ ticks: {{ callback: v => v + '%' }} }} 
                    }}
                }}
            }});
        </script>

    </body>
    </html>
    """
    return html_content



 # to test
#st.download_button(label="bouton test",data=generatereport(),file_name="market_summary.html",mime="text/html")