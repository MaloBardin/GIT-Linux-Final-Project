import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import streamlit as st
import pandas as pd


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



def callforlinux():
    report = generatereport()
    from datetime import datetime
    date_str = datetime.now().strftime("%d-%m-%Y")
    full_path = os.path.join("reports", f"report of {date_str}.html")
    with open(full_path, "w", encoding="utf-8") as file:
        file.write(report)

    body=report
    subject="Your daily newsletter from Malo and Adam"
    try:
        with open("subscribers.txt", "r") as file:
            emails = file.readlines()

        for email in emails:
            clean_email = email.strip()
            sendmail(clean_email, subject, body)
    except Exception as e:
        print("Error when sending emails", e)



def sendmail(mail, object, body):
    sender_email = "malo.adam.project@gmail.com"
    #password = "GitLinuxIsTheEldorado" #like our password x)
    password="xqfs qeeh vhzm rqsb"


    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = mail
    msg['Subject'] = object

    msg.attach(MIMEText(body, 'html')) #swap sur html quand les tests seront ok @adam
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"failed to send : {e}")
        return False





def bulknl(object, text):
    file_path = "subscribers.txt"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            emails = f.readlines()

        count = 0
        for email in emails:
            email_clean = email.strip()
            if email_clean:
                succes = bulknl(email_clean, object, text)
                if succes:
                    count += 1

        return f"Sended to {count} subscribers."
    else:
        return "No subscribers found lmaooo"


@st.dialog("📬 Keep track of the market")
def show_newsletter_popup():
    st.write(
        "Join our mailing list to receive daily portfolio reports directly in your inbox.")

    with st.form("newsletter_form"):
        email = st.text_input("Enter your email address", placeholder="malo@adam.fr")
        submit_btn = st.form_submit_button("Subscribe Now")

        if submit_btn:
            if email and "@" in email:
                file_path = "subscribers.txt"

                email_exists = False
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        if email in f.read():
                            email_exists = True

                if not email_exists:
                    with open(file_path, "a") as f:
                        f.write(f"{email}\n")
                    st.success("Success! You'll receive our next daily report tomorrow, see you then !")

                else:
                    st.warning("You are already subscribed!")
            else:
                st.error("Please enter a valid email address.")
