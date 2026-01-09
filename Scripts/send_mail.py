#!/usr/bin/env python3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import pandas as pd
from datetime import datetime
DATA_DIR = "data"
DATA_FILE = "data3y.csv"
REPORT_DIR = "reports"
SUBSCRIBERS_FILE = "subscribers.txt"

SENDER_EMAIL = "malo.adam.project@gmail.com"
SENDER_PASSWORD = "xqfs qeeh vhzm rqsb"  

os.makedirs(REPORT_DIR, exist_ok=True)


def generate_report():
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



def send_mail(recipient, subject, body, attachment_path):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html'))

    if attachment_path:
        with open(attachment_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {os.path.basename(attachment_path)}",
        )
        msg.attach(part)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email to {recipient}: {e}")
        return False

def main():
    report_html = generate_report()
    today_str = datetime.now().strftime("%d-%m-%Y")
    report_path = os.path.join(REPORT_DIR, f"report_{today_str}.html")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Report saved: {report_path}")

    if os.path.exists(os.path.join(DATA_DIR, SUBSCRIBERS_FILE)):
        with open(os.path.join(DATA_DIR, SUBSCRIBERS_FILE), "r") as f:
            emails = [e.strip() for e in f if e.strip()]
        
        subject = "Your daily newsletter from Malo and Adam"
        
        for email in emails:
            send_mail(email, subject, report_html, report_path)
        
        print(f"Sent report to {len(emails)} subscribers")
    else:
        print("No subscribers file found.")

if __name__ == "__main__":
    main()
