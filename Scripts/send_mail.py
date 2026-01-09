import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import pandas as pd
from datetime import datetime
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "data3y.csv")

REPORT_DIR = os.path.join(BASE_DIR,"reports")
SUBSCRIBERS_FILE = "subscribers.txt"

SENDER_EMAIL = "malo.adam.project@gmail.com"
SENDER_PASSWORD = "xqfs qeeh vhzm rqsb"  

os.makedirs(REPORT_DIR, exist_ok=True)

def generate_report():
    try:
        today_date = datetime.now().strftime("%d %B %Y")

        # Read CSV
        df_total = pd.read_csv(DATA_FILE)
        if "Date" in df_total.columns:
            df_total = df_total.set_index("Date")

        tickers_to_display = df_total.columns
        table_data = []
        for ticker in tickers_to_display:
            prices = df_total[ticker].dropna()
            if len(prices) < 31: 
                continue

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

        # Best/Worst performer
        df_returns_1d = df_total.pct_change().iloc[-1]
        best_ticker_name = df_returns_1d.idxmax()
        worst_ticker_name = df_returns_1d.idxmin()
        avg_return = df_returns_1d.mean() * 100
        best_val = df_returns_1d.max() * 100
        worst_val = df_returns_1d.min() * 100

        market_name = '^FCHI'
        if market_name not in df_total.columns: 
            market_name = df_total.columns[0]

        df_last_21 = df_total.tail(21)
        dates_list = list(df_last_21.index.astype(str))

        def normalize(series):
            return (series / series.iloc[0] * 100) - 100

        val_market = normalize(df_last_21[market_name]).fillna(0).tolist()
        val_best = normalize(df_last_21[best_ticker_name]).fillna(0).tolist()
        val_worst = normalize(df_last_21[worst_ticker_name]).fillna(0).tolist()

        # Generate sparkline SVG
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

        # Full HTML content (unchanged for brevity)
        html_content = f"""<!DOCTYPE html><html>...{table_rows_html}...</html>"""

        return html_content

    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")
        traceback.print_exc()
        return None


def send_mail(recipient, subject, body, attachment_path=None):
    try:
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
            part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(attachment_path)}")
            msg.attach(part)

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"[INFO] Email sent to {recipient}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to send email to {recipient}: {e}")
        traceback.print_exc()
        return False


def main():
    report_html = generate_report()
    if report_html is None:
        print("[ERROR] Report generation failed. Exiting.")
        return

    today_str = datetime.now().strftime("%d-%m-%Y")
    report_path = os.path.join(REPORT_DIR, f"report_{today_str}.html")

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        print(f"[INFO] Report saved: {report_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save report: {e}")
        traceback.print_exc()

    subscribers_path = os.path.join(DATA_DIR, SUBSCRIBERS_FILE)
    if os.path.exists(subscribers_path):
        try:
            with open(subscribers_path, "r") as f:
                emails = [e.strip() for e in f if e.strip()]

            subject = "Your daily newsletter from Malo and Adam"
            for email in emails:
                send_mail(email, subject, report_html, report_path)

            print(f"[INFO] Sent report to {len(emails)} subscribers")
        except Exception as e:
            print(f"[ERROR] Failed to read subscribers or send emails: {e}")
            traceback.print_exc()
    else:
        print("[INFO] No subscribers file found.")


if __name__ == "__main__":
    main()
