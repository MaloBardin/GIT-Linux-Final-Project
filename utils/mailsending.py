import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import streamlit as st
import pandas as pd



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
