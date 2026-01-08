import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os


def sendmail(mail, object, body):
    sender_email = "malo.adam.project@gmail.com"
    #password = "GitLinuxIsTheEldorado" #like our password x)
    password="xqfs qeeh vhzm rqsb"


    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = mail
    msg['Subject'] = object

    msg.attach(MIMEText(body, 'plain'))
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