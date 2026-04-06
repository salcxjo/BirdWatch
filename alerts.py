# alerts.py — BirdWatch
# Email alerts for specific species detections.
# Configure via .env file — see README for setup.

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/BirdWatch/.env"))

SMTP_HOST  = "smtp.gmail.com"
SMTP_PORT  = 587
EMAIL_USER = os.environ.get("BIRDWATCH_EMAIL")
EMAIL_PASS = os.environ.get("BIRDWATCH_EMAIL_PASS")
EMAIL_TO   = os.environ.get("BIRDWATCH_EMAIL_TO", EMAIL_USER)

# Species to alert on — empty set means alert on everything non-Unknown
# Add scientific names to restrict alerts to specific species
ALERT_SPECIES = set()

ALERT_COOLDOWN_MINUTES = 30
_last_alert = {}

def should_alert(species):
    if not ALERT_SPECIES or species in ALERT_SPECIES:
        last = _last_alert.get(species)
        if last is None or datetime.now() - last > timedelta(minutes=ALERT_COOLDOWN_MINUTES):
            return True
    return False

def send_alert(species, confidence, source, image_path):
    if not EMAIL_USER or not EMAIL_PASS:
        return
    if not should_alert(species):
        return

    _last_alert[species] = datetime.now()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    msg = MIMEMultipart('related')
    msg['Subject'] = f"BirdWatch: {species} spotted!"
    msg['From']    = EMAIL_USER
    msg['To']      = EMAIL_TO

    html = f"""
    <h2>🐦 {species} spotted on your balcony</h2>
    <p><strong>Time:</strong> {ts}</p>
    <p><strong>Confidence:</strong> {confidence:.1%} ({source})</p>
    <img src="cid:birdimage" style="max-width:600px; border-radius:8px;">
    """
    msg.attach(MIMEText(html, 'html'))

    if image_path and os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-ID', '<birdimage>')
            msg.attach(img)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(EMAIL_USER, EMAIL_PASS)
            s.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())
        print(f"Alert sent: {species}")
    except Exception as e:
        print(f"Email alert failed: {e}")
