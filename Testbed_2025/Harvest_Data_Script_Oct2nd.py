from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd
import re
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from datetime import timedelta
import numpy as np
import smtplib
import traceback
import os.path
import base64
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds_new.txt")

if gauth.credentials is None:
    gauth.CommandLineAuth()  # 👈 Use this instead of LocalWebserverAuth
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()

gauth.SaveCredentialsFile("mycreds_new.txt")
drive = GoogleDrive(gauth)

# folder ID to connect with the folder inside the drive
folder_id = '1klYbnCcTR_SFsC1nEmRKP2jNIoLnlKGY'

# List all files in the folder
file_list = drive.ListFile({
    'q': f"'{folder_id}' in parents and trashed=false"
}).GetList()

for file in file_list:
    print(f"Title: {file['title']}, ID: {file['id']}")

# === Step 2: Get all files in the folder ===
file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

# === Step 3: Setup ===
Harvest_files = []
Emlid_PT_Intergrated_files = []
DryWt_files = []

pattern_Harvest = re.compile(r'^Harvest_(\d+\.\d+\.\d+)\.csv$')
pattern_Emlid_PT_Intergrated = re.compile(r'^Emlid_PT_Intergrated_(\d+\.\d+\.\d+)\.csv$')
pattern_DryWt = re.compile(r'^DryWt_(\d+\.\d+\.\d+)\.csv$')

# === Function to sort files by date ===
def extract_date_key(filename):
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', filename)
    if match:
        month, day, year = map(int, match.groups())
        return (year, month, day)
    return (0, 0, 0)

# === Step 4: Classify files ===
for file in file_list:
    title = file['title']
    if pattern_Harvest.match(title):
        Harvest_files.append((title, file['id']))
    elif pattern_Emlid_PT_Intergrated.match(title):
        Emlid_PT_Intergrated_files.append((title, file['id']))
    elif pattern_DryWt.match(title):
        DryWt_files.append((title, file['id']))

# === Step 5-7: Download latest files ===
if Harvest_files:
    latest_Harvest = max(Harvest_files, key=lambda x: extract_date_key(x[0]))
    print(f"📥 Downloading Harvest file: {latest_Harvest[0]}")
    file = drive.CreateFile({'id': latest_Harvest[1]})
    file.GetContentFile(latest_Harvest[0])
else:
    print("❌ No Harvest files found.")

if Emlid_PT_Intergrated_files:
    latest_Emlid = max(Emlid_PT_Intergrated_files, key=lambda x: extract_date_key(x[0]))
    print(f"📥 Downloading Emlid_PT_Intergrated file: {latest_Emlid[0]}")
    file = drive.CreateFile({'id': latest_Emlid[1]})
    file.GetContentFile(latest_Emlid[0])
else:
    print("❌ No Emlid_PT_Intergrated files found.")

if DryWt_files:
    latest_DryWt = max(DryWt_files, key=lambda x: extract_date_key(x[0]))
    print(f"📥 Downloading DryWt file: {latest_DryWt[0]}")
    file = drive.CreateFile({'id': latest_DryWt[1]})
    file.GetContentFile(latest_DryWt[0])
else:
    print("❌ No DryWt files found.")

df = pd.read_csv("Harvest_10.02.24.csv")

# Check for negative values in the 'weight' column
negative_values = df[df['Weight in kg'] < 0]

# Drop rows where 'weight' is negative
df = df[df['Weight in kg'] >= 0]
del df['Units']

# Filter rows where Farm (Plot) is 14 and Paddock is 1
filtered_data = df[(df['Farm'] == 14) & (df['Paddock'] == 1)]
# Change the 'Strip' value from 100 to 10
df.loc[df['Strip'] == 100, 'Strip'] = 10
df[df['Strip'] == 10]

# Create a new column 'Plot' by merging 'Farm' and 'Paddock' columns
df['Plot'] = df['Farm'].astype(str) + '.' + df['Paddock'].astype(str)

dh = pd.read_csv("Emlid_PT_Intergrated_10.02.24.csv")
# Initialize plots and strips
plots = ["14.1", "14.2", "14.3", "14.4", "28.1", "28.2", "28.3", "28.4"]
strips = list(range(1, 11))  # Strips 1 to 10

# Assign plot and strip labels
plot_labels = []
strip_labels = []

for plot in plots:
    plot_labels.extend([plot] * len(strips))
    strip_labels.extend(strips)

# Truncate the labels to match the number of rows in the dataset
dh["Plot"] = plot_labels[:len(dh)]
dh["Strip"] = strip_labels[:len(dh)]

# Merge the two datasets
data = pd.merge(
    df,
    dh,
    left_on=["Plot", "Strip"],
    right_on=["Plot", "Strip"],
    how="left"
)

# Drop the redundant 'Plot' and 'Unnamed: 0' columns
data.drop(columns=["Unnamed: 0"], inplace=True)

# Rename height column
data.rename(columns={"Average Height": "PT Height (mm)"}, inplace=True)

dw = pd.read_csv("DryWt_10.02.24.csv")

# Ensure 'Plot' columns are of the same type (e.g., convert both to string)
dw['Plot'] = dw['Plot'].astype(str)
data['Plot'] = data['Plot'].astype(str)


# Merging datasets based on Plot and Strip
ds = pd.merge(
    data,
    dw,
    on=["Plot", "Strip"],
    how="inner"
)

# Dropping the specified columns
ds = ds.drop(columns=["Segment", "date", "NaN"], errors="ignore")

# Rename the column
ds.rename(columns={"Wet wt. (g)": "Sub Wet Wt (g)"}, inplace=True)

# Rename the column
ds.rename(columns={"Dry  wt. (g)": "Sub Dry Wt (g)"}, inplace=True)

# Insert the Width (m) column right after the Length in meters column
ds.insert(ds.columns.get_loc("Length in meters") + 1, "Width (m)", 0.8128)

# Calculate the area
ds["Area (m²)"] = ds["Length in meters"] * ds["Width (m)"]

# Insert the Area (m²) column right after the Width (m) column
ds.insert(ds.columns.get_loc("Width (m)") + 1, "Area (m²)", ds.pop("Area (m²)"))

# Calculate the Dry Matter column
ds["Dry Matter"] = ds["Sub Dry Wt (g)"] / ds["Sub Wet Wt (g)"]

# Calculate biomass using the formula
ds["Biomass (kg/ha)"] = (ds["Weight in kg"] * ds["Dry Matter"]) / ds["Area (m²)"] * 10000

# Add the Residual Dry Wt (kg/ha) column with a constant value of 950
ds["Residual Dry Wt (kg/ha)"] = 950

# Calculate Biomass by adding Yield (Biomass (kg/ha)) and Residue (Residual Dry Wt (kg/ha))
ds["Total Biomass (kg/ha)"] = ds["Biomass (kg/ha)"] + ds["Residual Dry Wt (kg/ha)"]
ds = ds.dropna(axis=1, how='all')
ds.to_csv("Yield_10.02.2024.csv")

# === Upload final Yield file to Google Drive ===
yield_filename = "Yield_10.02.2024.csv"

upload_file = drive.CreateFile({'title': yield_filename, 'parents': [{'id': folder_id}]})
upload_file.SetContentFile(yield_filename)
upload_file.Upload()
print(f"✅ Yield file uploaded to Google Drive as: {yield_filename}")

# === Gmail API setup ===
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def gmail_authenticate():
    """Authenticate using Gmail API and return service"""
    creds = None
    token_file = 'token_gmail.json'
    creds_file = 'credentials_gmail.json'  # Custom-named Gmail credentials

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def send_email_gmail_api(subject, body_text, to_emails, attachment_path=None):
    """Send email with optional attachment using Gmail API"""
    service = gmail_authenticate()

    msg = EmailMessage()
    msg.set_content(body_text)
    msg['To'] = ", ".join(to_emails)
    msg['From'] = "me"
    msg['Subject'] = subject

    if attachment_path:
        try:
            with open(attachment_path, 'rb') as f:
                file_data = f.read()
                file_name = os.path.basename(attachment_path)
            msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
        except Exception as file_err:
            print(f"⚠️ Failed to attach file: {file_err}")
            msg.set_content(f"{body_text}\n\n⚠️ Failed to attach file: {file_err}")

    encoded_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    send_message = {'raw': encoded_message}

    try:
        service.users().messages().send(userId="me", body=send_message).execute()
        print("✅ Email sent via Gmail API!")
    except Exception as e:
        print("❌ Failed to send email using Gmail API.")
        print(e)

# === Send Yield CSV via Email ===
receiver_emails = [
    'darapanenivandana1199@gmail.com',
    'vdzfb@missouri.edu',
    "bernardocandido@missouri.edu"

]

subject_success = '✅ Yield Data - Final Output CSV File'
body_success = 'Hi Team,\n\nPlease find the final Yield CSV output file attached.\n\nRegards,\nAutomated System'

try:
    send_email_gmail_api(subject_success, body_success, receiver_emails, attachment_path=yield_filename)
except Exception as e:
    error_trace = traceback.format_exc()
    body_failure = f"❌ Script execution failed with error:\n\n{error_trace}"
    send_email_gmail_api('❌ Yield Script Failed', body_failure, receiver_emails)





