from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd
import re
import os
import traceback
from datetime import datetime
import base64
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# === Authenticate with Google Drive ===
gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds_new.txt")

if gauth.credentials is None:
    gauth.CommandLineAuth()
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()

gauth.SaveCredentialsFile("mycreds_new.txt")
drive = GoogleDrive(gauth)

# === Folder IDs ===
folder_1_id = '1egHA55KaDp-jj2AsHymVUKarBpdq9xGB'  # Harvest
folder_2_id = '1o0BNrGeds912a2bdJkmUIwvWZPAaPS1J'  # Emlid_PT_Intergrated
folder_3_id = '1vJnBWKCWhUpJNWL3UVA5aJPH5-icfW77'  # Final output (Yield)

# === List files ===
harvest_files = drive.ListFile({'q': f"'{folder_1_id}' in parents and trashed=false"}).GetList()
pt_files = drive.ListFile({'q': f"'{folder_2_id}' in parents and trashed=false"}).GetList()

print("📂 Harvest Folder Files:")
for file in harvest_files:
    print(f"Title: {file['title']}, ID: {file['id']}")

print("\n📂 Emlid_PT_Intergrated Folder Files:")
for file in pt_files:
    print(f"Title: {file['title']}, ID: {file['id']}")

# === Patterns and sorting ===
def extract_date_key(filename):
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', filename)
    if match:
        month, day, year = map(int, match.groups())
        return (year, month, day)
    return (0, 0, 0)

harvest_files = [(f['title'], f['id']) for f in harvest_files if re.match(r'^Harvest_(\d+\.\d+\.\d+)\.csv$', f['title'])]
pt_files = [(f['title'], f['id']) for f in pt_files if re.match(r'^Emlid_PT_Intergrated_(\d+\.\d+\.\d+)\.csv$', f['title'])]

# === Download latest files ===
if not harvest_files:
    print("❌ No Harvest files found.")
    exit()

if not pt_files:
    print("❌ No Emlid_PT_Intergrated files found.")
    exit()

latest_harvest = max(harvest_files, key=lambda x: extract_date_key(x[0]))
latest_pt = max(pt_files, key=lambda x: extract_date_key(x[0]))

# Download files
drive.CreateFile({'id': latest_harvest[1]}).GetContentFile(latest_harvest[0])
drive.CreateFile({'id': latest_pt[1]}).GetContentFile(latest_pt[0])

print(f"📥 Downloaded: {latest_harvest[0]}")
print(f"📥 Downloaded: {latest_pt[0]}")

# === Load Harvest CSV ===
df = pd.read_csv(latest_harvest[0])
print("📋 Harvest Columns:", df.columns.tolist())

# === Rename columns safely ===
df.columns = df.columns.str.strip()  # Strip spaces

# rename_dict = {}
# if "Wet wt. (g)" in df.columns:
#     rename_dict["Wet wt. (g)"] = "Sub Wet Wt (g)"
# if "Dry wt. (g)" in df.columns:
#     rename_dict["Dry wt. (g)"] = "Sub Dry Wt (g)"
# elif "Dry  wt. (g)" in df.columns:
#     rename_dict["Dry  wt. (g)"] = "Sub Dry Wt (g)"

# df.rename(columns=rename_dict, inplace=True)

# === Clean and calculate ===
if "Weight in kg" in df.columns:
    df = df[df["Weight in kg"] >= 0]

if "Units" in df.columns:
    del df["Units"]

df.insert(df.columns.get_loc("Length in meters") + 1, "Width (m)", 0.8128)
df["Area (m²)"] = df["Length in meters"] * df["Width (m)"]
df.insert(df.columns.get_loc("Width (m)") + 1, "Area (m²)", df.pop("Area (m²)"))

df ["Dry Wt (g)"] = df["DW + bag (g)"] - df["Dry bag wt. (g)"]
df["Wet Wt (g)"] = df["WW + bag (g)"] - df["Wet bag wt. (g)"]
df["Dry Matter"] = df["Dry Wt (g)"] / df["Wet Wt (g)"]
df["Biomass (kg/ha)"] = (df["Weight in kg"] * df["Dry Matter"]) / df["Area (m²)"] * 10000
df["Residual Dry Wt (kg/ha)"] = 950
df["Total Biomass (kg/ha)"] = df["Biomass (kg/ha)"] + df["Residual Dry Wt (kg/ha)"]
df = df.dropna(axis=1, how='all')

# # === Load Emlid_PT_Intergrated CSV ===
# dh = pd.read_csv(latest_pt[0])
dh = pd.read_csv("Emlid_PT_Integration.csv")
# === Merge Yield + EMLID ===
data = pd.merge(df, dh, on=["Plot", "Strip"], how="left")

# Drop unwanted columns
if "Unnamed: 0" in data.columns:
    data.drop(columns=["Unnamed: 0"], inplace=True)

data.rename(columns={"Average Height": "PT Height (mm)"}, inplace=True)

# === Define preferred column order ===
preferred_columns = [
    "Experiment", "Farm", "Coordinates", "Date", "Pre/Post",
    "Plot", "Strip", "Strip Coordinates", "PT Height (mm)"
]

# === Remove duplicate 'Date' columns ===
# This ensures only one 'Date' column exists
data = data.loc[:, ~data.columns.duplicated()]

# === Filter columns that actually exist in the DataFrame
existing_preferred = [col for col in preferred_columns if col in data.columns]
remaining_columns = [col for col in data.columns if col not in existing_preferred]

# === Reorder columns
data = data[existing_preferred + remaining_columns]

# === Save final Yield file ===
date_str = extract_date_key(latest_harvest[0])
date_str_fmt = f"{date_str[1]:02}-{date_str[2]:02}-{date_str[0]}"
yield_filename = f"Yield_{date_str_fmt}.csv"
data.to_csv(yield_filename, index=False)

# === Upload Yield to Drive ===
upload_file = drive.CreateFile({'title': yield_filename, 'parents': [{'id': folder_3_id}]})
upload_file.SetContentFile(yield_filename)
upload_file.Upload()
print(f"✅ Yield file uploaded to Google Drive as: {yield_filename}")

# === Gmail API ===
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def gmail_authenticate():
    creds = None
    token_file = 'token_gmail.json'
    creds_file = 'credentials_gmail.json'

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

# === Send email with attachment ===
receiver_emails = [
      'darapanenivandana1199@gmail.com',
    'vdzfb@missouri.edu',
    'ummbv@missouri.edu',
    'bernardocandido@missouri.edu',
    'bpbf25@mizzou.edu',
    'emh3d9@missouri.edu',
    'rashmi.p.sharma@missouri.edu',
    'kayanbaptista@gmail.com'
    
]

subject_success = f"✅ Yield Data - {yield_filename}"
body_success = "Hi Team,\n\nPlease find the final Yield CSV output file attached.\n\nRegards,\nAutomated System"

try:
    send_email_gmail_api(subject_success, body_success, receiver_emails, attachment_path=yield_filename)
except Exception as e:
    error_trace = traceback.format_exc()
    send_email_gmail_api('❌ Yield Script Failed', f"Script failed:\n{error_trace}", receiver_emails)
