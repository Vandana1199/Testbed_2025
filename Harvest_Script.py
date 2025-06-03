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
    print("🔑 No credentials found, starting authentication...")
    gauth.CommandLineAuth()
elif gauth.access_token_expired:
    print("🔑 Token expired, attempting to refresh...")
    try:
        gauth.Refresh()
    except Exception as e:
        print(f"❌ Token refresh failed: {e}")
        print("🔑 Refresh failed, starting re-authentication...")
        gauth.CommandLineAuth()
else:
    print("✅ Token is valid.")
    gauth.Authorize()

gauth.SaveCredentialsFile("mycreds_new.txt")
drive = GoogleDrive(gauth)

# === Folder IDs ===
folder_1_id = '180uYCxLfyhZQNuN8XLS7sUIITTkqFcF1'  # Harvest
folder_2_id = '1eAlkuHwjsV0VxgB3pjRRbU39PgucE1f5'  # Emlid_PT_Intergrated
folder_3_id = '1HJq7XpoL7HaYWVdyMRej4wEysRRUSIvx'  # Final output (Yield)

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
pt_files = [(f['title'], f['id']) for f in pt_files if re.match(r"^Height_VI's_Weather_(\d+\.\d+\.\d+)\.csv$", f['title'])]

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
if "HarvesterWeight (kg)" in df.columns:
    df = df[df["HarvesterWeight (kg)"] >= 0]

if "Units" in df.columns:
    del df["Units"]
    
# Generate unique ID in the format: Pasture.Paddock_Strip
df["unique_id"] = df["Pasture"].astype(str) + "." + df["Paddock"].astype(str) + "_" + df["Strip"].astype(str)

df.insert(df.columns.get_loc("Length (m)") + 1, "Width (m)", 0.8128)
df["Area (m²)"] = df["Length (m)"] * df["Width (m)"]
df.insert(df.columns.get_loc("Width (m)") + 1, "Area (m²)", df.pop("Area (m²)"))

# df ["Dry Wt (g)"] = df["DW + bag (g)"] - df["Dry bag wt. (g)"]
# df["Wet Wt (g)"] = df["WW + bag (g)"] - df["Wet bag wt. (g)"]
# 1. Compute Dry Matter as float, rounded
df["Dry Matter %"] = (df["Dry  wt. (g)"] / df["Wet wt. (g)"]).round(2)

# 2. Compute Total Biomass as float, rounded
df["Total Biomass (kg/ha)"] = (
    (df["HarvesterWeight (kg)"] * df["Dry Matter %"]) / df["Area (m²)"] * 10000
).round(2)

# 3. Optional: format as string ONLY if needed before export
# df["Dry Matter %"] = df["Dry Matter %"].map(lambda x: f"{x:.2f}")
# df["Total Biomass (kg/ha)"] = df["Total Biomass (kg/ha)"].map(lambda x: f"{x:.2f}")
# Convert to float if previously formatted as strings
df["Total Biomass (kg/ha)"] = df["Total Biomass (kg/ha)"].astype(int)
df["Residual Dry Wt (kg/ha)"] = 1056
df["Total Biomass (kg/ha)"] = (df["Total Biomass (kg/ha)"] + df["Residual Dry Wt (kg/ha)"])
df = df.dropna(axis=1, how='all')

# # === Load Emlid_PT_Intergrated CSV ===
# dh = pd.read_csv(latest_pt[0])
dh = pd.read_csv(latest_pt[0])
print("📋Height_VIs Columns:", dh.columns.tolist())

# === Merge Yield + EMLID ===
data = pd.merge(df, dh, on=["unique_id"], how="left")
print("📋Final_Yield Columns:", data.columns.tolist())

# === Clean column conflicts
if "Strip_y" in data.columns:
    data.rename(columns={"Strip_y": "Strip"}, inplace=True)
if "Strip_x" in data.columns:
    data.drop(columns=["Strip_x"], inplace=True)

if "Plot_y" in data.columns:
    data.rename(columns={"Plot_y": "Plot"}, inplace=True)
if "Plot_x" in data.columns:
    data.drop(columns=["Plot_x"], inplace=True)


# Drop unwanted columns
if "Unnamed: 0" in data.columns:
    data.drop(columns=["Unnamed: 0"], inplace=True)

# data.rename(columns={"Average Height": "PT Height (mm)"}, inplace=True)

# === Define preferred column order ===
preferred_columns = [
    'Experiment', 'Date', 'JulianDate', 'PrePost', 'Plot', 
    'Strip', 'Coordinates', 'Farm_Coordinates', 'PT_Height(mm)',
    'NDVI_mean', 'GNDVI_mean', 'SAVI_mean', 'MSAVI_mean',
    'NDRE_mean', 'CLRE_mean', 'SRre_mean', 'unique_id',
    'Dry Matter %', 'Total Biomass (kg/ha)'
]

# === Remove duplicate 'Date' columns
# data = data.loc[:, ~data.columns.duplicated()]

# === Filter only preferred columns
existing_preferred = [col for col in preferred_columns if col in data.columns]
data = data[existing_preferred]

# # === Filter columns that actually exist in the DataFrame
# existing_preferred = [col for col in preferred_columns if col in data.columns]
# remaining_columns = [col for col in data.columns if col not in existing_preferred]

# # === Reorder columns
# data = data[existing_preferred + remaining_columns]

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

# === Gmail API Scope and Functions ===
import base64
import os
import traceback
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def gmail_authenticate():
    """Authenticate using Gmail API and return service"""
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
    "darapanenivandana1199@gmail.com",
    "vdzfb@missouri.edu", 
    "bernardocandido@missouri.edu",
    "emh3d9@missouri.edu",
    "ummbv@missouri.edu",
    "rashmi.p.sharma@missouri.edu",
    "bpbf25@mizzou.edu",
    "kbn8m@missouri.edu"
]

subject_success = f"✅ Yield Data - {yield_filename}"
body_success = "Hi Team,\n\nPlease find the final Yield CSV output file attached.\n\nRegards,\nAutomated System"

try:
    send_email_gmail_api(subject_success, body_success, receiver_emails, attachment_path=yield_filename)
except Exception as e:
    error_trace = traceback.format_exc()
    send_email_gmail_api('❌ Yield Script Failed', f"Script failed:\n{error_trace}", receiver_emails)
