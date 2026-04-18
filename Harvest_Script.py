from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd
import re
import os
import sys
import traceback
import base64
from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError

# =========================================================
# CONFIG
# =========================================================
DRIVE_CREDENTIALS_FILE = "mycreds_new.txt"

GMAIL_TOKEN_FILE = "token_gmail.json"
GMAIL_CREDENTIALS_FILE = "credentials_gmail.json"
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.send']

folder_1_id = '180uYCxLfyhZQNuN8XLS7sUIITTkqFcF1'  # Harvest
folder_2_id = '1eAlkuHwjsV0VxgB3pjRRbU39PgucE1f5'  # Emlid_PT_Intergrated
folder_3_id = '1HJq7XpoL7HaYWVdyMRej4wEysRRUSIvx'  # Final output (Yield)

receiver_emails = [
    'darapanenivandana1199@gmail.com',
    'vdzfb@missouri.edu',
]

# =========================================================
# HELPERS
# =========================================================
def extract_date_key(filename):
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', filename)
    if match:
        month, day, year = map(int, match.groups())
        return (year, month, day)
    return (0, 0, 0)


def safe_exit(message, code=1):
    print(message)
    sys.exit(code)


# =========================================================
# GOOGLE DRIVE AUTH
# =========================================================
def authenticate_drive():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(DRIVE_CREDENTIALS_FILE)

    if gauth.credentials is None:
        print("🔑 No Drive credentials found, starting authentication...")
        gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        print("🔑 Drive token expired, attempting to refresh...")
        try:
            gauth.Refresh()
        except Exception as e:
            print(f"❌ Drive token refresh failed: {e}")
            print("🔑 Refresh failed, starting re-authentication...")
            gauth.CommandLineAuth()
    else:
        print("✅ Drive token is valid.")
        gauth.Authorize()

    gauth.SaveCredentialsFile(DRIVE_CREDENTIALS_FILE)
    return GoogleDrive(gauth)


# =========================================================
# GMAIL AUTH
# =========================================================
def gmail_authenticate():
    """
    Authenticate using Gmail API.
    In CI/GitHub Actions, this should use a pre-generated token_gmail.json.
    If refresh token is invalid, raise a clear error instead of crashing unclearly.
    """
    creds = None

    if os.path.exists(GMAIL_TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {GMAIL_TOKEN_FILE}: {e}"
            ) from e

    if creds and creds.valid:
        return build('gmail', 'v1', credentials=creds)

    if creds and creds.expired and creds.refresh_token:
        try:
            print("🔄 Gmail token expired. Attempting refresh...")
            creds.refresh(Request())

            with open(GMAIL_TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())

            print("✅ Gmail token refreshed successfully.")
            return build('gmail', 'v1', credentials=creds)

        except RefreshError as e:
            raise RuntimeError(
                "Gmail refresh token is invalid or expired. "
                "Delete token_gmail.json and re-authorize locally, then upload the new token to your workflow environment. "
                "Also check whether your OAuth consent screen is still in Testing mode."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected Gmail token refresh error: {e}"
            ) from e

    raise RuntimeError(
        "No valid Gmail credentials available. "
        "In GitHub Actions, provide a valid token_gmail.json generated locally."
    )


def send_email_gmail_api(subject, body_text, to_emails, attachment_path=None):
    """
    Send email using Gmail API.
    Returns True if successful, False otherwise.
    """
    try:
        service = gmail_authenticate()
    except Exception as auth_error:
        print("❌ Gmail authentication failed.")
        print(auth_error)
        return False

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

            msg.add_attachment(
                file_data,
                maintype='application',
                subtype='octet-stream',
                filename=file_name
            )
        except Exception as file_err:
            print(f"⚠️ Failed to attach file: {file_err}")
            msg.set_content(f"{body_text}\n\n⚠️ Failed to attach file: {file_err}")

    encoded_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    send_message = {'raw': encoded_message}

    try:
        service.users().messages().send(userId="me", body=send_message).execute()
        print("✅ Email sent via Gmail API!")
        return True
    except Exception as e:
        print("❌ Failed to send email using Gmail API.")
        print(e)
        return False


# =========================================================
# MAIN LOGIC
# =========================================================
def main():
    drive = authenticate_drive()

    # === List files ===
    harvest_files_raw = drive.ListFile({
        'q': f"'{folder_1_id}' in parents and trashed=false"
    }).GetList()

    pt_files_raw = drive.ListFile({
        'q': f"'{folder_2_id}' in parents and trashed=false"
    }).GetList()

    print("📂 Harvest Folder Files:")
    for file in harvest_files_raw:
        print(f"Title: {file['title']}, ID: {file['id']}")

    print("\n📂 Emlid_PT_Intergrated Folder Files:")
    for file in pt_files_raw:
        print(f"Title: {file['title']}, ID: {file['id']}")

    # === Filter matching files ===
    harvest_files = [
        (f['title'], f['id'])
        for f in harvest_files_raw
        if re.match(r'^Harvest_(\d+\.\d+\.\d+)\.csv$', f['title'])
    ]

    pt_files = [
        (f['title'], f['id'])
        for f in pt_files_raw
        if re.match(r"^Height_VI's_Weather_(\d+\.\d+\.\d+)\.csv$", f['title'])
    ]

    if not harvest_files:
        safe_exit("❌ No Harvest files found.")

    if not pt_files:
        safe_exit("❌ No Emlid_PT_Intergrated files found.")

    latest_harvest = max(harvest_files, key=lambda x: extract_date_key(x[0]))
    latest_pt = max(pt_files, key=lambda x: extract_date_key(x[0]))

    # === Download latest files ===
    drive.CreateFile({'id': latest_harvest[1]}).GetContentFile(latest_harvest[0])
    drive.CreateFile({'id': latest_pt[1]}).GetContentFile(latest_pt[0])

    print(f"📥 Downloaded: {latest_harvest[0]}")
    print(f"📥 Downloaded: {latest_pt[0]}")

    # === Load Harvest CSV ===
    df = pd.read_csv(latest_harvest[0])
    print("📋 Harvest Columns:", df.columns.tolist())

    df.columns = df.columns.str.strip()

    if "HarvesterWeight (kg)" in df.columns:
        df = df[df["HarvesterWeight (kg)"] >= 0]

    if "Units" in df.columns:
        del df["Units"]

    # Ensure numeric ID columns
    for col in ["Pasture", "Paddock", "Strip"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(pd.Int64Dtype())
        else:
            raise ValueError(f"Required column missing in Harvest file: {col}")

    df["unique_id"] = (
        df["Pasture"].astype(str) + "." +
        df["Paddock"].astype(str) + "_" +
        df["Strip"].astype(str)
    )

    if "Length (m)" not in df.columns:
        raise ValueError("Required column missing in Harvest file: Length (m)")

    df.insert(df.columns.get_loc("Length (m)") + 1, "Width (m)", 0.8128)
    df["Area (m²)"] = df["Length (m)"] * df["Width (m)"]
    df.insert(df.columns.get_loc("Width (m)") + 1, "Area (m²)", df.pop("Area (m²)"))

    if "Dry  wt. (g)" not in df.columns or "Wet wt. (g)" not in df.columns:
        raise ValueError("Required wet/dry weight columns are missing in Harvest file.")

    df["Dry Matter %"] = (df["Dry  wt. (g)"] / df["Wet wt. (g)"]).round(2)

    df["Biomass (kg/ha)"] = (
        (df["HarvesterWeight (kg)"] * df["Dry Matter %"]) / df["Area (m²)"] * 10000
    ).round(2)

    df["Residual (kg/ha)"] = 980

    df["Biomass (kg/ha)"] = pd.to_numeric(df["Biomass (kg/ha)"], errors='coerce')
    df["Residual (kg/ha)"] = pd.to_numeric(df["Residual (kg/ha)"], errors='coerce')
    df["Total Biomass (kg/ha)"] = df["Biomass (kg/ha)"] + df["Residual (kg/ha)"]

    # Optional formatting
    df["Dry Matter %"] = df["Dry Matter %"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    df = df.dropna(axis=1, how='all')

    # === Load Height/VI/Weather CSV ===
    dh = pd.read_csv(latest_pt[0])
    print("📋 Height_VIs Columns:", dh.columns.tolist())

    # === Merge ===
    data = pd.merge(df, dh, on=["unique_id"], how="left")
    print("📋 Final_Yield Columns:", data.columns.tolist())

    # === Clean conflicting columns ===
    if "Strip_y" in data.columns:
        data.rename(columns={"Strip_y": "Strip"}, inplace=True)
    if "Strip_x" in data.columns:
        data.drop(columns=["Strip_x"], inplace=True)

    if "Plot_y" in data.columns:
        data.rename(columns={"Plot_y": "Plot"}, inplace=True)
    if "Plot_x" in data.columns:
        data.drop(columns=["Plot_x"], inplace=True)

    if "Unnamed: 0" in data.columns:
        data.drop(columns=["Unnamed: 0"], inplace=True)

    # === Preferred output columns ===
    preferred_columns = [
        'Experiment', 'Date', 'JulianDate', 'PrePost', 'Plot',
        'Strip', 'Coordinates', 'Farm_Coordinates', 'PT_Height(mm)',
        'NDVI_mean', 'GNDVI_mean', 'EVI_mean', 'SAVI_mean', 'MSAVI_mean',
        'NDRE_mean', 'CLRE_mean', 'SRre_mean', 'red_mean', 'green_mean',
        'blue_mean', 'nir_mean', 'rededge_mean', 'observation_sum',
        'prism_normals_sum', 'departure_from_normal_sum', 'percent_of_normal_sum',
        'Precipitation_inch', 'Min_Air_Temperature_F', 'Max_Air_Temperature_F',
        'Avg_Air_Temperature_F', 'Min_Temp_C', 'Max_Temp_C', 'Base_Temp_C',
        'GDD', 'unique_id', 'Dry Matter %', 'Total Biomass (kg/ha)'
    ]

    existing_preferred = [col for col in preferred_columns if col in data.columns]
    data = data[existing_preferred]

    # === Save final Yield file ===
    date_key = extract_date_key(latest_harvest[0])
    date_str_fmt = f"{date_key[1]:02}-{date_key[2]:02}-{date_key[0]}"
    yield_filename = f"Yield_{date_str_fmt}.csv"

    data.to_csv(yield_filename, index=False)
    print(f"💾 Final yield file saved locally as: {yield_filename}")

    # === Upload Yield to Drive ===
    upload_file = drive.CreateFile({
        'title': yield_filename,
        'parents': [{'id': folder_3_id}]
    })
    upload_file.SetContentFile(yield_filename)
    upload_file.Upload()
    print(f"✅ Yield file uploaded to Google Drive as: {yield_filename}")

    # === Send success email ===
    subject_success = f"✅ Yield Data - {yield_filename}"
    body_success = (
        "Hi Team,\n\n"
        "Please find the final Yield CSV output file attached.\n\n"
        "Regards,\n"
        "Automated System"
    )

    email_sent = send_email_gmail_api(
        subject_success,
        body_success,
        receiver_emails,
        attachment_path=yield_filename
    )

    if not email_sent:
        print("⚠️ Yield file was created and uploaded, but success email could not be sent.")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_trace = traceback.format_exc()
        print("❌ Script failed:")
        print(error_trace)

        # Do NOT try to email failure using the same broken Gmail auth blindly.
        # Try once, but do not crash again if Gmail is also broken.
        subject_fail = "❌ Yield Script Failed"
        body_fail = f"Script failed:\n\n{error_trace}"

        fail_email_sent = send_email_gmail_api(
            subject_fail,
            body_fail,
            receiver_emails,
            attachment_path=None
        )

        if not fail_email_sent:
            print("⚠️ Failure email could not be sent either. Check Gmail token/auth configuration.")

        sys.exit(1)
