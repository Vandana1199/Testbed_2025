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
gauth.LoadCredentialsFile("mycreds.txt")

if gauth.credentials is None:
    gauth.CommandLineAuth()  # 👈 Use this instead of LocalWebserverAuth
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()

gauth.SaveCredentialsFile("mycreds.txt")
drive = GoogleDrive(gauth)

# folder ID to connect with the folder inside the drive
folder_id = '1FVAhkz6bdtsOyBN4jtrUF0Dzg1kbEkbs'

# List all files in the folder
file_list = drive.ListFile({
    'q': f"'{folder_id}' in parents and trashed=false"
}).GetList()

for file in file_list:
    print(f"Title: {file['title']}, ID: {file['id']}")

# === Step 1: Define folder ID ===
folder_id = '1FVAhkz6bdtsOyBN4jtrUF0Dzg1kbEkbs'

# === Step 2: Get all files in the folder ===
file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

# === Step 3: Setup ===
emlid_files = []
pt_files = []
testbed_file = None

pattern_emlid = re.compile(r'^EMLID_(\d+\.\d+\.\d+)\.csv$')
pattern_pt = re.compile(r'^PT_(\d+\.\d+\.\d+)\.csv$')
testbed_filename = 'TestBed_EMLIDPT_Test_Corners.csv'

# Function to sort files by date in filename
def extract_date_key(filename):
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', filename)
    if match:
        month, day, year = map(int, match.groups())
        return (year, month, day)
    return (0, 0, 0)

# === Step 4: Classify files ===
for file in file_list:
    title = file['title']
    if title == testbed_filename:
        testbed_file = (title, file['id'])
    elif pattern_emlid.match(title):
        emlid_files.append((title, file['id']))
    elif pattern_pt.match(title):
        pt_files.append((title, file['id']))

# === Step 5: Download constant testbed file only if not present ===
if testbed_file:
    if not os.path.exists(testbed_file[0]):
        print(f"📥 Downloading Testbed file (once): {testbed_file[0]}")
        file = drive.CreateFile({'id': testbed_file[1]})
        file.GetContentFile(testbed_file[0])
    else:
        print(f"✅ Testbed file already exists locally: {testbed_file[0]}")
else:
    print("❌ Testbed file not found in Drive.")

# === Step 6: Get and download latest EMLID file ===
if emlid_files:
    latest_emlid = max(emlid_files, key=lambda x: extract_date_key(x[0]))
    print(f"📥 Latest EMLID file: {latest_emlid[0]}")
    file = drive.CreateFile({'id': latest_emlid[1]})
    file.GetContentFile(latest_emlid[0])
    GPS_raw = pd.read_csv(latest_emlid[0])
else:
    print("❌ No EMLID files found.")

# === Step 7: Get and download latest PT file ===
if pt_files:
    latest_pt = max(pt_files, key=lambda x: extract_date_key(x[0]))
    print(f"📥 Latest PT file: {latest_pt[0]}")
    file = drive.CreateFile({'id': latest_pt[1]})
    file.GetContentFile(latest_pt[0])
    PT_raw = pd.read_csv(latest_pt[0])
else:
    print("❌ No PT files found.")

# === 8. Load data ===
GPS_raw = pd.read_csv("EMLID_3.11.25.csv")
PT_raw = pd.read_csv("PT_3.11.25.csv")

# === 9. Clean timestamp in PT data ===
PT_raw['datetime_clean'] = PT_raw['datetime'].str.replace(r'^[A-Za-z]+ ', '', regex=True)
PT_raw['datetime_clean'] = PT_raw['datetime_clean'].str.replace(r' GMT.*', '', regex=True)
PT_raw['Timestamp'] = pd.to_datetime(PT_raw['datetime_clean'], format='%b %d %Y %H:%M:%S')
PT_raw.drop(columns=['datetime_clean'], inplace=True)

# === 10. Parse datetime, date, time ===
PT_raw['datetime'] = pd.to_datetime(PT_raw['Timestamp'])
PT_raw['date'] = PT_raw['datetime'].dt.date
PT_raw['time'] = pd.to_timedelta(PT_raw['datetime'].dt.strftime('%H:%M:%S'))


# === 11. Prepare GPS data ===
GPS = GPS_raw.rename(columns={"longitude(deg)": "X", "latitude(deg)": "Y"})
GPS = GPS[['X', 'Y', 'GPST']]


# === 12. Rescale PT time using time offset ===
time_diff = timedelta(hours=5, minutes=0, seconds=18)
PT_scaled = PT_raw.sort_values('time').copy()
PT_scaled['n'] = range(1, len(PT_scaled) + 1)
PT_scaled['scaled_time'] = PT_scaled['datetime'] + time_diff


# === 13. Add IDs for merging ===
PT_merge = PT_scaled[['rawdistance', 'scaled_time', "tare", 'date']].copy()
PT_merge['scaled_time2'] = PT_merge['scaled_time'].astype('int64') // 10**9
PT_merge['ID'] = PT_merge.groupby('scaled_time2').cumcount() + 1
PT_merge['scaled_time'] = PT_merge['scaled_time'].dt.strftime('%H:%M:%S')
GPS['scaled_time'] = pd.to_datetime(GPS['GPST'], format="%I:%M:%S %p").dt.strftime("%H:%M:%S")
GPS['scaled_time2'] = GPS['scaled_time'].str.replace(':', '').astype(int)
GPS['ID'] = GPS.groupby('scaled_time2').cumcount() + 1
GPS_merge = GPS[['X', 'Y', 'scaled_time', 'ID']]


# === 14. Merge PT and GPS ===
merged_data = pd.merge(PT_merge, GPS_merge, how='outer', on=['scaled_time', 'ID'])
merged_data = merged_data.ffill()

# === 15. Filter relevant columns ===
merged_filtered = merged_data[['rawdistance', 'X', 'Y', 'scaled_time', 'tare', "date"]].rename(columns={'scaled_time': 'time'})
merged_filtered
merged_filtered.isnull().sum()
merged_filtered.dropna()

# # === 16. Visualize mean per second ===
# means_per_sec = merged_filtered.groupby(['time', 'X', 'Y']).agg(mean_height=('rawdistance', 'mean')).reset_index()
# plt.figure()
# plt.scatter(means_per_sec['time'], means_per_sec['mean_height'])
# plt.xticks(rotation=90)
# plt.title("Mean height per second")
# plt.show()

# # === 17. Read and process plot corners ===
# Read the CSV file
corners = pd.read_csv("TestBed_EMLIDPT_Test_Corners.csv").rename(columns={'Longitude': 'x', 'Latitude': 'y'})

# step 18 : Initialize lists to store polygons, plot IDs, and strip IDs
polygon_list = []
ids = []
strips = []

# Iterate through unique Plot and Strip combinations
for (plot_id, strip_id) in corners[['Plot', 'Strip']].drop_duplicates().values:
    # Filter data for the current plot and strip combination
    polygon_data = corners[(corners['Plot'] == plot_id) & (corners['Strip'] == strip_id)]
    
    # Add strip information to the list
    strips.append(strip_id)
    
    # Create a closed polygon
    closed_polygon = pd.concat([polygon_data, polygon_data.iloc[[0]]], ignore_index=True)
    coords = closed_polygon[['x', 'y']].values.tolist()
    polygon = Polygon(coords)
    
    # Append polygon and plot ID
    polygon_list.append(polygon)
    ids.append(str(plot_id))

# Create a GeoDataFrame with 'Plot', 'Strip', and 'geometry'
polygon_gdf = gpd.GeoDataFrame({'Plot': ids, 'Strip': strips, 'geometry': polygon_list}, crs="EPSG:4326")

# === 19. Convert PT data to GeoDataFrame ===
merged_filtered_clean = merged_filtered.dropna(subset=["X", "Y"]).copy()
PT_gdf = gpd.GeoDataFrame(merged_filtered_clean, geometry=gpd.points_from_xy(merged_filtered_clean["X"], merged_filtered_clean["Y"]), crs="EPSG:4326")

# === 20. Intersect points with polygons ===
# gpd.options.use_pygeos = False  # to mimic sf::sf_use_s2(FALSE)
plot_intersect = gpd.sjoin(PT_gdf, polygon_gdf, how='inner', predicate='within')
plot_intersect

plot_intersect["Plot"].value_counts()

# # === 21. Visualize plots with points ===
# plt.figure()
# plot_corners_plot = corners.copy()
# for plot_id in plot_corners_plot['Plot'].unique():
#     poly = plot_corners_plot[plot_corners_plot['Plot'] == plot_id].copy()
#     poly = pd.concat([poly, poly.iloc[[0]]], ignore_index=True)
#     plt.plot(poly['x'], poly['y'], marker='o', label=f"Plot {plot_id}")
# plt.scatter(plot_intersect.geometry.x, plot_intersect.geometry.y, c='red', s=10, label='Points')
# plt.legend()
# plt.title("PT Points in Plots")
# plt.show()

# Calculate the height from the raw
plot_intersect["PT_Height(cm)"] = plot_intersect["rawdistance"] - plot_intersect["tare"]
plot_intersect


# Merge the plot_intersect dataframe with polygon_gdf based on the 'plot' column
result = plot_intersect.merge(polygon_gdf[['Plot', 'geometry']], on='Plot', how='left')
result = result.rename(columns = {"geometry_x" : "geometry", "geometry_y" : "Coordinates", "date" : "Date"})

# === 22. Final height normalization ===
result['PTdata_cm'] = (result['rawdistance'] * 0.8662) / 100
result['grass_height_cm'] = 85 - result['PTdata_cm'] - (85 - (result['tare'] * 0.8662 / 100))

# Aggregating data by 'plot' and including the required columns
Emlid_PT_Intergrated = result.groupby(['Plot', 'Strip']).agg(
    mean_height=('grass_height_cm', 'mean'),
    Coordinates=('Coordinates', 'first'),
    Date=('Date', 'first')
).reset_index()

# Add the Farm Coordinates
Emlid_PT_Intergrated["Farm_Coordiantes"] = """POLYGON ((-92.27126447977226,38.905762369582106, -92.27126447977226,38.90536179063761, -92.26988422615567,38.90536179063761, -92.26988422615567,38.905762369582106, -92.27126447977226,38.905762369582106))"""
Emlid_PT_Intergrated

# # === After generating and saving the final CSV ===
# Emlid_PT_Intergrated.to_csv('Emlid_PT_Intergrated.csv', index=False)
# print('✅ CSV file saved as Emlid_PT_Intergrated.csv')

# === Rename the file with date suffix ===
emlid_date_str = re.search(r'(\d+\.\d+\.\d+)', latest_emlid[0]).group(1).replace('.', '-')
intermediate_file = 'Emlid_PT_Intergrated.csv'
final_filename = f"Emlid_PT_Intergrated_{emlid_date_str}.csv"

# Save the file
Emlid_PT_Intergrated.to_csv(intermediate_file, index=False)
print(f"✅ CSV file saved as {intermediate_file}")

# Rename safely
if os.path.exists(final_filename):
    os.remove(final_filename)

os.rename(intermediate_file, final_filename)
print(f"📦 Renamed to: {final_filename}")

# === Upload to Google Drive ===
upload_file = drive.CreateFile({'title': final_filename, 'parents': [{'id': folder_id}]})
upload_file.SetContentFile(final_filename)
upload_file.Upload()
print(f"✅ Final file uploaded to Google Drive as: {final_filename}")

# Gmail API Scope
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def gmail_authenticate():
    """Authenticate using Gmail API and return service"""
    creds = None
    token_file = 'token_gmail.json'
    creds_file = 'credentials_gmail.json'  # <-- Custom named Gmail credentials

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

# === Final Email Execution ===

receiver_emails = [
    'darapanenivandana1199@gmail.com',
    'vdzfb@missouri.edu', 
    #"bernardocandido@missouri.edu"
]

subject_success = '✅ Final Output CSV File'
body_success = 'Hi, please find the final CSV output file attached.'

subject_failure = '❌ Script Execution Failed'
body_failure = 'Hi, the script encountered an error during execution. Please check the logs below:\n\n'

final_filename = "Emlid_PT_Intergrated_3-11-25.csv"  # or dynamic filename

try:
    send_email_gmail_api(subject_success, body_success, receiver_emails, attachment_path=final_filename)
except Exception as e:
    error_trace = traceback.format_exc()
    full_body = body_failure + error_trace
    send_email_gmail_api(subject_failure, full_body, receiver_emails)
