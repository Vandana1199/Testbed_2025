from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd
import re
import os
import sys
import traceback

# =========================================================
# CONFIG
# =========================================================

DRIVE_CREDENTIALS_FILE = "mycreds_new.txt"

folder_1_id = '1cC6k8mqJa9TyAFKXO2t8BTSHOmZM81XC'  # Harvest
folder_2_id = '13Ljj7woD1lsPplBc7EVocKy-jDhXiV7-'  # Emlid_PT_Intergrated
folder_3_id = '1G4lXcgNpGwTEHsJPpLxcS9pYfmWOsj78'  # Final output Yield

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

    # =========================================================
    # LOAD HARVEST CSV
    # =========================================================

    df = pd.read_csv(latest_harvest[0])
    print("📋 Harvest Columns:", df.columns.tolist())

    df.columns = df.columns.str.strip()

    # Remove fully empty columns
    df = df.dropna(axis=1, how='all')

    # Your current Harvest file has Plot, Strip, UniqueID
    required_cols = ["Plot", "Strip", "UniqueID"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing in Harvest file: {col}")

    df["Plot"] = pd.to_numeric(df["Plot"], errors="coerce").astype(pd.Int64Dtype())
    df["Strip"] = pd.to_numeric(df["Strip"], errors="coerce").astype(pd.Int64Dtype())

    # Create unique_id to match Height/VI/Weather file format: Plot_Strip
    df["unique_id"] = (
        df["Plot"].astype(str).str.replace("<NA>", "", regex=False)
        + "_"
        + df["Strip"].astype(str).str.replace("<NA>", "", regex=False)
    )

    # Keep original UniqueID as reference
    df.rename(columns={"UniqueID": "Harvest_UniqueID"}, inplace=True)

    # =========================================================
    # HARVEST WEIGHT CLEANING
    # =========================================================

    if "Harvestor_wt_kg" in df.columns:
        df["Harvestor_wt_kg"] = pd.to_numeric(df["Harvestor_wt_kg"], errors="coerce")
        df = df[df["Harvestor_wt_kg"] >= 0]
    else:
        raise ValueError("Required column missing in Harvest file: Harvestor_wt_kg")

    if "Units" in df.columns:
        del df["Units"]

    if "Length (m)" not in df.columns:
        raise ValueError("Required column missing in Harvest file: Length (m)")

    df["Length (m)"] = pd.to_numeric(df["Length (m)"], errors="coerce")

    df.insert(df.columns.get_loc("Length (m)") + 1, "Width (m)", 0.8128)
    df["Area (m²)"] = df["Length (m)"] * df["Width (m)"]
    df.insert(df.columns.get_loc("Width (m)") + 1, "Area (m²)", df.pop("Area (m²)"))

    if "Dry  wt. (g)" not in df.columns or "Wet wt. (g)" not in df.columns:
        raise ValueError("Required wet/dry weight columns are missing in Harvest file.")

    df["Dry  wt. (g)"] = pd.to_numeric(df["Dry  wt. (g)"], errors="coerce")
    df["Wet wt. (g)"] = pd.to_numeric(df["Wet wt. (g)"], errors="coerce")

    df["Dry Matter %"] = (df["Dry  wt. (g)"] / df["Wet wt. (g)"]).round(2)

    df["Biomass (kg/ha)"] = (
        (df["Harvestor_wt_kg"] * df["Dry Matter %"]) / df["Area (m²)"] * 10000
    ).round(2)

    df["Residual (kg/ha)"] = 980

    df["Biomass (kg/ha)"] = pd.to_numeric(df["Biomass (kg/ha)"], errors='coerce')
    df["Residual (kg/ha)"] = pd.to_numeric(df["Residual (kg/ha)"], errors='coerce')

    df["Total Biomass (kg/ha)"] = (
        df["Biomass (kg/ha)"] + df["Residual (kg/ha)"]
    )

    df["Dry Matter %"] = df["Dry Matter %"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else ""
    )

    # =========================================================
    # LOAD HEIGHT / VI / WEATHER CSV
    # =========================================================

    dh = pd.read_csv(latest_pt[0])
    dh.columns = dh.columns.str.strip()

    print("📋 Height_VIs Columns:", dh.columns.tolist())

    if "unique_id" not in dh.columns:
        raise ValueError("Required column missing in Height/VI/Weather file: unique_id")

    dh["unique_id"] = dh["unique_id"].astype(str).str.replace(r"\.0", "", regex=True)
    df["unique_id"] = df["unique_id"].astype(str).str.replace(r"\.0", "", regex=True)

    # =========================================================
    # MERGE HARVEST + HEIGHT/VI/WEATHER
    # =========================================================

    data = pd.merge(df, dh, on=["unique_id"], how="left")

    print("📋 Final_Yield Columns:", data.columns.tolist())

    # Clean conflicting columns
    if "Strip_y" in data.columns:
        data.rename(columns={"Strip_y": "Strip"}, inplace=True)
    if "Strip_x" in data.columns:
        data.drop(columns=["Strip_x"], inplace=True)

    if "Plot_y" in data.columns:
        data.rename(columns={"Plot_y": "Plot"}, inplace=True)
    if "Plot_x" in data.columns:
        data.drop(columns=["Plot_x"], inplace=True)

    if "Date_y" in data.columns:
        data.rename(columns={"Date_y": "Date"}, inplace=True)
    if "Date_x" in data.columns:
        data.rename(columns={"Date_x": "Harvest_Date"}, inplace=True)

    if "Unnamed: 0" in data.columns:
        data.drop(columns=["Unnamed: 0"], inplace=True)

    if "Unnamed: 12" in data.columns:
        data.drop(columns=["Unnamed: 12"], inplace=True)

    # =========================================================
    # PREFERRED OUTPUT COLUMNS
    # =========================================================

    preferred_columns = [
        'Experiment',
        'Date',
        'Harvest_Date',
        'JulianDate',
        'PrePost',
        'Plot',
        'Strip',
        'Coordinates',
        'Farm_Coordinates',

        'PT_Mean_Height(mm)',
        'PT_Median_Height(mm)',
        'PT_Height_STD(mm)',
        'PT_Height_SE(mm)',
        'PT_Height_Sample_Count',

        'NDVI_mean',
        'GNDVI_mean',
        'EVI_mean',
        'SAVI_mean',
        'MSAVI_mean',
        'NDRE_mean',
        'CLRE_mean',
        'SRre_mean',
        'red_mean',
        'green_mean',
        'blue_mean',
        'nir_mean',
        'rededge_mean',

        'observation_sum',
        'prism_normals_sum',
        'departure_from_normal_sum',
        'percent_of_normal_sum',

        'precip_7d_sum_in',
        'gdd_7d_sum',
        'tavg_7d_avg_F',
        'tmin_7d_avg_F',
        'tmax_7d_avg_F',

        'precip_14d_sum_in',
        'gdd_14d_sum',
        'tavg_14d_avg_F',
        'tmin_14d_avg_F',
        'tmax_14d_avg_F',

        'precip_21d_sum_in',
        'gdd_21d_sum',
        'tavg_21d_avg_F',
        'tmin_21d_avg_F',
        'tmax_21d_avg_F',

        'unique_id',
        'Harvest_UniqueID',
        'Dry Matter %',
        'Biomass (kg/ha)',
        'Residual (kg/ha)',
        'Total Biomass (kg/ha)'
    ]

    existing_preferred = [col for col in preferred_columns if col in data.columns]
    data = data[existing_preferred]

    # =========================================================
    # SAVE FINAL YIELD FILE
    # =========================================================

    date_key = extract_date_key(latest_harvest[0])
    date_str_fmt = f"{date_key[1]:02}-{date_key[2]:02}-{date_key[0]}"

    yield_filename = f"Yield_{date_str_fmt}.csv"

    data.to_csv(yield_filename, index=False)

    print(f"💾 Final yield file saved locally as: {yield_filename}")

    # =========================================================
    # UPLOAD YIELD TO GOOGLE DRIVE
    # =========================================================

    upload_file = drive.CreateFile({
        'title': yield_filename,
        'parents': [{'id': folder_3_id}]
    })

    upload_file.SetContentFile(yield_filename)
    upload_file.Upload()

    print(f"✅ Yield file uploaded to Google Drive as: {yield_filename}")

    # =========================================================
    # EMAIL SECTION DISABLED
    # =========================================================

    # Email sending has been removed/commented out.
    # The script now only creates and uploads the Yield CSV file.


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

        # Gmail failure email section disabled.
        # No email will be sent if the script fails.

        sys.exit(1)
