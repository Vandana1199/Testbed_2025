from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd
import re
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


def clean_plot_value(x):
    """
    Keeps decimal plot values like 14.1.
    Removes trailing .0 only if value is whole number like 14.0.
    """
    if pd.isna(x):
        return None

    x = str(x).strip()

    try:
        num = float(x)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except Exception:
        return x


def clean_strip_value(x):
    """
    Converts Strip values like 1.0 to 1.
    """
    if pd.isna(x):
        return None

    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


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
    df = df.dropna(axis=1, how='all')

    required_cols = ["Plot", "Strip", "UniqueID"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing in Harvest file: {col}")

    # =========================================================
    # CLEAN PLOT + STRIP SAFELY
    # =========================================================

    df["Plot_clean"] = df["Plot"].apply(clean_plot_value)
    df["Strip_clean"] = df["Strip"].apply(clean_strip_value)

    df = df.dropna(subset=["Plot_clean", "Strip_clean"])

    df["unique_id"] = df["Plot_clean"] + "_" + df["Strip_clean"]

    df.rename(columns={"UniqueID": "Harvest_UniqueID"}, inplace=True)

    print("✅ unique_id created successfully")
    print("🔎 Sample unique_id values:", df["unique_id"].head().tolist())

    # =========================================================
    # HARVEST WEIGHT CLEANING
    # =========================================================

    if "Harvestor_wt_kg" not in df.columns:
        raise ValueError("Required column missing in Harvest file: Harvestor_wt_kg")

    df["Harvestor_wt_kg"] = pd.to_numeric(
        df["Harvestor_wt_kg"],
        errors="coerce"
    )

    df = df[df["Harvestor_wt_kg"] >= 0]

    if "Units" in df.columns:
        del df["Units"]

    if "Unnamed: 12" in df.columns:
        del df["Unnamed: 12"]

    # =========================================================
    # LENGTH + AREA
    # =========================================================

    # Your current Harvest file does not show Length (m).
    # So using fixed strip width and fixed length fallback.
    # Update this value if your strip length is different.
    if "Length (m)" not in df.columns:
        print("⚠️ Length (m) column missing. Using default Length (m) = 1.")
        df["Length (m)"] = 1

    df["Length (m)"] = pd.to_numeric(
        df["Length (m)"],
        errors="coerce"
    )

    df.insert(
        df.columns.get_loc("Length (m)") + 1,
        "Width (m)",
        0.8128
    )

    df["Area (m²)"] = df["Length (m)"] * df["Width (m)"]

    df.insert(
        df.columns.get_loc("Width (m)") + 1,
        "Area (m²)",
        df.pop("Area (m²)")
    )

    # =========================================================
    # DRY MATTER CALCULATION
    # =========================================================

    required_weight_cols = [
        "Dry  wt. (g)",
        "Wet wt. (g)"
    ]

    for col in required_weight_cols:
        if col not in df.columns:
            raise ValueError(f"Required weight column missing: {col}")

    df["Dry  wt. (g)"] = pd.to_numeric(
        df["Dry  wt. (g)"],
        errors="coerce"
    )

    df["Wet wt. (g)"] = pd.to_numeric(
        df["Wet wt. (g)"],
        errors="coerce"
    )

    df["Dry Matter %"] = (
        df["Dry  wt. (g)"] /
        df["Wet wt. (g)"]
    ).round(2)

    # =========================================================
    # BIOMASS CALCULATION
    # =========================================================

    df["Biomass (kg/ha)"] = (
        (
            df["Harvestor_wt_kg"] *
            df["Dry Matter %"]
        ) /
        df["Area (m²)"]
    ) * 10000

    df["Biomass (kg/ha)"] = pd.to_numeric(
        df["Biomass (kg/ha)"],
        errors="coerce"
    ).round(2)

    df["Residual (kg/ha)"] = 980

    df["Total Biomass (kg/ha)"] = (
        df["Biomass (kg/ha)"] +
        df["Residual (kg/ha)"]
    )

    df["Dry Matter %"] = df["Dry Matter %"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else ""
    )

    print("✅ Harvest preprocessing completed successfully")

    # =========================================================
    # LOAD HEIGHT / VI / WEATHER CSV
    # =========================================================

    dh = pd.read_csv(latest_pt[0])
    dh.columns = dh.columns.str.strip()

    print("📋 Height_VIs Columns:", dh.columns.tolist())

    if "unique_id" not in dh.columns:
        raise ValueError("Required column missing in Height/VI/Weather file: unique_id")

    dh["unique_id"] = dh["unique_id"].astype(str).str.strip()
    dh["unique_id"] = dh["unique_id"].str.replace(r"\.0_", "_", regex=True)

    df["unique_id"] = df["unique_id"].astype(str).str.strip()

    print("🔎 Harvest unique_id sample:", df["unique_id"].head().tolist())
    print("🔎 Height file unique_id sample:", dh["unique_id"].head().tolist())

    # =========================================================
    # MERGE HARVEST + HEIGHT/VI/WEATHER
    # =========================================================

    data = pd.merge(df, dh, on=["unique_id"], how="left")

    print("📋 Final_Yield Columns:", data.columns.tolist())
    print(f"✅ Rows after merge: {len(data)}")
    print(f"✅ Matched Height rows: {data['Coordinates'].notna().sum() if 'Coordinates' in data.columns else 'Coordinates column not found'}")

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
