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

# Check if credentials exist or if the token is expired
if gauth.credentials is None:
    print("🔑 No credentials found, starting authentication...")
    gauth.CommandLineAuth()  # Trigger manual authentication and save new tokens
elif gauth.access_token_expired:
    print("🔑 Token expired, attempting to refresh...")
    try:
        gauth.Refresh()  # Attempt to refresh the token using the refresh token
    except Exception as e:
        print(f"❌ Token refresh failed: {e}")
        print("🔑 Refresh failed, starting re-authentication...")
        gauth.CommandLineAuth()        
              
else:
    print("✅ Token is valid.")
    gauth.Authorize()  # Authorize with valid credentials

# Save the refreshed credentials to file
gauth.SaveCredentialsFile("mycreds.txt")
drive = GoogleDrive(gauth)

# folder ID to connect with the folder inside the drive
folder_id = "16HMtw8qijxLL8WhTHv9zRqezFRv1eRtu"

# List all files in the folder
file_list = drive.ListFile({
    'q': f"'{folder_id}' in parents and trashed=false"
}).GetList()

for file in file_list:
    print(f"Title: {file['title']}, ID: {file['id']}")

# === Step 3: Setup ===
emlid_files = []
pt_files = []
testbed_file = None

pattern_emlid = re.compile(r'^EMLID_(\d+\.\d+\.\d+)\.csv$')
pattern_pt = re.compile(r'^PT_(\d+\.\d+\.\d+)\.csv$')
# testbed_filename = 'TestBed_StripCorners.csv'
testbed_filename = 'Warrensburg_Corners_7.2.25.csv'

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
GPS_raw = pd.read_csv(latest_emlid[0])
PT_raw = pd.read_csv(latest_pt[0])

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
GPS['scaled_time'] = pd.to_datetime(GPS['GPST'], format="%H:%M:%S", errors='coerce').dt.strftime("%H:%M:%S")
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

# # === 17. Read and process plot corners ===
# Read the CSV file
corners = pd.read_csv(testbed_filename).rename(columns={'POINT_X': 'x', 'POINT_Y': 'y'})
# Rename the 'plot' column to 'Plot'
corners.rename(columns={'PlotArea': 'Plot'}, inplace=True)
# print(corners.head())

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
# # gpd.options.use_pygeos = False  # to mimic sf::sf_use_s2(FALSE)
# plot_intersect = gpd.sjoin(PT_gdf, polygon_gdf, how='inner', predicate='within')
# plot_intersect

# Intersect
plot_intersect_full = gpd.sjoin(PT_gdf, polygon_gdf, how='inner', predicate='within')

# Sort for chronological safety
plot_intersect = plot_intersect_full.sort_values(['Plot', 'Strip', 'time'])

# # Store first 4 and last 6 readings per Plot-Strip
# dropped_start = plot_intersect_full.groupby(['Plot', 'Strip']).head(0)
# dropped_end = plot_intersect_full.groupby(['Plot', 'Strip']).tail(0)

# # Drop them from main data
# plot_intersect = (
#     plot_intersect_full.groupby(['Plot', 'Strip'])
#     .apply(lambda df: df.iloc[0:0] if len(df) > 10 else df.iloc[0:0])  # Avoid errors on small groups
#     .reset_index(drop=True)
# )

# # Print dropped readings
# print("📤 Dropped first 4 readings per Plot-Strip:")
# print(dropped_start)

# print("📤 Dropped last 6 readings per Plot-Strip:")
# print(dropped_end)

# # (Optional) Save dropped readings if needed
# dropped_start.to_csv("Dropped_First4_Readings.csv", index=False)
# dropped_end.to_csv("Dropped_Last6_Readings.csv", index=False)

# === Upload plot_intersect data to Google Drive ===
plot_intersect_file = "Raw_PT_Data.csv"
plot_intersect.to_csv(plot_intersect_full, index=False)

# Incorporate the date from emlid_date_str into the filename
emlid_date_str = re.search(r'(\d+\.\d+\.\d+)', latest_emlid[0]).group(1)  # Extract date from the latest EMLID file
Intfile = f"Raw_PT_Data_{emlid_date_str}.csv"  # Add the date to the filename
print(f"✅ plot_intersect CSV file saved as {Intfile}")  # Update to print the new file name

# Upload plot_intersect file to Google Drive
upload_plot_intersect_file = drive.CreateFile({
    'title': Intfile,
    'parents': [{'id': '1dcThJuXN3kK0Dr2aVBwZFRURicUOc5Sr'}]  # Specify the folder ID here
})
upload_plot_intersect_file.SetContentFile(plot_intersect_file)
upload_plot_intersect_file.Upload()
print(f"✅ plot_intersect file uploaded to Google Drive as: {Intfile}")

# Merge the plot_intersect dataframe with polygon_gdf based on the 'plot' column
# ✅ FIX: Merge on both Plot and Strip to preserve correct polygons
result = plot_intersect.merge(polygon_gdf[['Plot', 'Strip', 'geometry']], on=['Plot', 'Strip'], how='left')
result = result.rename(columns={"geometry_x": "geometry", "geometry_y": "Coordinates", "date": "Date"})
print("✅ Unique Plot-Strip combinations in result:", result[['Plot', 'Strip', 'Coordinates']].drop_duplicates().shape[0])


# === 22. Final height normalization ===
result['PTdata_cm'] = result['rawdistance']
result['grass_height_cm'] = (result['tare'] - result['PTdata_cm']) * 0.0859536

# Aggregating data by 'plot' and including the required columns
Emlid_PT_Intergrated = result.groupby(['Plot', 'Strip']).agg(
    mean_height=('grass_height_cm', 'mean'),
    Coordinates=('Coordinates', 'first'),
    Date=('Date', 'first')
).reset_index()

# Add the Farm Coordinates
Emlid_PT_Intergrated["Farm_Coordinates"] = """POLYGON ((-92.27126447977226 38.905762369582106, -92.27126447977226 38.90536179063761, -92.26988422615567 38.90536179063761, -92.26988422615567 38.905762369582106, -92.27126447977226 38.905762369582106))"""
Emlid_PT_Intergrated["unique_id"] = Emlid_PT_Intergrated["Plot"].astype(str) + "_" + Emlid_PT_Intergrated["Strip"].astype(str)
Emlid_PT_Intergrated


intermediate_file = 'Emlid_PT_Intergrated.csv'
final_file = f"Emlid_PT_Intergrated_{emlid_date_str}.csv"

# Save the file
Emlid_PT_Intergrated.to_csv(intermediate_file, index=False)
print(f"✅ CSV file saved as {intermediate_file}")

# Rename safely
if os.path.exists(final_file):
    os.remove(final_file)

os.rename(intermediate_file, final_file)
print(f"📦 Renamed to: {final_file}")

# === Upload to Google Drive ===
upload_file = drive.CreateFile({'title': final_file, 'parents': [{'id': '1nfKmQMzP5Oio0eWxOXU1tHLzDMG6Vnni'}]})
upload_file.SetContentFile(final_file)
upload_file.Upload()
print(f"✅ Final file uploaded to Google Drive as: {final_file}")

def fetch_and_process_farm_data(clipped_df):
    import pyodbc
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Polygon
    from shapely import wkt
    from datetime import timedelta
    from sentinelhub import (
        CRS, DataCollection, Geometry, SentinelHubStatistical,
        SentinelHubStatisticalDownloadClient, SHConfig, parse_time
    )


    if clipped_df.empty:
        print("Clipped DataFrame is empty")

    df = clipped_df

    # Sentinel Hub processing
    CLIENT_ID = '670e1809-9266-4a20-9857-d077e19962fb'
    CLIENT_SECRET = 'oK3noJhbwQmgtvyVbSDT9OewGBFLTDyV'
    config = SHConfig()
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

    # collection_id = "f1b3b558-17a3-4d40-8768-4870cd74cb06" #Anthony bucket
    # collection_id = "fb477b0a-47ef-4a8b-b020-19c0d7b35e4f" #testbed_bucket
    collection_id = "d7c0f6f9-284b-4337-b06a-db4d2f2c9350"
    PlanetScope_data_collection = DataCollection.define_byoc(collection_id)

    # df['Coordinates'] = df['Coordinates'].apply(lambda x: Polygon(eval(str(x))))
    df['Coordinates'] = df['Coordinates'].apply(wkt.loads)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Img_date'] = (df['Date']).dt.strftime("%Y-%m-%d")

    gdf = gpd.GeoDataFrame(df, geometry='Coordinates')
    gdf.crs = "EPSG:4326"

    # Rest of the Sentinel Hub processing code...
    # Convert date column to datetime
    gdf['Img_date'] = pd.to_datetime(gdf['Img_date'])

    # Define the data collection
    input_data = SentinelHubStatistical.input_data(data_collection=PlanetScope_data_collection)

    # Define a resolution, below is the minimum pixel size of planet image
    resx = 3
    resy = 3

    # Evaluation script to calculate NDVI, GNDVI, EVI, SAVI, and MSAVI
    # Evaluation script to calculate NDVI, GNDVI, EVI, SAVI, and MSAVI
    evalscript = """
    //VERSION=3

    function setup() {
    return {
        input: [
        {
            bands: [
            "red",
            "green",
            "blue",
            "nir",
            "rededge",
            "dataMask",
            "clear"
            ]
        }
        ],
        output: [
        {
            id: "ndvi",
            bands: 1
        },
        {
            id: "gndvi",
            bands: 1
        },
        {
            id: "evi",
            bands: 1
        },
        {
            id: "savi",
            bands: 1
        },
        {
            id: "msavi",
            bands: 1
        },
        {
            id: "ndre",
            bands: 1
        },
        {
            id: "Clre",
            bands: 1
        },
        {
            id: "SRre",
            bands: 1
        },
        {
            id: "dataMask",
            bands: 1
        }
        ]
    }
    }

    function isClear(clear) {
    return clear === 1;
    }

    function evaluatePixel(samples) {
    let ndvi = (samples.nir - samples.red) / (samples.nir + samples.red);
    let gndvi = (samples.nir - samples.green) / (samples.nir + samples.green);
    let evi = 2.5 * (samples.nir - samples.red) / (samples.nir + 6.0 * samples.red - 7.5 * samples.blue + 1.0);
    let L = 0.5;
    let savi = (samples.nir - samples.red) * (1 + L) / (samples.nir + samples.red + L);
    let msavi = (2 * samples.nir + 1 - Math.sqrt((2 * samples.nir + 1) * (2 * samples.nir + 1) - 8 * (samples.nir - samples.red))) / 2;
    let ndre = (samples.nir - samples.rededge) / (samples.nir + samples.rededge);
    let Clre = ((samples.nir / samples.rededge)-1);
    let SRre = (samples.nir / samples.rededge);

    return {
        ndvi: [ndvi],
        gndvi: [gndvi],
        evi: [evi],
        savi: [savi],
        msavi: [msavi],
        ndre: [ndre],
        Clre: [Clre],
        SRre: [SRre],
        dataMask: [samples.dataMask]
    };
    }
    """

    # Create a list to hold requests
    ndvi_requests = []

    # Iterate over each row in the GeoDataFrame
    for index, row in gdf.iterrows():
        # Define the time interval for each row (1 day interval around Image_Acquisition_date)
        start_date = (row['Img_date'] - timedelta(days=5)).strftime('%Y-%m-%d')
        end_date = (row['Img_date']).strftime('%Y-%m-%d')

        time_interval = (start_date, end_date)

        # Define the aggregation settings
        aggregation = SentinelHubStatistical.aggregation(
            evalscript=evalscript, time_interval=time_interval, aggregation_interval="P1D", resolution=(resx, resy)
        )

        histogram_calculations = {
            "ndvi": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
            "gndvi": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
            "evi": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
            "savi": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
            "msavi": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
            "ndre": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
            "Clre": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
            "SRre": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}}
        }

        # Define the request
        request = SentinelHubStatistical(
            aggregation=aggregation,
            input_data=[input_data],
            geometry=Geometry(row['Coordinates'], crs=CRS.WGS84),
            calculations=histogram_calculations,
            config=config,
        )
        ndvi_requests.append(request)

    print(f"{len(ndvi_requests)} Statistical API requests prepared!")

    # Download the data
    download_requests = [ndvi_request.download_list[0] for ndvi_request in ndvi_requests]

    client = SentinelHubStatisticalDownloadClient(config=config)

    ndvi_stats = client.download(download_requests)

    print(f"{len(ndvi_stats)} Results from the Statistical API!")

    # Optional: process the results as needed
    # print(ndvi_stats)

    #converting statistical output from above script into df with polgon coordinates
    def stats2df(stats_data, original_index, Polygon_coordinates):
        """Transform Statistical API response into a pandas.DataFrame"""
        df_data = []
    
        for single_data in stats_data["data"]:
            df_entry = {}
            is_valid_entry = True
            df_entry["interval_from"] = parse_time(single_data["interval"]["from"]).date()
            df_entry["interval_to"] = parse_time(single_data["interval"]["to"]).date()
    
            # Add the subplot coordinates and original index to the entry
            df_entry["original_index"] = original_index
            df_entry["Polygon_coordinates"] = Polygon_coordinates
    
            for output_name, output_data in single_data["outputs"].items():
                for band_name, band_values in output_data["bands"].items():
                    band_stats = band_values["stats"]
                    if band_stats["sampleCount"] == band_stats["noDataCount"]:
                        is_valid_entry = False
                        break
    
                    for stat_name, value in band_stats.items():
                        col_name = f"{output_name}_{band_name}_{stat_name}"
                        if stat_name == "percentiles":
                            for perc, perc_val in value.items():
                                perc_col_name = f"{col_name}_{perc}"
                                df_entry[perc_col_name] = perc_val
                        else:
                            df_entry[col_name] = value
    
            if is_valid_entry:
                df_data.append(df_entry)
    
        return pd.DataFrame(df_data)
    
        
    ndvi_dfs = [
        stats2df(polygon_stats, original_index=index, Polygon_coordinates=row.Coordinates)
        for index, (polygon_stats, row) in enumerate(zip(ndvi_stats, gdf.itertuples()))
    ]
    
    combined_df = pd.concat(ndvi_dfs, ignore_index=True)
    print(combined_df)
    
    combined1_df = combined_df
    # Convert interval_to to datetime for proper sorting
    combined1_df["interval_to"] = pd.to_datetime(combined1_df["interval_to"])
    # Assign rank within each original_index group based on interval_to descending
    combined1_df["rank"] = combined1_df.groupby("original_index")["interval_to"].rank(method="dense", ascending=False)
    # Filter only rows where rank == 1
    df_filtered = combined1_df[combined1_df["rank"] == 1]
    # Drop rank column , since it not needed going forward
    df_filtered = df_filtered.drop(columns=["rank"])
    
    gdf.reset_index(drop=True, inplace=True)
    
    #aligning VIs with existing df
    # Assuming 'original_index' is included in both dataframes
    final_df = gdf.merge(df_filtered, left_index=True, right_on='original_index', how='left')
    
    # Drop 'original_index' if not needed anymore
    final_df = final_df.drop(columns=['original_index'])
    print(final_df)


# Weather data extraction

    import pandas as pd
    import rasterio
    import geopandas as gpd
    from shapely.geometry import Polygon
    from rasterstats import zonal_stats
    import requests
    import os

    # Sample DataFrame with 'timestamp' and 'Coordinates'
    # df = pd.read_csv('your_dataframe.csv')  # Replace with your CSV file containing timestamp and Coordinates

    # Create folder to save downloaded TIFFs
    os.makedirs('weather_tiffs', exist_ok=True)

    weather_data = []

    for idx, row in final_df.iterrows():
        # Extract date components
        date_obj = pd.to_datetime(row['Date'])
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        date_str = date_obj.strftime('%Y%m%d')

        # NOAA URL for this date
        tif_url = f"https://water.noaa.gov/resources/downloads/precip/stageIV/{year}/{month}/{day}/nws_precip_1day_{date_str}_conus.tif"
        tif_file = f"weather_tiffs/nws_precip_1day_{date_str}_conus.tif"

        # Download TIFF if not already present
        if not os.path.exists(tif_file):
            print(f"Downloading: {tif_url}")
            response = requests.get(tif_url, stream=True)
            if response.status_code == 200:
                with open(tif_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            else:
                print(f"Failed to download {tif_url}")
                continue  # Skip this iteration if download fails

    

        from shapely import wkt
        coords = row['Farm_Coordinates']

        farm_poly = wkt.loads(coords)     # Convert WKT to Shapely Polygon

        farm_gdf = gpd.GeoDataFrame([{'farm_id': row['unique_id'], 'geometry': farm_poly}], crs='EPSG:4326')

        
        with rasterio.open(tif_file) as src:
            farm_reprojected = farm_gdf.to_crs(src.crs)
            farm_area = farm_reprojected.geometry.area.values[0]
        
            if farm_area < 5_000_000:
                farm_reprojected['geometry'] = farm_reprojected.buffer(5000)
        
            band_results = {'unique_id': row['unique_id'], 'Date': row['Date']}
            for band_num in range(1, 5):
                stats = zonal_stats(
                    farm_reprojected, tif_file,
                    nodata=src.nodata,
                    stats=['sum'],
                    band=band_num,
                    geojson_out=True
                )
                rainfall_sum = stats[0]['properties']['sum']
                rainfall_sum = rainfall_sum if rainfall_sum is not None else 0
                band_results[f'band_{band_num}_sum'] = rainfall_sum
        
            weather_data.append(band_results)



    # Final Weather DataFrame
    weather_df = pd.DataFrame(weather_data)


    weather_df = weather_df.rename(columns={
        'band_1_sum': 'observation_sum',
        'band_2_sum': 'prism_normals_sum',
        'band_3_sum': 'departure_from_normal_sum',
        'band_4_sum': 'percent_of_normal_sum'
    })

    print(weather_df)

    # Optional: Save to CSV
    # weather_df.to_csv('weather_results.csv', index=False)


    # final_df = pd.concat([final_df, weather_df.drop(columns=['unique_id', 'Date'])], axis=1)
    final_df = final_df.merge(weather_df.drop(columns=[]), on=['unique_id', 'Date'],how='left')


    print(final_df.columns)

# print(final_df1.head())



    model_df=final_df[['Plot', 'Strip', 'mean_height', 'Coordinates', 'Date','Farm_Coordinates', 'unique_id',
                    "interval_from","interval_to","savi_B0_mean", "ndvi_B0_mean","msavi_B0_mean",
                    "gndvi_B0_mean","ndre_B0_mean","Clre_B0_mean","SRre_B0_mean",
                    "observation_sum","prism_normals_sum","departure_from_normal_sum","percent_of_normal_sum"  ]]

    model_df.dropna(subset=['mean_height', "savi_B0_mean","ndvi_B0_mean","msavi_B0_mean","gndvi_B0_mean",
                            "ndre_B0_mean","Clre_B0_mean","SRre_B0_mean",
                        "observation_sum","prism_normals_sum","departure_from_normal_sum","percent_of_normal_sum"
                        ], inplace=True)

    model_df.reset_index(drop=True, inplace=True)

    model_df = model_df.rename(columns={
        'mean_height':'PT_Height(cm)',
        'ndvi_B0_mean': 'NDVI_mean',
        'gndvi_B0_mean': 'GNDVI_mean',
        'savi_B0_mean': 'SAVI_mean',
        'msavi_B0_mean': 'MSAVI_mean',
        'ndre_B0_mean': 'NDRE_mean',
        'Clre_B0_mean': 'CLRE_mean',
        'SRre_B0_mean': 'SRre_mean'
        
    })
    model_df['JulianDate'] = pd.to_datetime(model_df['Date'], errors='coerce').dt.dayofyear

    
    return model_df

# ✅ Place this here:
# === Now use the file in the next step ===
clipped_df = pd.read_csv(final_file)

# === Process uploaded file to get VIs and weather ===
model_df = fetch_and_process_farm_data(clipped_df)

# === Save final model output CSV ===
final_model_file = f"Height_VI's_Weather_{emlid_date_str}.csv"
model_df.to_csv(final_model_file, index=False)
print(f"📊 Final model data saved as: {final_model_file}")

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

def extract_field(station_prefix, date, field_element, field_name):
    base_url = (
        "http://agebb.missouri.edu/weather/history/report.asp?"
        f"station_prefix={station_prefix}"
        f"&start_month={date.month}&start_day={date.day}&start_year={date.year}"
        f"&end_month={date.month}&end_day={date.day}&end_year={date.year}"
        "&period_type=1&convert=1"
        f"&field_elements={field_element}"
    )

    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    pre_text = soup.find('pre').get_text()

    pattern = re.compile(r'\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)')
    match = pattern.search(pre_text)
    
    if match:
        return float(match.group(4))
    return None


# Load your CSV
df = pd.read_csv(final_model_file)  # Replace with your actual CSV path
df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' column exists

# Get unique dates
unique_dates = df['Date'].dt.date.unique()

# Station prefix and fields
station_prefix = 'sfm'
field_elements = {
    'Precipitation_inch': 70,
    'Min_Air_Temperature_F': 51,
    'Max_Air_Temperature_F': 3,
    'Avg_Air_Temperature_F': 23
}

# Prepare a mapping for each date to weather values
weather_data = {}

for date in unique_dates:
    values = {}
    for name, code in field_elements.items():
        value = extract_field(station_prefix, pd.to_datetime(date), code, name)
        values[name] = value
    weather_data[date] = values

# Map values to original DataFrame
for name in field_elements.keys():
    df[name] = df['Date'].dt.date.map(lambda d: weather_data.get(d, {}).get(name))

# === Convert temperatures from Fahrenheit to Celsius ===
df['Max_Temp_C'] = ((df['Max_Air_Temperature_F'] - 32) * 5 / 9).round(2)
df['Min_Temp_C'] = ((df['Min_Air_Temperature_F'] - 32) * 5 / 9).round(2)

# === Add Base Temperature ===
df['Base_Temp_C'] = 4.4

# === Compute GDD ===
df['GDD'] = ((df['Max_Temp_C'] + df['Min_Temp_C']) / 2 - df['Base_Temp_C']).apply(lambda x: x if x > 0 else 0).round(2)

# Round necessary columns to 2 decimal places
cols = ['NDVI_mean', 'GNDVI_mean', 'SAVI_mean', 'MSAVI_mean', 'NDRE_mean', 'CLRE_mean', 'SRre_mean']
for col in cols:
    df[col] = df[col].apply(lambda x: f"{x:.3f}")

# # Convert PT height from cm to mm and round to 2 decimal places
# df['PT_Height(mm)'] = (df['PT_Height(cm)']).round(2)
# df['unique_id'] = df['unique_id'].astype(str).str.replace(r'\.0', '', regex=True)


# Desired column order
ordered_cols = [
    'Date', 'JulianDate',
    'Plot', 'Strip', 'Coordinates', 'Farm_Coordinates', 'PT_Height(mm)', 'unique_id',
    'interval_from', 'interval_to',
    'NDVI_mean', 'GNDVI_mean', 'SAVI_mean', 'MSAVI_mean', 'NDRE_mean', 'CLRE_mean', 'SRre_mean',
    'observation_sum', 'prism_normals_sum', 'departure_from_normal_sum', 'percent_of_normal_sum',
    'Precipitation_inch', 'Min_Air_Temperature_F', 'Max_Air_Temperature_F', 'Avg_Air_Temperature_F',
    'Min_Temp_C', 'Max_Temp_C', 'Base_Temp_C', 'GDD'
]

# Reorder columns
df = df[ordered_cols]

# Overwrite CSV file with updated content
df.to_csv(final_model_file, index=False)
print("✅ Final model updated with AGEBB temperature data.")

# === Upload Final Model CSV to specific Google Drive folder ===
upload_model_file = drive.CreateFile({
    'title': final_model_file,
    'parents': [{'id': '1eAlkuHwjsV0VxgB3pjRRbU39PgucE1f5'}]
})
upload_model_file.SetContentFile(final_model_file)
upload_model_file.Upload()
print(f"✅ Final file uploaded to Google Drive folder: {final_model_file}")

