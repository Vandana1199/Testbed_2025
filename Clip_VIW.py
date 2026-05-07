from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd
import re
import os
import geopandas as gpd
from shapely.geometry import Polygon
from shapely import wkt
from datetime import timedelta
import numpy as np
import requests
from bs4 import BeautifulSoup

# ============================================================
# GOOGLE DRIVE AUTH
# ============================================================

gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.txt")

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

gauth.SaveCredentialsFile("mycreds.txt")
drive = GoogleDrive(gauth)

# ============================================================
# DRIVE FOLDER SETUP
# ============================================================

folder_id = "1G4s-qT5_VYZSu3PVU0vCYxo6_NWqA_0q"  # TESTBED_2026_PT EMLID DATA FOLDER ID

file_list = drive.ListFile({
    'q': f"'{folder_id}' in parents and trashed=false"
}).GetList()

for file in file_list:
    print(f"Title: {file['title']}, ID: {file['id']}")

emlid_files = []
pt_files = []
testbed_file = None

pattern_emlid = re.compile(r'^EMLID_(\d+\.\d+\.\d+)\.csv$')
pattern_pt = re.compile(r'^PT_(\d+\.\d+\.\d+)\.csv$')
testbed_filename = 'stripcorners.csv'

def extract_date_key(filename):
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', filename)
    if match:
        month, day, year = map(int, match.groups())
        return (year, month, day)
    return (0, 0, 0)

for file in file_list:
    title = file['title']
    if title == testbed_filename:
        testbed_file = (title, file['id'])
    elif pattern_emlid.match(title):
        emlid_files.append((title, file['id']))
    elif pattern_pt.match(title):
        pt_files.append((title, file['id']))

# ============================================================
# DOWNLOAD FILES
# ============================================================

if testbed_file:
    if not os.path.exists(testbed_file[0]):
        print(f"📥 Downloading Testbed file: {testbed_file[0]}")
        file = drive.CreateFile({'id': testbed_file[1]})
        file.GetContentFile(testbed_file[0])
    else:
        print(f"✅ Testbed file already exists locally: {testbed_file[0]}")
else:
    raise FileNotFoundError("❌ Testbed file not found in Drive.")

if emlid_files:
    latest_emlid = max(emlid_files, key=lambda x: extract_date_key(x[0]))
    print(f"📥 Latest EMLID file: {latest_emlid[0]}")
    file = drive.CreateFile({'id': latest_emlid[1]})
    file.GetContentFile(latest_emlid[0])
    GPS_raw = pd.read_csv(latest_emlid[0])
else:
    raise FileNotFoundError("❌ No EMLID files found.")

if pt_files:
    latest_pt = max(pt_files, key=lambda x: extract_date_key(x[0]))
    print(f"📥 Latest PT file: {latest_pt[0]}")
    file = drive.CreateFile({'id': latest_pt[1]})
    file.GetContentFile(latest_pt[0])
    PT_raw = pd.read_csv(latest_pt[0])
else:
    raise FileNotFoundError("❌ No PT files found.")

emlid_date_str = re.search(r'(\d+\.\d+\.\d+)', latest_emlid[0]).group(1)

# ============================================================
# PROCESS PT + EMLID
# ============================================================

PT_raw['datetime_clean'] = PT_raw['datetime'].str.replace(r'^[A-Za-z]+ ', '', regex=True)
PT_raw['datetime_clean'] = PT_raw['datetime_clean'].str.replace(r' GMT.*', '', regex=True)

PT_raw['Timestamp'] = pd.to_datetime(
    PT_raw['datetime_clean'],
    format='%b %d %Y %H:%M:%S',
    errors='coerce'
)

PT_raw.drop(columns=['datetime_clean'], inplace=True)

PT_raw['datetime'] = pd.to_datetime(PT_raw['Timestamp'], errors='coerce')
PT_raw['date'] = PT_raw['datetime'].dt.date
PT_raw['time'] = pd.to_timedelta(PT_raw['datetime'].dt.strftime('%H:%M:%S'))

GPS = GPS_raw.rename(columns={"longitude(deg)": "X", "latitude(deg)": "Y"})
GPS = GPS[['X', 'Y', 'GPST']]

# time_diff = timedelta(hours=6, minutes=0, seconds=24)  # CST
time_diff = timedelta(hours=5, minutes=0, seconds=24)  # CDT

PT_scaled = PT_raw.sort_values('time').copy()
PT_scaled['n'] = range(1, len(PT_scaled) + 1)
PT_scaled['scaled_time'] = PT_scaled['datetime'] + time_diff

PT_merge = PT_scaled[['rawdistance', 'scaled_time', 'tare', 'date']].copy()
PT_merge['scaled_time2'] = PT_merge['scaled_time'].astype('int64') // 10**9
PT_merge['ID'] = PT_merge.groupby('scaled_time2').cumcount() + 1
PT_merge['scaled_time'] = PT_merge['scaled_time'].dt.strftime('%H:%M:%S')

GPS['scaled_time'] = pd.to_datetime(
    GPS['GPST'],
    format="%I:%M:%S %p",
    errors='coerce'
).dt.strftime("%H:%M:%S")

GPS['scaled_time2'] = GPS['scaled_time'].str.replace(':', '', regex=False).astype(float)
GPS['ID'] = GPS.groupby('scaled_time2').cumcount() + 1

GPS_merge = GPS[['X', 'Y', 'scaled_time', 'ID']]

merged_data = pd.merge(
    PT_merge,
    GPS_merge,
    how='outer',
    on=['scaled_time', 'ID']
)

merged_data = merged_data.ffill()

merged_filtered = merged_data[
    ['rawdistance', 'X', 'Y', 'scaled_time', 'tare', 'date']
].rename(columns={'scaled_time': 'time'})

merged_filtered = merged_filtered.dropna(subset=['rawdistance', 'X', 'Y', 'tare'])

# ============================================================
# CREATE STRIP POLYGONS
# ============================================================

corners = pd.read_csv(testbed_filename).rename(columns={'POINT_X': 'x', 'POINT_Y': 'y'})
corners.rename(columns={'PlotArea': 'Plot'}, inplace=True)

polygon_list = []
ids = []
strips = []

for (plot_id, strip_id) in corners[['Plot', 'Strip']].drop_duplicates().values:
    polygon_data = corners[
        (corners['Plot'] == plot_id) &
        (corners['Strip'] == strip_id)
    ].copy()

    closed_polygon = pd.concat(
        [polygon_data, polygon_data.iloc[[0]]],
        ignore_index=True
    )

    coords = closed_polygon[['x', 'y']].values.tolist()
    polygon = Polygon(coords)

    polygon_list.append(polygon)
    ids.append(str(plot_id))
    strips.append(str(strip_id))

polygon_gdf = gpd.GeoDataFrame(
    {'Plot': ids, 'Strip': strips, 'geometry': polygon_list},
    crs="EPSG:4326"
)

# ============================================================
# SPATIAL JOIN PT POINTS WITH STRIPS
# ============================================================

merged_filtered_clean = merged_filtered.dropna(subset=["X", "Y"]).copy()

PT_gdf = gpd.GeoDataFrame(
    merged_filtered_clean,
    geometry=gpd.points_from_xy(merged_filtered_clean["X"], merged_filtered_clean["Y"]),
    crs="EPSG:4326"
)

plot_intersect_full = gpd.sjoin(
    PT_gdf,
    polygon_gdf,
    how='inner',
    predicate='within'
)

plot_intersect_full['Plot'] = plot_intersect_full['Plot'].astype(str)
plot_intersect_full['Strip'] = plot_intersect_full['Strip'].astype(str)

plot_intersect_full = plot_intersect_full.sort_values(['Plot', 'Strip', 'time'])

dropped_start = plot_intersect_full.groupby(['Plot', 'Strip']).head(4)
dropped_end = plot_intersect_full.groupby(['Plot', 'Strip']).tail(6)

plot_intersect = (
    plot_intersect_full.groupby(['Plot', 'Strip'])
    .apply(lambda x: x.iloc[4:-6] if len(x) > 10 else x.iloc[0:0])
    .reset_index(drop=True)
)

dropped_start.to_csv("Dropped_First4_Readings.csv", index=False)
dropped_end.to_csv("Dropped_Last6_Readings.csv", index=False)

plot_intersect_file = "Raw_PT_Data.csv"
plot_intersect.to_csv(plot_intersect_file, index=False)

raw_pt_file = f"Raw_PT_Data_{emlid_date_str}.csv"

upload_plot_intersect_file = drive.CreateFile({
    'title': raw_pt_file,
    'parents': [{'id': '1iER32B8BprkMqNjECaEcKOCZ7B1b7dWc'}]
})

upload_plot_intersect_file.SetContentFile(plot_intersect_file)
upload_plot_intersect_file.Upload()

print(f"✅ Raw PT file uploaded as: {raw_pt_file}")

# ============================================================
# HEIGHT CALCULATION: MEAN + MEDIAN
# ============================================================

result = plot_intersect.merge(
    polygon_gdf[['Plot', 'Strip', 'geometry']],
    on=['Plot', 'Strip'],
    how='left'
)

result = result.rename(
    columns={
        "geometry_x": "geometry",
        "geometry_y": "Coordinates",
        "date": "Date"
    }
)

result['PTdata_cm'] = result['rawdistance']
result['grass_height_cm'] = (result['tare'] - result['PTdata_cm']) * 0.0859536

Emlid_PT_Intergrated = result.groupby(['Plot', 'Strip']).agg(
    mean_height=('grass_height_cm', 'mean'),
    median_height=('grass_height_cm', 'median'),
    Coordinates=('Coordinates', 'first'),
    Date=('Date', 'first')
).reset_index()

# Emlid_PT_Intergrated["Farm_Coordinates"] = (
#     "POLYGON ((-92.27126447977226 38.905762369582106, "
#     "-92.27126447977226 38.90536179063761, "
#     "-92.26988422615567 38.90536179063761, "
#     "-92.26988422615567 38.905762369582106, "
#     "-92.27126447977226 38.905762369582106))"
# ) ##TESTBED FARMLEVEL POLYGON

Emlid_PT_Intergrated["Farm_Coordinates"] = (
    POLYGON ((
-92.2615122 38.8894407,
-92.2615494 38.888182,
-92.2587427 38.8880663,
-92.2587272 38.8893611,
-92.2600779 38.8894118,
-92.2615122 38.8894407
))
)

Emlid_PT_Intergrated["unique_id"] = (
    Emlid_PT_Intergrated["Plot"].astype(str) +
    "_" +
    Emlid_PT_Intergrated["Strip"].astype(str)
)

intermediate_file = 'Emlid_PT_Intergrated.csv'
final_file = f"Emlid_PT_Intergrated_{emlid_date_str}.csv"

Emlid_PT_Intergrated.to_csv(intermediate_file, index=False)

if os.path.exists(final_file):
    os.remove(final_file)

os.rename(intermediate_file, final_file)

upload_file = drive.CreateFile({
    'title': final_file,
    'parents': [{'id': '1bNyD7WR7Zbk-OKtNdpDjta-LuAGLkSk2'}]
})

upload_file.SetContentFile(final_file)
upload_file.Upload()

print(f"✅ EMLID + PT integrated file uploaded as: {final_file}")

# ============================================================
# SENTINEL HUB + NOAA WEATHER FUNCTION
# ============================================================

def fetch_and_process_farm_data(clipped_df):
    import pandas as pd
    import geopandas as gpd
    from shapely import wkt
    from datetime import timedelta
    from sentinelhub import (
        CRS, DataCollection, Geometry, SentinelHubStatistical,
        SentinelHubStatisticalDownloadClient, SHConfig, parse_time
    )
    import rasterio
    from rasterstats import zonal_stats
    import requests
    import os

    if clipped_df.empty:
        raise ValueError("Clipped DataFrame is empty")

    df = clipped_df.copy()

    CLIENT_ID = '670e1809-9266-4a20-9857-d077e19962fb'
    CLIENT_SECRET = 'O9rK3noJhbwQmgtvyVbSDT9OewGBFLTDyV'

    config = SHConfig()
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

    # collection_id = "0b8f4bdd-390d-4665-abd7-39ff23cfd44b" TESTBED 2026 Collection bucket
    collection_id = "bbe7e0d7-ed45-4482-8d3e-c96288cde87c"  ##CERELRYE Collectin bucket
    PlanetScope_data_collection = DataCollection.define_byoc(collection_id)

    df['Coordinates'] = df['Coordinates'].apply(wkt.loads)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Img_date'] = df['Date']

    gdf = gpd.GeoDataFrame(df, geometry='Coordinates', crs="EPSG:4326")

    input_data = SentinelHubStatistical.input_data(
        data_collection=PlanetScope_data_collection
    )

    resx = 0.00001
    resy = 0.00001

    evalscript = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: [
                    "red", "green", "blue", "nir", "rededge",
                    "dataMask", "clear"
                ]
            }],
            output: [
                { id: "ndvi", bands: 1 },
                { id: "gndvi", bands: 1 },
                { id: "evi", bands: 1 },
                { id: "savi", bands: 1 },
                { id: "msavi", bands: 1 },
                { id: "ndre", bands: 1 },
                { id: "Clre", bands: 1 },
                { id: "SRre", bands: 1 },
                { id: "red", bands: 1 },
                { id: "green", bands: 1 },
                { id: "blue", bands: 1 },
                { id: "nir", bands: 1 },
                { id: "rededge", bands: 1 },
                { id: "dataMask", bands: 1 }
            ]
        }
    }

    function evaluatePixel(samples) {
        let s = 10000;
        let ndvi = (samples.nir - samples.red) / (samples.nir + samples.red);
        let gndvi = (samples.nir - samples.green) / (samples.nir + samples.green);
        let evi = 2.5 * (samples.nir - samples.red) / (samples.nir + 6.0 * samples.red - 7.5 * samples.blue + 10000);
        let L = 0.5;
        let savi = (samples.nir - samples.red) * (1 + L) / (samples.nir + samples.red + L * 10000);
        let msavi = (2 * (samples.nir/s) + 1 - Math.sqrt((2 * (samples.nir/s) + 1) ** 2 - 8 * ((samples.nir/s) - (samples.red/s)))) / 2;
        let ndre = (samples.nir - samples.rededge) / (samples.nir + samples.rededge);
        let Clre = (samples.nir / samples.rededge) - 1;
        let SRre = samples.nir / samples.rededge;

        return {
            ndvi: [ndvi],
            gndvi: [gndvi],
            evi: [evi],
            savi: [savi],
            msavi: [msavi],
            ndre: [ndre],
            Clre: [Clre],
            SRre: [SRre],
            red: [samples.red/s],
            green: [samples.green/s],
            blue: [samples.blue/s],
            nir: [samples.nir/s],
            rededge: [samples.rededge/s],
            dataMask: [samples.dataMask]
        };
    }
    """

    calculations = {
        band: {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}}
        for band in [
            "ndvi", "gndvi", "evi", "savi", "msavi", "ndre", "Clre", "SRre",
            "red", "green", "blue", "nir", "rededge"
        ]
    }

    ndvi_requests = []

    for index, row in gdf.iterrows():
        start_date = (row['Img_date'] - timedelta(days=5)).strftime('%Y-%m-%d')
        end_date = row['Img_date'].strftime('%Y-%m-%d')

        aggregation = SentinelHubStatistical.aggregation(
            evalscript=evalscript,
            time_interval=(start_date, end_date),
            aggregation_interval="P1D",
            resolution=(resx, resy)
        )

        request = SentinelHubStatistical(
            aggregation=aggregation,
            input_data=[input_data],
            geometry=Geometry(row['Coordinates'], crs=CRS.WGS84),
            calculations=calculations,
            config=config
        )

        ndvi_requests.append(request)

    print(f"{len(ndvi_requests)} Statistical API requests prepared!")

    download_requests = [req.download_list[0] for req in ndvi_requests]
    client = SentinelHubStatisticalDownloadClient(config=config)
    ndvi_stats = client.download(download_requests)

    print(f"{len(ndvi_stats)} Results from Statistical API!")

    def stats2df(stats_data, original_index, polygon_coordinates):
        df_data = []

        for single_data in stats_data["data"]:
            df_entry = {}
            is_valid_entry = True

            df_entry["interval_from"] = parse_time(single_data["interval"]["from"]).date()
            df_entry["interval_to"] = parse_time(single_data["interval"]["to"]).date()
            df_entry["original_index"] = original_index
            df_entry["Polygon_coordinates"] = polygon_coordinates

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
                                df_entry[f"{col_name}_{perc}"] = perc_val
                        else:
                            df_entry[col_name] = value

            if is_valid_entry:
                df_data.append(df_entry)

        return pd.DataFrame(df_data)

    ndvi_dfs = [
        stats2df(stats, index, row.Coordinates)
        for index, (stats, row) in enumerate(zip(ndvi_stats, gdf.itertuples()))
    ]

    combined_df = pd.concat(ndvi_dfs, ignore_index=True)

    combined_df["interval_to"] = pd.to_datetime(combined_df["interval_to"])
    combined_df["rank"] = combined_df.groupby("original_index")["interval_to"].rank(
        method="dense",
        ascending=False
    )

    df_filtered = combined_df[combined_df["rank"] == 1].drop(columns=["rank"])

    gdf.reset_index(drop=True, inplace=True)

    final_df = gdf.merge(
        df_filtered,
        left_index=True,
        right_on='original_index',
        how='left'
    )

    final_df = final_df.drop(columns=['original_index'])

    # ========================================================
    # NOAA STAGE IV WEATHER
    # ========================================================

    os.makedirs('weather_tiffs', exist_ok=True)

    weather_data = []

    for idx, row in final_df.iterrows():
        date_obj = pd.to_datetime(row['Date'])
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        date_str = date_obj.strftime('%Y%m%d')

        tif_url = f"https://water.noaa.gov/resources/downloads/precip/stageIV/{year}/{month}/{day}/nws_precip_1day_{date_str}_conus.tif"
        tif_file = f"weather_tiffs/nws_precip_1day_{date_str}_conus.tif"

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
                continue

        farm_poly = wkt.loads(row['Farm_Coordinates'])
        farm_gdf = gpd.GeoDataFrame(
            [{'farm_id': row['unique_id'], 'geometry': farm_poly}],
            crs='EPSG:4326'
        )

        with rasterio.open(tif_file) as src:
            farm_reprojected = farm_gdf.to_crs(src.crs)
            farm_area = farm_reprojected.geometry.area.values[0]

            if farm_area < 5_000_000:
                farm_reprojected['geometry'] = farm_reprojected.buffer(5000)

            band_results = {'unique_id': row['unique_id'], 'Date': row['Date']}

            for band_num in range(1, 5):
                stats = zonal_stats(
                    farm_reprojected,
                    tif_file,
                    nodata=src.nodata,
                    stats=['sum'],
                    band=band_num,
                    geojson_out=True
                )

                rainfall_sum = stats[0]['properties']['sum']
                rainfall_sum = rainfall_sum if rainfall_sum is not None else 0
                band_results[f'band_{band_num}_sum'] = rainfall_sum

            weather_data.append(band_results)

    weather_df = pd.DataFrame(weather_data)

    weather_df = weather_df.rename(columns={
        'band_1_sum': 'observation_sum',
        'band_2_sum': 'prism_normals_sum',
        'band_3_sum': 'departure_from_normal_sum',
        'band_4_sum': 'percent_of_normal_sum'
    })

    final_df = final_df.merge(
        weather_df,
        on=['unique_id', 'Date'],
        how='left'
    )

    model_df = final_df[[
        'Plot', 'Strip',
        'mean_height', 'median_height',
        'Coordinates', 'Date', 'Farm_Coordinates', 'unique_id',
        'interval_from', 'interval_to',
        'savi_B0_mean', 'ndvi_B0_mean', 'msavi_B0_mean', 'evi_B0_mean',
        'gndvi_B0_mean', 'ndre_B0_mean', 'Clre_B0_mean', 'SRre_B0_mean',
        'red_B0_mean', 'green_B0_mean', 'blue_B0_mean', 'nir_B0_mean', 'rededge_B0_mean',
        'observation_sum', 'prism_normals_sum', 'departure_from_normal_sum', 'percent_of_normal_sum'
    ]]

    model_df = model_df.dropna(subset=[
        'mean_height', 'median_height',
        'savi_B0_mean', 'ndvi_B0_mean', 'msavi_B0_mean', 'gndvi_B0_mean',
        'ndre_B0_mean', 'Clre_B0_mean', 'SRre_B0_mean', 'evi_B0_mean',
        'red_B0_mean', 'green_B0_mean', 'blue_B0_mean', 'nir_B0_mean', 'rededge_B0_mean',
        'observation_sum', 'prism_normals_sum', 'departure_from_normal_sum', 'percent_of_normal_sum'
    ]).reset_index(drop=True)

    model_df = model_df.rename(columns={
        'mean_height': 'PT_Mean_Height(cm)',
        'median_height': 'PT_Median_Height(cm)',
        'ndvi_B0_mean': 'NDVI_mean',
        'gndvi_B0_mean': 'GNDVI_mean',
        'evi_B0_mean': 'EVI_mean',
        'savi_B0_mean': 'SAVI_mean',
        'msavi_B0_mean': 'MSAVI_mean',
        'ndre_B0_mean': 'NDRE_mean',
        'Clre_B0_mean': 'CLRE_mean',
        'SRre_B0_mean': 'SRre_mean',
        'red_B0_mean': 'red_mean',
        'green_B0_mean': 'green_mean',
        'blue_B0_mean': 'blue_mean',
        'nir_B0_mean': 'nir_mean',
        'rededge_B0_mean': 'rededge_mean'
    })

    model_df['JulianDate'] = pd.to_datetime(
        model_df['Date'],
        errors='coerce'
    ).dt.dayofyear

    return model_df

# ============================================================
# AGEBB WEATHER FUNCTIONS
# ============================================================

def extract_field(station_prefix, date, field_element, field_name):
    base_url = (
        "http://agebb.missouri.edu/weather/history/report.asp?"
        f"station_prefix={station_prefix}"
        f"&start_month={date.month}&start_day={date.day}&start_year={date.year}"
        f"&end_month={date.month}&end_day={date.day}&end_year={date.year}"
        "&period_type=1&convert=1"
        f"&field_elements={field_element}"
    )

    try:
        response = requests.get(base_url, timeout=30)
        soup = BeautifulSoup(response.content, 'html.parser')
        pre = soup.find('pre')

        if pre is None:
            return None

        pre_text = pre.get_text()
        pattern = re.compile(r'\s+(\d+)\s+(\d+)\s+(\d+)\s+([-+]?\d*\.?\d+)')
        match = pattern.search(pre_text)

        if match:
            return float(match.group(4))

    except Exception as e:
        print(f"AGEBB error for {date.date()} - {field_name}: {e}")

    return None

def add_agebb_rolling_weather(df, station_prefix='sfm'):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    field_elements = {
        'Precipitation_inch': 70,
        'Min_Air_Temperature_F': 51,
        'Max_Air_Temperature_F': 3,
        'Avg_Air_Temperature_F': 23
    }

    unique_dates = pd.to_datetime(df['Date'].dt.date.unique())

    start_date = unique_dates.min() - pd.Timedelta(days=21)
    end_date = unique_dates.max()

    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    weather_rows = []

    for date in all_dates:
        row = {'Date': date}

        for name, code in field_elements.items():
            row[name] = extract_field(station_prefix, date, code, name)

        weather_rows.append(row)

    weather_daily = pd.DataFrame(weather_rows)
    weather_daily['Date'] = pd.to_datetime(weather_daily['Date'])

    weather_daily['Max_Temp_C'] = (
        (weather_daily['Max_Air_Temperature_F'] - 32) * 5 / 9
    ).round(2)

    weather_daily['Min_Temp_C'] = (
        (weather_daily['Min_Air_Temperature_F'] - 32) * 5 / 9
    ).round(2)

    weather_daily['Base_Temp_C'] = 4.4

    weather_daily['GDD'] = (
        ((weather_daily['Max_Temp_C'] + weather_daily['Min_Temp_C']) / 2)
        - weather_daily['Base_Temp_C']
    ).apply(lambda x: x if pd.notna(x) and x > 0 else 0).round(2)

    weather_daily = weather_daily.sort_values('Date').set_index('Date')

    for window in [7, 14, 21]:
        weather_daily[f'precip_{window}d_sum_in'] = (
            weather_daily['Precipitation_inch']
            .rolling(f'{window}D', closed='left')
            .sum()
        )

        weather_daily[f'gdd_{window}d_sum'] = (
            weather_daily['GDD']
            .rolling(f'{window}D', closed='left')
            .sum()
        )

        weather_daily[f'tavg_{window}d_avg_F'] = (
            weather_daily['Avg_Air_Temperature_F']
            .rolling(f'{window}D', closed='left')
            .mean()
        )

        weather_daily[f'tmin_{window}d_avg_F'] = (
            weather_daily['Min_Air_Temperature_F']
            .rolling(f'{window}D', closed='left')
            .mean()
        )

        weather_daily[f'tmax_{window}d_avg_F'] = (
            weather_daily['Max_Air_Temperature_F']
            .rolling(f'{window}D', closed='left')
            .mean()
        )

    weather_features = weather_daily.reset_index()

    weather_keep_cols = [
        'Date',

        'precip_7d_sum_in', 'gdd_7d_sum',
        'tavg_7d_avg_F', 'tmin_7d_avg_F', 'tmax_7d_avg_F',

        'precip_14d_sum_in', 'gdd_14d_sum',
        'tavg_14d_avg_F', 'tmin_14d_avg_F', 'tmax_14d_avg_F',

        'precip_21d_sum_in', 'gdd_21d_sum',
        'tavg_21d_avg_F', 'tmin_21d_avg_F', 'tmax_21d_avg_F'
    ]

    df = df.merge(
        weather_features[weather_keep_cols],
        on='Date',
        how='left'
    )

    return df

# ============================================================
# RUN FINAL PROCESS
# ============================================================

clipped_df = pd.read_csv(final_file)

model_df = fetch_and_process_farm_data(clipped_df)

final_model_file = f"Height_VIs_Weather_{emlid_date_str}.csv"

model_df.to_csv(final_model_file, index=False)
print(f"📊 Initial model data saved as: {final_model_file}")

# ============================================================
# ADD AGEBB 7/14/21 DAY WEATHER FEATURES
# ============================================================

df = pd.read_csv(final_model_file)
df['Date'] = pd.to_datetime(df['Date'])

df = add_agebb_rolling_weather(df, station_prefix='sfm')

# ============================================================
# FINAL FORMATTING
# ============================================================

vi_cols = [
    'NDVI_mean', 'GNDVI_mean', 'EVI_mean', 'SAVI_mean', 'MSAVI_mean',
    'NDRE_mean', 'CLRE_mean', 'SRre_mean',
    'red_mean', 'green_mean', 'blue_mean', 'nir_mean', 'rededge_mean'
]

for col in vi_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').round(3)

df['PT_Mean_Height(mm)'] = df['PT_Mean_Height(cm)'].round(2)
df['PT_Median_Height(mm)'] = df['PT_Median_Height(cm)'].round(2)

df['unique_id'] = df['unique_id'].astype(str).str.replace(r'\.0', '', regex=True)

ordered_cols = [
    'Date', 'JulianDate',
    'Plot', 'Strip', 'Coordinates', 'Farm_Coordinates',
    'PT_Mean_Height(mm)', 'PT_Median_Height(mm)', 'unique_id',
    'interval_from', 'interval_to',

    'NDVI_mean', 'GNDVI_mean', 'EVI_mean', 'SAVI_mean', 'MSAVI_mean',
    'NDRE_mean', 'CLRE_mean', 'SRre_mean',
    'red_mean', 'green_mean', 'blue_mean', 'nir_mean', 'rededge_mean',

    'observation_sum', 'prism_normals_sum',
    'departure_from_normal_sum', 'percent_of_normal_sum',

    'precip_7d_sum_in', 'gdd_7d_sum',
    'tavg_7d_avg_F', 'tmin_7d_avg_F', 'tmax_7d_avg_F',

    'precip_14d_sum_in', 'gdd_14d_sum',
    'tavg_14d_avg_F', 'tmin_14d_avg_F', 'tmax_14d_avg_F',

    'precip_21d_sum_in', 'gdd_21d_sum',
    'tavg_21d_avg_F', 'tmin_21d_avg_F', 'tmax_21d_avg_F'
]

existing_ordered_cols = [col for col in ordered_cols if col in df.columns]
df = df[existing_ordered_cols]

df.to_csv(final_model_file, index=False)

print("✅ Final model updated with mean height, median height, and AGEBB rolling weather features.")
print(f"✅ Final saved file: {final_model_file}")

# ============================================================
# UPLOAD FINAL FILE TO GOOGLE DRIVE
# ============================================================

upload_model_file = drive.CreateFile({
    'title': final_model_file,
    'parents': [{'id': '13Ljj7woD1lsPplBc7EVocKy-jDhXiV7-'}]
})

upload_model_file.SetContentFile(final_model_file)
upload_model_file.Upload()

print(f"✅ Final file uploaded to Google Drive folder: {final_model_file}")
