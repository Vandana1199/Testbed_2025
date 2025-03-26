def fetch_and_process_farm_data(farm_id, date):
    import pyodbc
    import pandas as pd
    import geopandas as gpd
    from shapely import wkt
    from shapely.geometry import Polygon
    from datetime import timedelta
    from sentinelhub import (
        CRS, DataCollection, Geometry, SentinelHubStatistical,
        SentinelHubStatisticalDownloadClient, SHConfig, parse_time
    )

       # Sentinel Hub processing
    CLIENT_ID = '670e1809-9266-4a20-9857-d077e19962fb'
    CLIENT_SECRET = 'oK3noJhbwQmgtvyVbSDT9OewGBFLTDyV'
    config = SHConfig()
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

    # collection_id = "f6adb334-9fac-4d3d-bcf7-112e0903bee7" #common bucket
    collection_id = "f1b3b558-17a3-4d40-8768-4870cd74cb06" #testbed_bucket
    PlanetScope_data_collection = DataCollection.define_byoc(collection_id)

    df = pd.read_csv("Emlid_PT_Intergrated_3-15-25.csv")
    df['Coordinates'] = df['Coordinates'].apply(wkt.loads)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Img_date'] = (df['Date']).dt.strftime('%Y-%m-%d')

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
    # print(combined_df)
    
    
    
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
        date_obj = pd.to_datetime(row['timestamp'])
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

        # Parse farm coordinates from the DataFrame (convert string to list if needed)
        coords = row['Coordinates']  # Directly use the list
        farm_poly = Polygon(coords)
        farm_gdf = gpd.GeoDataFrame([{'farm_id': row['GroupID'], 'geometry': farm_poly}], crs='EPSG:4326')
        
        with rasterio.open(tif_file) as src:
            farm_reprojected = farm_gdf.to_crs(src.crs)
            farm_area = farm_reprojected.geometry.area.values[0]
        
            if farm_area < 5_000_000:
                farm_reprojected['geometry'] = farm_reprojected.buffer(5000)
        
            band_results = {'GroupID': row['GroupID'], 'timestamp': row['timestamp']}
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


    final_df = pd.concat([final_df, weather_df.drop(columns=['GroupID', 'timestamp'])], axis=1)

# print(final_df1.head())



    model_df=final_df[["timestamp","CenterLatitude","CenterLongitude","mean_height","Coordinates",
                    "interval_from","interval_to","savi_B0_mean","evi_B0_mean","ndvi_B0_mean","msavi_B0_mean",
                    "gndvi_B0_mean","ndre_B0_mean","Clre_B0_mean","SRre_B0_mean",
                    "observation_sum","prism_normals_sum","departure_from_normal_sum","percent_of_normal_sum"  ]]

    model_df.dropna(subset=['mean_height', "savi_B0_mean","evi_B0_mean","ndvi_B0_mean","msavi_B0_mean","gndvi_B0_mean",
                            "ndre_B0_mean","Clre_B0_mean","SRre_B0_mean",
                        "observation_sum","prism_normals_sum","departure_from_normal_sum","percent_of_normal_sum"
                        ], inplace=True)

    model_df.reset_index(drop=True, inplace=True)

    model_df = model_df.rename(columns={
        'mean_height':'MeanHeight(mm)',
        'evi_B0_mean': 'EVI_mean',
        'ndvi_B0_mean': 'NDVI_mean',
        'gndvi_B0_mean': 'GNDVI_mean',
        'savi_B0_mean': 'SAVI_mean',
        'msavi_B0_mean': 'MSAVI_mean',
        'ndre_B0_mean': 'NDRE_mean',
        'Clre_B0_mean': 'CLRE_mean',
        'SRre_B0_mean': 'SRre_mean'
        
    })
    model_df['JulianDate'] = pd.to_datetime(model_df['timestamp'], errors='coerce').dt.dayofyear

    
    return model_df



# # Example usage:
result = fetch_and_process_farm_data( farm_id = 764, date = '10/07/2024' )
print(result)
