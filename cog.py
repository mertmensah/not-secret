import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pydeck as pdk
from math import radians, sin, cos, sqrt, atan2
import boto3
import io

# LOCALbring in ZIP Codes from external source, to match coordinates to orders
zips = pd.read_csv('zip_codes_us.csv')


# Define S3 bucket and file key
#s3_bucket = 'optiscs3'
#s3_key = 'scriptsNdata/zip_codes_us.csv'

# Initialize a session using Amazon S3
#s3 = boto3.client('s3')

# Download the file from S3 into a bytes object
#obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
#data = obj['Body'].read()

# Use pandas to read the CSV file from the bytes object
zips = pd.read_csv(io.BytesIO(data))


# cleaning
zips['zip'] = zips['zip'].astype(str) # covert type
zips['zip'] = zips['zip'].astype(str).str.zfill(5) # add leading 0 until there are 5 char.s 

# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="opti-Network: Center of Gravity Solver", page_icon=":articulated_lorry:")

st.sidebar.title("Parameters")
#st.logo('emailICON.jpg')

st.title('Center of Gravity Solver')
'Upload your demand data to discover your optimal distribution network.'


# Your Mapbox API token
mapbox_api_token = ""  # Replace 

# Sidebar configuration
with st.sidebar:
#    st.header("Configuration")
    num_cogs = st.slider('1. Number of Centers of Gravity', min_value=1, max_value=10, value=3)
    st.write("**Note:** Additional nodes will impact solution time.")
    daily_driver_distance = st.slider('2. Daily Driver Distance Coverage (km)', 0, 1500, 800)
    st.write("800 km is widely used as a long-haul ground daily coverage estimate")

# File upload
uploaded_file = st.file_uploader("Upload Demand Data", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df_open_orders = pd.read_csv(uploaded_file, dtype={'zip_code': 'str', 'weight': 'float'})
        elif uploaded_file.name.endswith('.xlsx'):
            df_open_orders = pd.read_excel(uploaded_file, dtype={'zip_code': 'str', 'weight': 'float'})

        if 'zip_code' not in df_open_orders.columns or 'weight' not in df_open_orders.columns:
            st.error("The uploaded file must contain both (and only) these two columns: 'zip_code' and 'weight'.")
        else:
            verified_zips = []
            unmatched_zips = []
            for zip_code in df_open_orders['zip_code'].dropna().unique():
                if zip_code in zips['zip'].values:
                    verified_zips.append(zip_code)
                else:
                    unmatched_zips.append(zip_code)

            # Show uploaded data in an expandere
            with st.expander("Uploaded Demand Data"):
                st.write(df_open_orders)

            if unmatched_zips:
                with st.expander("Unmatched ZIP Codes"):
                    st.write(unmatched_zips)

            # Filter the demand data for verified ZIP codes
            df_verified = df_open_orders[df_open_orders['zip_code'].isin(verified_zips)]

            if not df_verified.empty:
                # Merge data with ZIP coordinates
                data = pd.merge(df_verified, zips[['zip', 'longitude', 'latitude']], left_on='zip_code', right_on='zip', how='left')

                # Initialize K-Means with weighted clustering centers
                coordinates = data[['longitude', 'latitude']].values
                weights = data['weight'].values
                kmeans = KMeans(n_clusters=num_cogs, random_state=42)
                kmeans.fit(coordinates, sample_weight=weights)
                data['cluster'] = kmeans.labels_

                # Calculate weighted centers of gravity for each cluster
                weighted_cog_data = []
                for cluster in range(num_cogs):
                    cluster_data = data[data['cluster'] == cluster]
                    weighted_lon = np.average(cluster_data['longitude'], weights=cluster_data['weight'])
                    weighted_lat = np.average(cluster_data['latitude'], weights=cluster_data['weight'])
                    weighted_cog_data.append({'longitude': weighted_lon, 'latitude': weighted_lat, 'cluster': cluster})

                cog_df = pd.DataFrame(weighted_cog_data )

                ## Get Cities that match the CoGs
                # Sort zips by 'irs_estimated_population' in descending order
                zips = zips.sort_values(by='irs_estimated_population', ascending=False)

                # Round the coordinates to the second decimal place
                cog_df['rounded_longitude'] = cog_df['longitude'].round(0)
                cog_df['rounded_latitude'] = cog_df['latitude'].round(0)
                zips['rounded_longitude'] = zips['longitude'].round(0)
                zips['rounded_latitude'] = zips['latitude'].round(0)

                # Drop duplicates in zips to only take the first matching value after sorting by population
                zips = zips.drop_duplicates(subset=['rounded_latitude', 'rounded_longitude'], keep='first')

                # Merge cog_df with zips to get the city and state information
                cog_df = pd.merge(cog_df, zips[['rounded_latitude', 'rounded_longitude', 'primary_city', 'state']], 
                                  how='left', 
                                  left_on=['rounded_latitude', 'rounded_longitude'], 
                                  right_on=['rounded_latitude', 'rounded_longitude'])

                # Rename the columns
                cog_df.rename(columns={'primary_city': 'cog_city', 'state': 'cog_state'}, inplace=True)

                # Drop the rounded coordinate columns
                cog_df.drop(columns=['rounded_longitude', 'rounded_latitude'], inplace=True)

                # Create a DataFrame for the arcs
                arcs_data = pd.merge(data, cog_df, left_on='cluster', right_on='cluster', suffixes=('_order', '_cog'))

                # Haversine formula to calculate the distance between two coordinates
                def haversine(lon1, lat1, lon2, lat2):
                    R = 6371  # Earth radius in kilometers
                    dlon = radians(lon2 - lon1)
                    dlat = radians(lat2 - lat1)
                    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
                    c = 2 * atan2(sqrt(a), sqrt(1 - a))
                    distance = R * c
                    return distance

                # Calculate distances and add to the DataFrame
                arcs_data['distance_km'] = arcs_data.apply(lambda row: haversine(row['longitude_cog'], row['latitude_cog'], row['longitude_order'], row['latitude_order']), axis=1)

                # Define a function to map distance to a color gradient
                def get_color(distance):
                    if distance < 10:
                        return [0, 255, 0]  # Green for short distances
                    elif distance < 50:
                        return [255, 255, 0]  # Yellow for medium distances
                    else:
                        return [255, 215, 0]  # Orange for long distances

                # Apply the color function to each row
                arcs_data['color'] = arcs_data['distance_km'].apply(get_color)

                # Calculate the additional columns
                cog_df['num_customers_served'] = arcs_data.groupby('cluster').size().values
                cog_df['avg_distance_km'] = arcs_data.groupby('cluster')['distance_km'].mean().values
                cog_df['avg_transit_days'] = cog_df['avg_distance_km'] / daily_driver_distance
                cog_df['total_kg_served'] = data.groupby('cluster')['weight'].sum().values
                
                # Display the Centers of Gravity DataFrame
                st.subheader("Centers of Gravity: Optimal Results")
                st.dataframe(cog_df)

                # Define the initial view state for the map
                initial_view_state = pdk.ViewState(
                    latitude=data["latitude"].mean(),
                    longitude=data["longitude"].mean(),
                    zoom=4,
                    pitch=50,
                )

                # Define the layer for demand nodes with added opacity setting
                layer_demand_nodes = pdk.Layer(
                    "HexagonLayer",
                    data=data,
                    get_position=['longitude', 'latitude'],
                    radius=10000,  # Radius of the hexagons
                    elevation_scale=50,  # Adjust this value to control the height scaling
                    elevation_range=[0, 3000],  # Range of elevation scaling
                    pickable=True,
                    extruded=True,
                    opacity=0.25,  # Adjust opacity: 0 is fully transparent, 1 is fully opaque
                )

                # Define the layer for the Centers of Gravity
                layer_cog = pdk.Layer(
                    "ScatterplotLayer",
                    data=cog_df,
                    get_position=['longitude', 'latitude'],
                    get_color=[0, 0, 255, 255],
                    get_radius=10000,
                    pickable=True,
                )

                # Define the ArcLayer with gradient colors based on distance
                layer_arcs = pdk.Layer(
                    "ArcLayer",
                    data=arcs_data,
                    get_source_position=['longitude_cog', 'latitude_cog'],
                    get_target_position=['longitude_order', 'latitude_order'],
                    get_source_color=[0, 255, 0, 0],
                    get_target_color='color',
                    pickable=True,
                    auto_highlight=True,
                )

                deck = pdk.Deck(
                    map_style="mapbox://styles/mapbox/satellite-v9",
                    initial_view_state=initial_view_state,
                    layers=[layer_demand_nodes, layer_cog, layer_arcs],
                    api_keys={"mapbox": mapbox_api_token}
                )

                # Render the deck graph
                st.pydeck_chart(deck)



    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.write("Please upload a demand data file to proceed.")
