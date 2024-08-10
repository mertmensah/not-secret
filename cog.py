import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pydeck as pdk
from math import radians, sin, cos, sqrt, atan2

# Initialize an empty list for verified zips
verified_zips = []

# LOCALbring in ZIP Codes from external source, to match coordinates to orders
zips = pd.read_csv('data/zip_codes_us.csv')

# Ensure 'state' column is present
if 'state' not in zips.columns:
    st.error("The ZIP codes dataset must contain a 'state' column.")
else:
    # Cleaning
    zips['zip'] = zips['zip'].astype(str)  # Convert type
    zips['zip'] = zips['zip'].astype(str).str.zfill(5)  # Add leading 0 until there are 5 characters 

    # SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
    st.set_page_config(layout="wide", page_title="opti-Network | optiSC", page_icon="images/optsci-logo-onlycircles.svg")
    st.title('opti-Network')

    # Your Mapbox API token
    mapbox_api_token = "pk.eyJ1IjoibWVydG1lbnNhaCIsImEiOiJjbHlhZWJ1bW4xMmZxMmpwdWtiN3VqZTVoIn0.cJSoRI6C7zJQjwKALaki2w"  # Replace with your actual Mapbox API token

    # Placeholder for the state options in the sidebar
    state_options = []

    # SIDEBAR
    with st.sidebar:
        st.image("C:/Users/MertMM/Desktop/vsc-test/images/optisc-logo-black.svg", width=250)
        st.title("Parameters")
        num_cogs = st.slider('**1. Number of Centers of Gravity**', min_value=1, max_value=10, value=3)
        st.write("**Note:** Additional nodes will impact solution time.")
        ""
        ""
        daily_driver_distance = st.slider('**2. Daily Driver Distance Coverage (km)**', 0, 1500, 800)
        st.write("800 km is widely used as a long-haul ground daily coverage estimate")

    # Body Text
    st.write('Analyze your demand or supply flows to discover your optimal network. The locations provided by this tools will be those which minimize the total weighed distance amongst all of the locations in your network.')

    # Create a container for the specific section
    with st.expander("**Instructions:** Structuring your data"):
        st.write('For further instructions, please refer to the opti-Network product page.')

        # Create two columns within this expander
        col1, col2 = st.columns([1, 4])  # Adjust the ratio as needed

        # Add the image to the first column
        with col1:
            st.image("images/demand_sample_screen_tut.png", width=120)

        # Add the text to the second column
        with col2:
            st.write("To ensure compatibility, structure your data in two columns as seen in the image:")
            st.write("1. **Zip Code** of your demand location")
            st.write("2. **Total volume** associated with that location")

    # Sample file download
    with st.expander("**Try the tool with sample data**"):
        st.write("You can download a sample CSV file to try the tool.")

        # Read the file content for download
        with open('data/demand_sample.csv', 'rb') as file:
            st.download_button(
                label="Download Sample CSV",
                data=file,
                file_name="demand_sample.csv",
                mime="text/csv"
            )
    ""
    ""

    st.subheader("Center of Gravity Solver")
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

                # Filter the demand data for verified ZIP codes
                df_verified = df_open_orders[df_open_orders['zip_code'].isin(verified_zips)]

                if not df_verified.empty:
                    # Merge data with ZIP coordinates and state
                    data = pd.merge(df_verified, zips[['zip', 'state', 'longitude', 'latitude']], left_on='zip_code', right_on='zip', how='left')

                    # Update verified_zips with the state information
                    verified_zips = sorted(data['state'].unique().tolist())  # Sort alphabetically descending

                    # Scenario Exploration Section (Collapsible) in Sidebar
                    with st.sidebar:
                        ""
                        st.title("What-if Analysis")
                        with st.expander("Scenario: Demand Fluctuation"):
                            st.markdown("### Adjust Demand for Selected States")
                            
                            shock_factors = {}
                            for i in range(1, 6):  # Up to 5 states
                                state = st.selectbox(f"State {i}", options=verified_zips, key=f'state_{i}')  # Unique key for each dropdown
                                factor = st.number_input(f"Shock Factor for {state}", min_value=0.0, value=1.0, key=f'shock_factor_{i}')  # Unique key for each input
                                if state:
                                    shock_factors[state] = factor

                        # Existing facilities input from the user
                        with st.expander("Scenario: Existing Facilities"):
                            st.markdown("### Coming Soon")

                    # Apply demand shock factors
                    for state, factor in shock_factors.items():
                        data.loc[data['state'] == state, 'weight'] *= factor

                    # Initialize K-Means with weighted clustering centers
                    coordinates = data[['longitude', 'latitude']].values
                    weights = data['weight'].values

                    kmeans = KMeans(n_clusters=num_cogs, random_state=42)
                    kmeans.fit(coordinates, sample_weight=weights)

                    # Separate the cluster labels for the demand data and existing facilities
                    data['cluster'] = kmeans.labels_[:len(data)]  # Assign clusters to original data only

                    # Calculate weighted centers of gravity for each cluster
                    weighted_cog_data = []
                    for cluster in range(num_cogs):
                        cluster_data = coordinates[kmeans.labels_ == cluster]
                        weighted_lon = np.average(cluster_data[:, 0], weights=weights[kmeans.labels_ == cluster])
                        weighted_lat = np.average(cluster_data[:, 1], weights=weights[kmeans.labels_ == cluster])
                        weighted_cog_data.append({'longitude': weighted_lon, 'latitude': weighted_lat, 'cluster': cluster})

                    cog_df = pd.DataFrame(weighted_cog_data)

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
                    
                    # Rename the columns
                    cog_df.rename(columns={
                        'longitude': 'CoG Longitude',
                        'latitude': 'CoG Latitude',
                        'cog_city': 'CoG City',
                        'cog_state': 'CoG State',
                        'num_customers_served': 'Customer Locations Served',
                        'avg_distance_km': 'AVG OB Distance (km)',
                        'avg_transit_days': 'AVG OB Transit Days',
                        'total_kg_served': 'Total Demand Served (kg)'
                    }, inplace=True)

                    # Round the CoG coordinates to two decimal places
                    cog_df['CoG Longitude'] = cog_df['CoG Longitude'].round(3)
                    cog_df['CoG Latitude'] = cog_df['CoG Latitude'].round(3)

                    # Remove the cluster column
                    cog_df = cog_df.drop(columns=['cluster'])

                    # Display the Centers of Gravity DataFrame
                    ""
                    ""
                    st.subheader("Solution Results")
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
                        layers=[layer_demand_nodes, layer_arcs],
                        api_keys={"mapbox": mapbox_api_token}
                    )

                    # Render the deck graph
                    st.pydeck_chart(deck)

                "**Scenario Data**"

                # Create two columns side by side
                col1, col2 = st.columns(2)

                # Show uploaded data in the first column
                with col1:
                    with st.expander("Matched Demand Data"):
                        st.write(df_open_orders)

                # Show unmatched ZIP codes in the second column, if any
                if unmatched_zips:
                    with col2:
                        with st.expander("Unmatched ZIP Codes"):
                            st.write(unmatched_zips)


        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.write("Upload your demand data file above to proceed.")
