import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time


# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="Center of Gravity Solver", page_icon=":articulated_lorry:")
#https://www.webfx.com/tools/emoji-cheat-sheet/

st.title('Center of Gravity Solver')
'Upload your demand data to discover your optimal distribution network.'

### SIDE BAR ###
# Centers of Gravities desired for calculation
with st.sidebar.container():
    st.header("Centers of Gravity")
    cogs_to_solve = st.slider(
        'Select the desired number of CoGs to solve for:',
        0, 5, 1
    )
st.sidebar.write("**Note:** Additional nodes will impact solution time.")

# Existing Facilities
# Initialize session state for storing ZIP codes
if 'existing_facilities' not in st.session_state:
    st.session_state.existing_facilities = []

# Form to add ZIP code and clear facilities
with st.sidebar.form(key='add_zip_form', clear_on_submit=True):
    add_textbox = st.text_input("**Enter Current Facility ZIP Codes**")
    col1, col2 = st.columns([1, 1])
    with col1:
        submit_button = st.form_submit_button(label='Add ZIP')
    with col2:
        clear_button = st.form_submit_button(label='Clear All')

# Handle form submission
if submit_button:
    if add_textbox:
        st.session_state.existing_facilities.append(add_textbox)

# Handle clear facilities button
if clear_button:
    st.session_state.existing_facilities = []




# Slider for Daily Driver Distance Coverage
with st.sidebar.container():
    st.header("Daily Driver Distance Coverage")
    add_slider = st.slider(
        '800 km is widely used as a long-haul ground daily coverage estimates',
        0, 1500, 800
    )

#sample selectbox - maybe use to change map layout map_style="mapbox://styles/mapbox/satellite-v9", #https://docs.mapbox.com/api/maps/styles/
add_selectbox = st.sidebar.selectbox(
    'TEST: selectbox',
    ('Email', 'Home phone', 'Mobile phone')
)


### MAIN PAGE###
# Demand file upload button
st.header("Upload Demand Data")
uploaded_file = st.file_uploader("Add Demand Data", type=["csv", "xlsx"])

# Process the uploaded file
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_demand = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df_demand = pd.read_excel(uploaded_file)
        st.write("Uploaded Demand Data")
        st.write(df_demand)
    except Exception as e:
        st.error(f"Error reading file: {e}")
'If you experience errors regarding file size, consider aggregating by ZIP Code.'


# Convert list to DataFrame and display
df_existing_facilities = pd.DataFrame(st.session_state.existing_facilities, columns=["Facility ZIP"])
st.write('#### Current Facilities')
st.write(df_existing_facilities)

# SOLVER PROGRESS BAR:
st.write('#### Optimization Progress')
'Calculating optimal network...'
latest_iteration = st.empty() # Add a placeholder 
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Progress: {i+1} %')
  bar.progress(i + 1)
  time.sleep(0.1)
'...solved'


# store models and non-ser objs with: st.cache_resource
# store data and computations that return data with: st.cache_data






