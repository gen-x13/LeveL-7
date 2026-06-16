# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 2025

@author: #genxcode 
"""

# PIL
from PIL import Image

# Random
import random

# Pandas
import pandas as pd

# Path For Data
from pathlib import Path

# OS
import os

# Plotly
import plotly.express as px

# Streamlit
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
# Hazard Mapping Class
from hazard_mapping_globe import HazardMapping, HazardMappingAnimation

# Tsunami Risk Prediction, Early Warning
from tsunami_risk_pred_ew import TsunamiRisk, TsunamiRiskPred, TsunamiRiskEW

# Page Icon, side bar collpase
st.set_page_config(page_title="Tsunami Risk Assessment", 
                   initial_sidebar_state="collapsed")

selected = option_menu(
    menu_title=None,
    options=["Mapping", "Estimation", "Prediction", "Early Warning"],
    icons=["geo-alt", "bar-chart", "cpu", "bell"],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0.3rem",
            "background-color": "transparent",
            "backdrop-filter": "blur(6px)",
            "border-radius": "14px",
        },
        "nav-link": {
            "font-size": "15px",
            "text-align": "center",
            "color": "#dbeafe",
            "border-radius": "10px",
            "margin": "0 6px",
            "--hover-color": "rgba(30, 58, 138, 0.6)",
            "transition": "all 0.2s ease",
        },
        "nav-link-selected": {
            "background-color": "#38bdf8",
            "color": "#020617",
            "font-weight": "600",
            "border-radius": "10px",
        },
    },
)


# ----------------------- Background ------------------------------------ #

# CSS Background
css_path = Path(__file__).parent / "files" / "css.css"
with open(css_path) as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# --------------------- Background Colors -------------------------------- #

# Background Colors for plots
brig_blue = "rgb(0, 150, 255)"
coba_blue = "rgb(0, 71, 171)"
egyp_blue = "rgb(20, 52, 164)"
neon_blue = "rgb(31, 81, 255)"

# List of the colors
colors = [brig_blue, coba_blue, egyp_blue, neon_blue]

# Randomization of the color background
backg_color = random.choice(colors)

# ------------------------------- Loading data --------------------------- #

# Security preventing any reading problem and any cache data problem
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return pd.read_csv(
        Path(__file__).parent / "data" / "earthquake_data_tsunami.csv",
        dtype={
            "latitude": "float32",
            "longitude": "float32",
            "depth": "float32",
            "magnitude": "float32",
            "Year": "int16",
            "Month": "int8",
            "country": "category",
            "tsunami": "int8",
        }
    )

data = load_data()
data = data.drop(['nst', 'dmin', 'gap', 'cdi', 'mmi', 'sig'], axis=1)

# ---------------------------------- Cache ---------------------------------- #

# For every object : st.cache_resource -> avoid problems with cache later

# Hazard Mapping
@st.cache_resource # only cache for objects
def get_mapping(data):
    return HazardMappingAnimation(data)

# Tsunami Risk 
@st.cache_resource # same cache technic
def get_tsunami_risk(data):
    return TsunamiRisk(data)

# ------------------------------- Mapping the data --------------------------- #

import pandas as pd
import plotly.express as px

# Mapping Page Selection :
if selected == "Mapping": 
    
    st.title("Tsunami & Earthquake Mapping Animation")
    
    right, left = st.columns(2)
    
    # Choice of Mapping :
    mapping = right.selectbox("Select the animate mapping you want to visualize :",
                 ("Tsunami Mapping", "Earthquake Mapping", "Depth Mapping"),
                 index=0,
                 on_change=None,
                 disabled=False, 
                 label_visibility="visible", 
                 accept_new_options=False)
    
    # Seleciton of Time Frame for Animation :   
    frame = left.selectbox("Select a time frame :",
                 ("Years", "Months"),
                 index=0)
    
    # Mapping Choices :
    if mapping == "Tsunami Mapping" and frame == "Years":
        
        # Figure's Title
        st.subheader("Tsunami Mapping Animation Through Years")
        
        # Figure
        map_maker = get_mapping(data)
        st.plotly_chart(map_maker.fig_tsunami)
        
    elif mapping == "Tsunami Mapping" and frame == "Months":
        
        # Figure's title
        st.subheader("Tsunami Mapping Animation Through Months")
        
        # Figure 
        map_maker = get_mapping(data)
        st.plotly_chart(map_maker.fig_tsunami_month)
        
    elif mapping == "Earthquake Mapping" and frame == "Years":
        
        # Figure's title
        st.subheader("Earthquake Mapping Animation Through Years")
        
        # Figure
        map_maker = get_mapping(data)
        st.plotly_chart(map_maker.fig_earthquake)
        
    elif mapping == "Earthquake Mapping" and frame == "Months":
        
        # Figure's Title
        st.subheader("Earthquake Mapping Animation Through Months")
        
        # Figure
        map_maker = get_mapping(data)
        st.plotly_chart(map_maker.fig_earthquake_month)
        
    elif mapping == "Depth Mapping" and frame == "Years":
        
        # Figure's Title
        st.subheader("Depth Mapping Animation Through Years")
        
        # Figure
        map_maker = get_mapping(data)
        st.plotly_chart(map_maker.fig_depth)
        
    elif mapping == "Depth Mapping" and frame == "Months":
        
        # Figure's Title
        st.subheader("Depth Mapping Animation Through Months")
        
        # Figure
        map_maker = get_mapping(data)
        st.plotly_chart(map_maker.fig_depth_month)

# ---------------------------- Estimation from data ------------------------- #

elif selected == "Estimation":
    
    # Magnitude in areas on earth -> putting lat+long of all countries
    # so they can choose theirs and see the estimation in % through 
    # years (2001-2020) not prediction
    # Magnitude Estimation and Tsunami Estimation from the 2001-2020
    
    st.title("Tsunamis & Earthquakes Estimation between 2001-2020")
    st.markdown("Select a country and it'll give you the percentage of tsunamis and earthquakes that had already occured in this area!")
    
    with st.spinner("Initialization..."):
        tsunami_risk = get_tsunami_risk(data)
        countries_list = tsunami_risk.df2

    # List for choice
    c_list = countries_list['country'].sort_values()
  
    # Choice
    country_name = st.selectbox(
        "Which country ?",
        options= c_list.unique(),
        index=0
    )

    # Figure 
    tsunami_risk.tsunami_estimation_graph(country=country_name)

    #df_country = countries_list[countries_list["country"] == country_name]

    import pycountry

    df_map = (countries_list.groupby("country", as_index=False).agg({"tsunami": "sum"}))

    df_map["iso3"] = df_map["country"].apply(
        lambda x: pycountry.countries.get(alpha_2=x).alpha_3
    )
  
    fig1 = px.choropleth(
        df_map,
        locations="iso3",
        color="tsunami",
        locationmode="ISO-3"
    )

    fig2 = px.choropleth(df_map, locations="iso3",
                        color=backg_color,
                        hover_name="magnitude",
                        hover_data=["tsunami"],
                        locationmode="ISO-3",
                        animation_frame='Years',
                        color_continuous_midpoint = 3,
    color_continuous_scale=px.colors.sequential.thermal_r)
    fig.update_layout(margin=dict(l=20,r=0,b=0,t=70,pad=0),paper_bgcolor="white",height= 700,title_text = f"Earthquake & Tsunami's Risk in {country_name}",font_size=18)

    st.plotly_chart(fig1, use_container_width=False)
    st.plotly_chart(fig2, use_container_width=False)


# ---------------------------- Prediction from data ------------------------- #
    
elif selected == "Prediction":
    
    st.title("Tsunamis Prediction")
    st.markdown("Select a country, a year, a month, a magnitude for the earthquake and its depth.")
    st.markdown("Then, you will see if a tsunami would occure or not with your selection.")
    
    with st.spinner("Initialization..."):
        tsunami_risk = get_tsunami_risk(data)
        df = tsunami_risk.df2
    
    one, two, three, four, five = st.columns(5)
    
    city = df['country'].sort_values()
    city_select = one.selectbox(
        "Which country ?",
        options= city.unique(),
        placeholder="No selection...",
        index=None
    )
    
    year = df['Year'].sort_values()
    year_select = two.selectbox(
        "Which year ?",
        options= year.unique(),
        placeholder="No selection...",
        index=None
    )
    
    month = df['Month'].sort_values()
    month_select = three.selectbox(
        "Which month ?",
        options= month.unique(),
        placeholder="No selection...",
        index=None
    )
    
    depth = df['depth'].sort_values()
    depth_select = four.selectbox(
        "Which depth ?",
        options= depth.unique(),
        placeholder="No selection...",
        index=None
    )
    
    magnitude = df['magnitude'].sort_values()
    magnitude_select = five.selectbox(
        "Which magnitude ?",
        options= magnitude.unique(),
        placeholder="No selection...",
        index=None
    )
    
    if (
        city_select is not None
        and year_select is not None
        and month_select is not None
        and depth_select is not None
        and magnitude_select is not None
    ):
        
        if st.button("Predict tsunami risk"):
            with st.spinner("Prediction..."):
                model = TsunamiRiskPred(
                    city_select,
                    year_select,
                    month_select,
                    depth_select,
                    magnitude_select,
                    data
                )
                
            tsunami_pred = model.result
    
            if tsunami_pred == 1:
                st.success("There's a chance that a tsunami occurs based on the selected parameters.")
            else:
                st.info("There's no chance that a tsunami occurs based on the selected parameters.")
        
    else:
        st.warning("Select all of your parameters for the prediction.")
        
    
    st.caption("🏗 It's still under construction, come back in a few days")
    
    
    
# ---------------------------- Early Warn from data ------------------------- #

elif selected == "Early Warning": 
    
    # Utilisation d'une API pour des données en temps réelles
    # Sans que l'utilisateur n'ait à rentrer des données, il sélectionne
    # son pays puis observe !

    # en utilisant une API ! 
    # Chacun pourrait voir où il se trouve et s'il y a un risque !
    
    # API used : USGS GEOJSON

    # Init threshold
    threshold_green = 0.25
    threshold_yellow = 0.50
    threshold_orange = 0.75
    threshold_red =  1.00
    
    # Title 
    st.title("Tsunami Alert") 
    st.markdown("Select a latitude and a longitude. "
                "The system analyzes seismic activity within a 2000 km radius (last 24h).") 
    
    # Link to help the user find his latitude and longitude
    st.markdown(f":color[If you need help to find your coordinates, this site will help you : https://latlongdata.com/]{{foreground='black'}}")
    
    # Columns to display the features side by side
    one, two = st.columns(2)
    
    # Charging the data with elegance
    with st.spinner("Initialization..."):
        tsunami_risk = get_tsunami_risk(data)
        df = tsunami_risk.df2
      
    # Choice of latitude and longitude for the user in a text entry
    lat_select = one.text_input("Write a latitude...")
    lon_select = two.text_input("Write a longitude...")
    
    # Condition to start the analysis and real-time prediction
    if lat_select is not None and lon_select is not None:
        
        if st.button("Run Tsunami Early Warning"):
            
            with st.spinner("Analyzing seismic activity..."):
                model = TsunamiRiskEW(float(lat_select), float(lon_select))
            
            # Display the results by comparison and threshold
            tsunami_pred = model.result
            
            # Display thatb there's no earthquake instead of raising an error
            no_earthquake = model.no_earthquake
            
            # If the variable is true, that means there's no data provided 
            # within 24 hours data from the USGS API.
            if no_earthquake == True:
                st.info("No earthquake nor data for this coordinates within 24 hours.")
            
            # Traffic light for every state of the probability to have a tsunami
            elif tsunami_pred == 0:
                st.markdown(":color[🟢 No significant seismic activity detected.]{{background='rgb(9, 121, 105)' foreground='black'}}")
            elif tsunami_pred < threshold_green:
                st.markdown(f":color[🟢 Green Alert  : Low risk of tsunami. Percentage : {tsunami_pred*100:.1f} %]{{background='rgb(9, 121, 105)' foreground='black'}}")
            elif tsunami_pred < threshold_yellow:
                st.markdown(f":color[🟡 Yellow Alert : Moderate risk of tsunami. Percentage : {tsunami_pred*100:.1f} %]{{background='rgb(223, 255, 0)' foreground='black'}}")
            elif tsunami_pred < threshold_orange:
                st.markdown(f":color[🟠 Orange Alert : High risk of tsunami. Percentage : {tsunami_pred*100:.1f} %]{{background='rgb(255, 170, 0)' foreground='black'}}")
            else :
                st.markdown(f":color[🔴 Red Alert : Critical risk of tsunami. Percentage : {tsunami_pred*100:.1f} %]{{background='rgb(255, 0, 0)' foreground='black'}}")

            cols = st.columns(2)
          
            for index, place in enumerate(model.places):
              
                col = cols[index % 2] # rotation 0,1
              
                result_title = f"Result {index+1}"
                results = f"""
                           ***{result_title}***
                           
                           **Place selected** : {place}
                           
                           **Magnitude** : {model.mags[index]}
                           
                           **Depth** : {model.depths[index]}
                           
                           **Hour** : {model.times[index]}
                          """
                with col:
                  st.markdown(results, text_alignment="center")

      
    else:
        st.warning("Select all of your parameters for the prediction.")
        
    
    st.caption("🏗 It's still under construction, come back in a few days")

































































