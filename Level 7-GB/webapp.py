# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 2025

@author: #gencode 
"""

# PIL
from PIL import Image

# Random
import random

# Pandas
import pandas as pd

# OS
import os

# Streamlit
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

# Streamlit Language Detection
from language_detection import detect_browser_language

# Hazard Mapping Class
from hazard_mapping import HazardMapping, HazardMappingAnimation

# Tsunami Risk Prediction

# Magnitude Estimation Class

# Early Warning Class

# Page Icon, side bar collpase
st.set_page_config(page_title="Tsunami Risk Assessment", 
                   initial_sidebar_state="collapsed")

selected=option_menu(
        menu_title="Menu",
        options = ["Mapping", "Estimation", "Prediction", "Early Warning"], # Page pour relier plusieurs pdf entre eux !
        icons = ["geo-alt", "bar-chart", "cpu", "bell"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
)   

# ------------------------------- EDA of the data --------------------------- #

# Security preventing any reading problem
current_dir = os.path.dirname(__file__)

# Construire le chemin absolu vers le fichier
data_path = os.path.join(current_dir, "data", "earthquake_data_tsunami.csv")

# Lecture sécurisée
try:
    data = pd.read_csv(data_path)
except Exception as e:
    data = pd.read_excel(data_path)
    print(f"{e}")

nb_na = (data.isnull().sum().sum() / data.size) * 100
print("Number of missing data : ", nb_na) # Check NA in the dataset

#                          Removing useless data                              #

data = data.drop(['nst', 'dmin', 'gap'], axis=1)

# ------------------------------- Mapping the data --------------------------- #

import pandas as pd
import plotly.express as px

if selected == "Mapping": # Ajouter un bouton pour mois vs année 

    # Hazard Mapping
    
    def tsunami_mapping():
    
        map_maker = HazardMappingAnimation(data)
        fig_tsu = map_maker.fig_tsunami
        st.plotly_chart(fig_tsu)
        
    def tsunami_mapping_month():
    
        map_maker = HazardMappingAnimation(data)
        fig_tsu_month = map_maker.fig_tsunami_month
        st.plotly_chart(fig_tsu_month)
    
    def earthquake_mapping():
    
        map_maker = HazardMappingAnimation(data)
        fig_ear = map_maker.fig_earthquake
        st.plotly_chart(fig_ear)
        
    def earthquake_mapping_month():
    
        map_maker = HazardMappingAnimation(data)
        fig_ear_month = map_maker.fig_earthquake_month
        st.plotly_chart(fig_ear_month)
    
    def depth_mapping():
    
        map_maker = HazardMappingAnimation(data)
        fig_depth = map_maker.fig_depth
        st.plotly_chart(fig_depth)
        
    def depth_mapping_month():
    
        map_maker = HazardMappingAnimation(data)
        fig_depth_month = map_maker.fig_depth_month
        st.plotly_chart(fig_depth_month)
        
    
    st.title("Tsunami & Earthquake Mapping Animation")
    
    right, left = st.columns(2)
    
    mapping = right.selectbox("Select the animate mapping you want to visualize :",
                 ("Tsunami Mapping", "Earthquake Mapping", "Depth Mapping"),
                 index=0,
                 on_change=None,
                 disabled=False, 
                 label_visibility="visible", 
                 accept_new_options=False)
    
    frame = left.selectbox("Select a time frame :",
                 ("Years", "Months"),
                 index=0)
    
    if mapping == "Tsunami Mapping" and frame == "Years":
        
        st.subheader("Tsunami Mapping Animation Through Years")
        tsunami_mapping()
        
    elif mapping == "Tsunami Mapping" and frame == "Months":
        
        st.subheader("Tsunami Mapping Animation Through Months")
        tsunami_mapping_month()
        
    elif mapping == "Earthquake Mapping" and frame == "Years":
        
        st.subheader("Earthquake Mapping Animation Through Years")
        earthquake_mapping()
        
    elif mapping == "Earthquake Mapping" and frame == "Months":
        
        st.subheader("Earthquake Mapping Animation Through Months")
        earthquake_mapping_month()
        
    elif mapping == "Depth Mapping" and frame == "Years":
        
        st.subheader("Depth Mapping Animation Through Years")
        depth_mapping()
        
    elif mapping == "Depth Mapping" and frame == "Months":
        
        st.subheader("Depth Mapping Animation Through Months")
        depth_mapping_month()

# ---------------------------- Estimation from data ------------------------- #

elif selected == "Estimation":
    
    st.title("🏗 It's under construction, come back in a few days")
    
    # Magnitude in areas on earth -> putting lat+long of all countries
    # so they can choose theirs and see the estimation in % through 
    # years (2001-2020) not prediction
    # Ou alors entrer son année de naissance et voir s'il y avait un risque
    # de tsunami (pour le fun)

# ---------------------------- Prediction from data ------------------------- #
    
elif selected == "Prediction":
    
    st.title("🏗 It's under construction, come back in a few days")
    
    # Ecrire le nom du pays et la date, et même possiblement la magnitude
    # et la depth d'un earthquake pour savoir si un tsunami est possible
    # Et si pas de magnitude ou depth mais simplement date et pays
    # Dire la probabilité des tsunamis, earthquake et leur magnitude + depth
    
    # Faire une colonne avec les pays et au mieux les villes en précis
    # 
    
# ---------------------------- Early Warn from data ------------------------- #

elif selected == "Early Warning": 
    
    st.title("🏗 It's under construction, come back in a few days")
    
    # Utilisation d'une API pour des données en temps réelles
    # Sans que l'utilisateur n'ait à rentrer des données, il sélectionne
    # son pays puis observe !

# en utilisant une API ! 
# Chacun pourrait voir où il se trouve et s'il y a un risque !






















































