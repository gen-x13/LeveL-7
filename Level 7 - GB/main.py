# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 17:28:45 2025

@author: @genxcode - Form with Cluster
"""

# Streamlit
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

# Streamlit Language Detection
from language_detection import detect_browser_language


# PIL
from PIL import Image

# Random
import random

# Page Icon, side bar collpase
st.set_page_config(page_title="Spectrum App", page_icon="📋", 
                   initial_sidebar_state="collapsed")

# Disable sidebar
st.markdown("""
            <style>
            [data-testid="stSidebar"] {
                display: none
            }

            [data-testid="collapsedControl"] {
                display: none
            }
            </style>
            """, unsafe_allow_html=True)
    
# Detection of browser language
browser_language = detect_browser_language()

if browser_language.startswith('fr'):
    st.switch_page("pages/1_fr_intro.py")
else:
    st.switch_page("pages/0_en_intro.py")

#st.write(f"👥 You are the visitor #{len(st.session_state.get('visitors', []))+1}")