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

# Session State
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
if "disabled" not in st.session_state:
    st.session_state.disabled = False
if "horizontal" not in st.session_state:
    st.session_state.horizontal = True

# Page Icon, side bar collpase
st.set_page_config(page_title="Form", page_icon="📋", 
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
    


# Score storing and increasing
def score_fr(points):
    
    if "score" not in st.session_state:
        st.session_state.score = 0
    st.session_state.score += points
    st.toast("Score actuel :", st.session_state.get("score", 0))
    
    

st.title("Spectrum App Test : What kind of digital user are you?")
st.header("A short questionnaire to find out in less than 10 minutes!")
st.subheader("Why this test ?")

st.markdown("Everyone surfs the web for entertainment, learning, or work.")
st.markdown("But are we really consuming content that matches our values, desires, and needs?")
st.markdown("Some people seek productivity, others well-being, and still others data security.")
st.markdown("Take this mini test to find out which apps are right for you!")
st.markdown("**Answer a few questions and find out what type of user you are... and what you really need!**")

# Start test button switching to another page
if st.button("Start the test"):
    st.switch_page("pages/0_en_p_one.py")
