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
    
    

st.title("Spectrum App Test : Quel utilisateur numérique êtes-vous ?")
st.header("Un mini questionnaire pour le découvrir en moins de 10 minutes !")
st.subheader("Pourquoi ce test ?")

st.markdown("Tout le monde navigue sur la toile, pour se divertir, apprendre ou travailler.")
st.markdown("Mais consommons-nous réellement ce qui correspond à nos valeurs, nos envies, nos besoins ?")
st.markdown("Certains recherchent la productivité, d'autres le bien-être ou encore la sécurité de leurs données.")
st.markdown("Avec ce mini test, faites le point et découvrez des applications en phase avec votre profil !")
st.markdown("**Répondez à quelques questions et découvrez quel type d'utilisateur vous êtes... et ce dont vous avez réellement besoin !**")

# Start test button switching to another page
if st.button("Commencer le test"):
    st.switch_page("pages/1_fr_p_one.py")
