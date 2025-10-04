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

# Badges
from streamlit_extras.badges import badge

# Pandas for the df type
import pandas as pd

# Model for Clustering
from sklearn.cluster import KMeans

# False data for clustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # using "blobs" for cluster

# Display the results
import plotly.express as px

# Session State
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
if "disabled" not in st.session_state:
    st.session_state.disabled = False
if "horizontal" not in st.session_state:
    st.session_state.horizontal = True
if "score" not in st.session_state:
    st.session_state.score = 0
if "bar" not in st.session_state:
    st.session_state.bar = 0
if "progress_value" not in st.session_state:
    st.session_state.progress_value = 0

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
    


# Storing the data in JSON


# Clustering all the data


# Result Part
st.title("Spectrum App Test : Results")
st.header("Your results :")

# Visual Results part

# Samples as blobs 
X, y = make_blobs(
    n_samples=1000,
    centers=[[-30, -30], [30, 30], [-30, 30], [30, -30]], # Coordonates
    cluster_std=5,
    random_state=1
)

# KMEANS
model = KMeans(n_clusters=4)

# Training
model.fit(X)

# Prediction
pred = model.predict(X)

# Turning it into a dataframe : scatter can't build graphic without y
df = pd.DataFrame(X, columns=["x", "y"])
df["pred"] = pred

# Figure of the clusters
fig2 = px.scatter(df, x="x", y="y", color=df["pred"].astype(str)) # int -> str

fig2.update_xaxes(
    tickvals=[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
    ticktext=["-50", "-40", "-30", "-20", "-10", "0", "10", "20", "30", "40", "50"]
)

fig2.update_yaxes(
    tickvals=[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
    ticktext=["-50", "-40", "-30", "-20", "-10", "0", "10", "20", "30", "40", "50"]
)

if st.session_state.score < 25:
    
    fig2.add_scatter(x=[-30],
                    y=[30],
                    marker=dict(
                        color='black',
                        size=10
                    ),
                   name='Multitask Mind User Type') 

elif 25 < st.session_state.score < 49:
    
    fig2.add_scatter(x=[30],
                    y=[-30],
                    marker=dict(
                        color='black',
                        size=10
                    ),
                   name='Early Adopters User type')

elif 50 < st.session_state.score < 74:
    
    fig2.add_scatter(x=[30],
                    y=[30],
                    marker=dict(
                        color='black',
                        size=10
                    ),
                   name='Delegator User Type')

elif 75 < st.session_state.score < 100:
    
    fig2.add_scatter(x=[-30],
                    y=[-30],
                    marker=dict(
                        color='black',
                        size=10
                    ),
                   name='Data Sovereign User Type')
    
st.plotly_chart(fig2)

# Results Written Part


st.subheader(f"Your score is {st.session_state.score} %")

if st.session_state.score < 25:
    
    st.subheader("Multitask Mind User Type:")
    
    st.markdown("As a multitasker, you tend to spread yourself thin.")
    st.markdown("You are also very likely to have multiple passions.")
    st.markdown("We therefore have a series of websites/applications to offer you:")
    
    st.markdown("⚫ https://mindscout.net/")
    st.markdown("⚫ https://mindscout.net/") # replacement of Notion
    st.markdown("⚫ https://www.notion.com/")
    st.markdown("⚫ https://obsidian.md/")
    

elif 25 < st.session_state.score < 49:
    
    st.subheader("Early Adopters User Type :")
    
    st.markdown("As an early adopter, you enjoy testing new features before anyone else and sharing them.")
    st.markdown("You are also what we call a beta tester, testing even prototypes before the finished model.")
    st.markdown("We therefore have a series of sites/applications to recommend to you:")
    
    st.markdown("⚫ https://www.producthunt.com/")
    st.markdown("⚫ https://betalist.com/")
    st.markdown("⚫ https://www.indiehackers.com/")
    st.markdown("⚫ https://lookerstudio.google.com/")
    

elif 50 < st.session_state.score < 74:
    
    st.markdown("Delegator User Type :")
    
    st.markdown("As a delegator, you prefer to assign tasks rather than do them yourself.")
    st.markdown("You prefer to have tailor-made solutions, pay for professionals, and not have to learn new skills that would waste your time.")
    st.markdown("We therefore have this series of websites/applications to present to your profile type:")
    
    st.markdown("⚫ https://freelancer.com/")
    st.markdown("⚫ https://www.upwork.com/")

elif 75 < st.session_state.score < 100:
    
    st.markdown("Data Sovereign User Type :")
    
    st.markdown("As a data sovereign, you value autonomy and simplicity, and you want to control your data.")
    st.markdown("You are open to the world of technology but wary of how online services use your sensitive data.")
    st.markdown("We therefore have a series of websites/applications for your category to discover:")
    
    st.markdown("⚫ https://live-report-generator.streamlit.app/")
    st.markdown("⚫ https://nextcloud.com/")
    st.markdown("⚫ https://plausible.io/")


# Notification for the user
def notif_score():
    msg = st.toast(f"Your results !")
 

if st.button("Return to the Main Page"):
    st.session_state.score = 0
    st.switch_page("pages/0_en_intro.py")