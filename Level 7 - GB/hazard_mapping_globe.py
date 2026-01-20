# Imports for the project
import pathlib as Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------------------------------------------------- #
#              Security preventing any reading problem                        #

try:
    data = pd.read_csv(Path(__file__).parent / "data" / "earthquake_data_tsunami.csv")
except Exception as e:
    print(f"{e}")
    
    
# --------------------------------------------------------------------------- #
#                          Removing possible NaN                              #

nb_na = (data.isnull().sum().sum() / data.size) * 100

#                          Removing useless data                              #

data = data.drop(['nst', 'dmin', 'gap'], axis=1)

# Creating another column combining year and month
# data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))
data["YearMonth"] = data["Year"].astype(str) + "-" + data["Month"].astype(str).str.zfill(2)
data = data.sort_values(by=["Year", "Month"], ascending=True)
data = data.sort_values(by=["Year"], ascending=True)


# --------------------------------------------------------------------------- #
#                         Creation of the 3D Globe                            #

# Class for the animation :  
class HazardMapping():
    
    def __init__(self):
        
        self.fig_tsunami = self.create_globe_tsunami(data, 'tsunami') 
        self.fig_earthquake = self.create_globe_earthquake(data, 'magnitude') 
        self.fig_depth = self.create_globe_depth(data, 'depth')
        
        self.fig_tsunami_month = self.create_globe_tsunami_month(data, 'tsunami') 
        self.fig_earthquake_month = self.create_globe_earthquake_month(data, 'magnitude') 
        self.fig_depth_month = self.create_globe_depth_month(data, 'depth')
        
    def create_globe_tsunami(self, data, value_column='Tsunami'):
        
        fig1 = px.scatter_geo(
            data,
            lon="longitude",
            lat="latitude",
            color="tsunami",
            color_continuous_scale="Blues",
            projection="orthographic",
            size="magnitude",
            animation_frame='Year'
        )
        
        fig1.update_geos(
            showcountries=True,
            showcoastlines=True, coastlinecolor="rgba(255, 255, 255, 0.8)",
            showland=True, landcolor="#3D3D3D",
            showocean=True, oceancolor="#1e1e1e",
            projection_type="orthographic",
            bgcolor="#111111"
        )
        
        fig1.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            #scene=dict(dragmode='orbit'),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font=dict(color='white', family="Arial", size=14),
            title=dict(
                font=dict(color="white", size=24),
                x=1,
                y=0
            ),
        coloraxis_colorbar=dict(
                title=value_column,
                title_side="right",  # NOTE: use title_side (underscore) if needed
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                xanchor="left"
            )
        )
    
        return fig1
    
    def create_globe_tsunami_month(self, data, value_column='Tsunami'):
        
        fig1 = px.scatter_geo(
            data,
            lon="longitude",
            lat="latitude",
            color="tsunami",
            color_continuous_scale="Blues",
            projection="orthographic",
            size="magnitude",
            animation_frame='YearMonth'
        )
        
        fig1.update_geos(
            showcountries=True,
            showcoastlines=True, coastlinecolor="rgba(255, 255, 255, 0.8)",
            showland=True, landcolor="#3D3D3D",
            showocean=True, oceancolor="#1e1e1e",
            projection_type="orthographic",
            bgcolor="#111111"
        )
        
        fig1.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            #scene=dict(dragmode='orbit'),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font=dict(color='white', family="Arial", size=14),
            title=dict(
                font=dict(color="white", size=24),
                x=1,
                y=0
            ),
        coloraxis_colorbar=dict(
                title=value_column,
                title_side="right",  # NOTE: use title_side (underscore) if needed
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                xanchor="left"
            )
        )
    
        return fig1
    
    def create_globe_earthquake(self, data, value_column='magnitude'):
        
        fig2 = px.scatter_geo(
            data,
            lon="longitude",
            lat="latitude",
            color="magnitude",
            color_continuous_scale="Oranges",
            projection="orthographic",
            size="magnitude",
            animation_frame='Year'
        )
        
        fig2.update_geos(
            showcountries=True,
            showcoastlines=True, coastlinecolor="rgba(255, 255, 255, 0.8)",
            showland=True, landcolor="#3D3D3D",
            showocean=True, oceancolor="#1e1e1e",
            projection_type="orthographic",
            bgcolor="#111111"
        )
        
        fig2.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            #scene=dict(dragmode='orbit'),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font=dict(color='white', family="Arial", size=14),
            title=dict(
                font=dict(color="white", size=24),
                x=1,
                y=0.9 # 1 à 10
            ),
        coloraxis_colorbar=dict(
                title=value_column,
                title_side="right",  # NOTE: use title_side (underscore) if needed
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                xanchor="left"
            )
        )
    
        return fig2
    
    def create_globe_earthquake_month(self, data, value_column='magnitude'):
        
        fig2 = px.scatter_geo(
            data,
            lon="longitude",
            lat="latitude",
            color="magnitude",
            color_continuous_scale="Oranges",
            projection="orthographic",
            size="magnitude",
            animation_frame='YearMonth'
        )
        
        fig2.update_geos(
            showcountries=True,
            showcoastlines=True, coastlinecolor="rgba(255, 255, 255, 0.8)",
            showland=True, landcolor="#3D3D3D",
            showocean=True, oceancolor="#1e1e1e",
            projection_type="orthographic",
            bgcolor="#111111"
        )
        
        fig2.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            #scene=dict(dragmode='orbit'),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font=dict(color='white', family="Arial", size=14),
            title=dict(
                font=dict(color="white", size=24),
                x=1,
                y=0.9 # 1 à 10
            ),
        coloraxis_colorbar=dict(
                title=value_column,
                title_side="right",  # NOTE: use title_side (underscore) if needed
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                xanchor="left"
            )
        )
    
        return fig2
    
    def create_globe_depth(self, data, value_column='depth'):
        
        fig3 = px.scatter_geo(
            data,
            lon="longitude",
            lat="latitude",
            color="depth",
            color_continuous_scale="Reds",
            projection="orthographic",
            size="depth",
            animation_frame='Year'
        )
        
        fig3.update_geos(
            showcountries=True,
            showcoastlines=True, coastlinecolor="rgba(255, 255, 255, 0.8)",
            showland=True, landcolor="#3D3D3D",
            showocean=True, oceancolor="#1e1e1e",
            projection_type="orthographic",
            bgcolor="#111111"
        )
        
        fig3.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            #scene=dict(dragmode='orbit'),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font=dict(color='white', family="Arial", size=14),
            title=dict(
                font=dict(color="white", size=24),
                x=1,
                y=0.9 # 2 à 671
            ),
        coloraxis_colorbar=dict(
                title=value_column,
                title_side="right",  # NOTE: use title_side (underscore) if needed
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                xanchor="left"
            )
        )
    
        return fig3
    
    def create_globe_depth_month(self, data, value_column='depth'):
        
        fig3 = px.scatter_geo(
            data,
            lon="longitude",
            lat="latitude",
            color="depth",
            color_continuous_scale="Reds",
            projection="orthographic",
            size="depth",
            animation_frame='YearMonth'
        )
        
        fig3.update_geos(
            showcountries=True,
            showcoastlines=True, coastlinecolor="rgba(255, 255, 255, 0.8)",
            showland=True, landcolor="#3D3D3D",
            showocean=True, oceancolor="#1e1e1e",
            projection_type="orthographic",
            bgcolor="#111111"
        )
        
        fig3.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            scene=dict(dragmode='orbit'),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font=dict(color='white', family="Arial", size=14),
            title=dict(
                font=dict(color="white", size=24),
                x=1,
                y=0.9 # 2 à 671
            ),
        coloraxis_colorbar=dict(
                title=value_column,
                title_side="right",  # NOTE: use title_side (underscore) if needed
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                xanchor="left"
            )
        )
    
        return fig3

# --------------------------------------------------------------------------- #
#                         Animation of the 3D Globe                           #

class HazardMappingAnimation():
    
    def __init__(self, data):
        
        self.map = HazardMapping()
        self.fig_tsunami = self.map.fig_tsunami
        self.fig_tsunami_month = self.map.fig_tsunami_month
        self.fig_earthquake = self.map.fig_earthquake
        self.fig_earthquake_month = self.map.fig_earthquake_month
        self.fig_depth = self.map.fig_depth
        self.fig_depth_month = self.map.fig_depth_month
        
        fig_liste = self.fig_tsunami, self.fig_tsunami_month, self.fig_earthquake, self.fig_earthquake_month, self.fig_depth, self.fig_depth_month
        
        for fig in fig_liste:

            if fig.layout.updatemenus and fig.layout.updatemenus[0].buttons:
                fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
                fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 600
            
            fig.layout.coloraxis.showscale = True 
            fig.layout.sliders[0].pad.t = 10
            fig.layout.updatemenus[0].pad.t= 10   


