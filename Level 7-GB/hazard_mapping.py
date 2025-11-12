# Imports for the project
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# Security preventing any reading problem
try:
    data = pd.read_csv("data/earthquake_data_tsunami.csv")
except Exception as e:
    data = pd.read_excel("data/earthquake_data_tsunami.csv")
    print(f"{e}")

# Visualize the dataset
print(data.head())

# ------------------------------- EDA of the data --------------------------- #

# Data :
    # Mondial EarthQuake Data (without the extrems pole such as the deep Antarctic)
    # 782 earthquakes registered between 2001 and 2022
    # No missing values (still checking just in case)
    # Target : Tsunami potential indicator (binary classification)
        # Non-Tsunami event : 478 records (61.1%)
        # Tsunami event     : 304 records (38.9%)
        # (Possible shape problem later?)
    # Range                   : 6.5 - 9.1 Richter scale
    # Mean Magnitude          : 6.94
    # Major Earthquakes (≥8.0): 28 events including the 2004 (9.1) and 2011 (9.1) mega-earthquakes

# 4 Quests to achieve :
    # Tsunami Risk Prediction: Binary classification using seismic parameters
    # Early Warning Systems: Real-time tsunami threat assessment
    # Hazard Mapping: Geographic risk zone identification # plotly map ou st.map
    # Magnitude Estimation: Earthquake strength prediction from network data


#            Selecting only the relevant columns for each cases               #   
    
# Relevant columns for Tsunami risk prediction :
    # magnitude : Magnitude of earthquakes
    # depth : Earthquake local depth
    # latitude : epicenter latitude
    # longitude : epicenter longitude
    # year : Year of occurrence
    # month : Month of occurrence (in streamlit)

# Relevant columns for Early Warning Systems :
    # magnitude : Magnitude of earthquakes
    # depth : Earthquake local depth
    # latitude : epicenter latitude
    # longitude : epicenter longitude
    # year : Year of occurrence
    # month : Month of occurrence (in streamlit)
    
# Relevant columns for Hazard Mapping :
    # cdi : community decimal intensity (possible)
    # sig : event signifiance score
    # latitude : epicenter latitude
    # longitude : epicenter longitude
    # year : Year of occurrence
    # month : Month of occurrence (in streamlit)
    
# Relevant columns for Magnitude Estimations :
    # magnitude : Magnitude of earthquakes
    # latitude : epicenter latitude
    # longitude : epicenter longitude
    # year : Year of occurrence
    # month : Month of occurrence (in streamlit)
    
    
#                       What kind of data we have ?                           #

print("Data Shape : ", data.shape) # Show the shape of the data (rows x columns)
print("Data Types : ", data.dtypes) # Show the types of data in the dataframe
print("Data Types Count :", data.dtypes.value_counts())  # float, integer & objects

nb_na = (data.isnull().sum().sum() / data.size) * 100  
print("Number of missing data : ", nb_na) # Check NA in the dataset

#                          Removing useless data                              #

data = data.drop(['nst', 'dmin', 'gap'], axis=1)
print("After removing columns :", data.head())

# Creating another column combining year and month
# data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))
data["YearMonth"] = data["Year"].astype(str) + "-" + data["Month"].astype(str).str.zfill(2)
data = data.sort_values(by=["Year", "Month"], ascending=True)
data = data.sort_values(by=["Year"], ascending=True)

#                           Visualizing the data                              #

import cartopy.crs as ccrs # Map of the world
import matplotlib.pyplot as plt # Visualizing graphics
import cartopy.feature as cfeature # Borders and colors

#                              Maps Visualization

# 1- Mapping Earthquakes

# Dividing the latitude, longitude columns for the mapping
min_lat, max_lat = data['latitude'].min(), data['latitude'].max() 
min_lon, max_lon = data['longitude'].min(), data['longitude'].max()

# Creation of the figure
plt.figure(figsize=(20,10))
ax = plt.axes(projection=ccrs.PlateCarree()) # Axes displayed on a flatten map 

ax.stock_img() # Add an underlay image to the map

ax.set_global() # Display a global map

# Adding features to the map 
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Draw borders on the map 
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Boundary of the map with latitude, longitude data
ax.set_extent([min_lon, 
               max_lon, 
               min_lat, 
               max_lat], 
               ccrs.PlateCarree())

ax.coastlines() # Automatically scaled coastline, including major islands.

# Scatter on the zones touched by earthquakes
map_earth = ax.scatter(x=data['longitude'], 
                       y=data['latitude'],
                       c=data['magnitude'],
                       cmap='BuPu',
                       alpha=0.8,
                       transform=ccrs.PlateCarree())

plt.colorbar(map_earth, label='Magnitude') # Scale of the magnitude
plt.title("Earthquake - 2001/2022")
plt.show()

# 2- Mapping Tsunamis

# Creation of the figure
plt.figure(figsize=(20,10))
ax = plt.axes(projection=ccrs.PlateCarree()) # Axes displayed on a flatten map 

ax.stock_img() # Add an underlay image to the map

ax.set_global() # Display a global map

# Adding features to the map 
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Draw borders on the map 
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Boundary of the map with latitude, longitude data
ax.set_extent([min_lon, 
               max_lon, 
               min_lat, 
               max_lat], 
               ccrs.PlateCarree())

ax.coastlines() # Automatically scaled coastline, including major islands.


# Scatter on the zones touched by tsunamis
map_tsuna = ax.scatter(x=data['longitude'], 
                       y=data['latitude'],
                       c=data['tsunami'],
                       cmap='Blues',
                       alpha=0.7,
                       transform=ccrs.PlateCarree())

plt.colorbar(map_tsuna) # Scale of the magnitude
plt.title("Tsunami - 2001/2022")
plt.show()

# 3- Mapping SIG

# Creation of the figure
plt.figure(figsize=(20,10))
ax = plt.axes(projection=ccrs.PlateCarree()) # Axes displayed on a flatten map 

ax.stock_img() # Add an underlay image to the map

ax.set_global() # Display a global map

# Adding features to the map 
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Draw borders on the map 
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Boundary of the map with latitude, longitude data
ax.set_extent([min_lon, 
               max_lon, 
               min_lat, 
               max_lat], 
               ccrs.PlateCarree())

ax.coastlines() # Automatically scaled coastline, including major islands.


# Scatter on the zones touched by tsunamis
map_tsuna = ax.scatter(x=data['longitude'], 
                       y=data['latitude'],
                       c=data['sig'],
                       cmap='YlOrRd',
                       alpha=0.6,
                       transform=ccrs.PlateCarree())

plt.colorbar(map_tsuna) # Scale of the magnitude
plt.title("Signifiance of Event - 2001/2022")
plt.show()


# globe tsunami
# globe earthquake + button for depth and magnitude 
# or using another mapping above it


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
            scene=dict(dragmode='orbit'),
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
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                titleside="right",
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
            scene=dict(dragmode='orbit'),
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
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                titleside="right",
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
            scene=dict(dragmode='orbit'),
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
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                titleside="right",
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
            scene=dict(dragmode='orbit'),
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
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                titleside="right",
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
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                titleside="right",
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
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="middle", y=0.5,
                titleside="right",
                xanchor="left"
            )
        )
    
        return fig3

class HazardMappingAnimation():
    
    def __init__(self, data):
        
        # animation_frame = ['year', 'month'] button
        # figure_choice = ['tsunami', 'earthquake', 'depth'] button
        
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
            

        
        
"""

dataframe

df = pd.DataFrame({'Date': {0: '2018-09-29 00:00:00', 1: '2018-07-28 00:00:00', 2: '2018-07-29 00:00:00', 3: '2018-07-29 00:00:00', 4: '2018-08-01 00:00:00', 5: '2018-08-01 00:00:00', 6: '2018-08-01 00:00:00', 7: '2018-08-05 00:00:00', 8: '2018-09-06 00:00:00', 9: '2018-09-07 00:00:00', 10: '2018-09-07 00:00:00', 11: '2018-09-08 00:00:00', 12: '2018-09-08 00:00:00', 13: '2018-09-08 00:00:00', 14: '2018-10-08 00:00:00', 15: '2018-10-10 00:00:00', 16: '2018-10-10 00:00:00', 17: '2018-10-11 00:00:00', 18: '2018-10-11 00:00:00', 19: '2018-10-11 00:00:00'},
                  'lat': {0: 40.6908284, 1: 40.693601, 2: 40.6951317, 3: 40.6967261, 4: 40.697593, 5: 40.6987141, 6: 40.7186497, 7: 40.7187772, 8: 40.7196151, 9: 40.7196865, 10: 40.7187408, 11: 40.7189716, 12: 40.7214273, 13: 40.7226571, 14: 40.7236955, 15: 40.7247207, 16: 40.7221074, 17: 40.7445859, 18: 40.7476252, 19: 40.7476451},
                  'lon': {0: -73.9336094, 1: -73.9350917, 2: -73.9351778, 3: -73.9355315, 4: -73.9366737, 5: -73.9393797, 6: -74.0011939, 7: -74.0010918, 8: -73.9887851, 9: -74.0035125, 10: -74.0250842, 11: -74.0299202, 12: -74.029886, 13: -74.027542, 14: -74.0290157, 15: -74.0291541, 16: -74.0220728, 17: -73.9442636, 18: -73.9641326, 19: -73.9533039},
                  'count': {0: 1, 1: 2, 2: 5, 3: 1, 4: 6, 5: 1, 6: 3, 7: 2, 8: 1, 9: 7, 10: 3, 11: 3, 12: 1, 13: 2, 14: 1, 15: 1, 16: 2, 17: 1, 18: 1, 19: 1}})

fig = px.density_mapbox(df, lat=df['lat'], 
                            lon=df['lon'], 
                            z=df['count'],
                            radius=10,
                            animation_frame="Date"
                                )
fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=10, mapbox_center = {"lat": 40.7831, "lon": -73.9712},)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

if fig.layout.updatemenus and fig.layout.updatemenus[0].buttons:
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 600

fig.layout.coloraxis.showscale = True   
fig.layout.sliders[0].pad.t = 10
fig.layout.updatemenus[0].pad.t= 10             

st.title('Test')
st.plotly_chart(fig)


"""