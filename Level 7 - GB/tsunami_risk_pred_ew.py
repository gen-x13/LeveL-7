# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 06:44:24 2026

@author: Genxcode
"""

# Imports for the project
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Security preventing any reading problem
try:
    data = pd.read_csv(Path(__file__).parent / "data" / "earthquake_data_tsunami.csv")
except Exception as e:
    print(f"{e}")
  
    
#                       What kind of data we have ?                           #

nb_na = (data.isnull().sum().sum() / data.size) * 100

#                          Removing useless data                              #

data = data.drop(['nst', 'dmin', 'gap', 'cdi', 'mmi', 'sig'], axis=1)

#                           Visualizing the data                              #

df1 = data.copy() # copy from the EDA in case we need the EDA version later

# Slitting the data into : Train set | Test set
from sklearn.model_selection import train_test_split

trainset, testset = train_test_split(df1, test_size=0.2, random_state=0)

def preprocessing(df1):
    
    # Separate target from the rest of the data
    X = df1.drop('tsunami', axis=1)
    y = df1['tsunami']
    
    return X, y

# Train Part
X_train, y_train = preprocessing(trainset)

# Evaluation Part
X_test, y_test = preprocessing(testset)


# --------------------------------------------------------------------------- #

#                             MODULE IMPORTATION                              #

# Used of GridSearchCV for Best Estimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Classification models
from sklearn.svm import SVC
# Module to create a pipeline
from sklearn.pipeline import make_pipeline
# Test with one classification model :
from sklearn.tree import DecisionTreeClassifier
# Classification Models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# To select features/data
from sklearn.feature_selection import SelectKBest, f_classif
# Standardization and Polynomial Features function (nonlinear data)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# Procedure for evaluating the model with data 
from sklearn.metrics import f1_score # harmonic mean of the precision and recall
from sklearn.metrics import confusion_matrix # F-P / T-P / T-N / F-N
from sklearn.metrics import classification_report # Report of confusion, accuracy, f1
from sklearn.model_selection import learning_curve # Over-fitting / Under-fitting


#                            MODEL CREATION PART                              #
# Preprocessor which will be used with all the models
preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), 
                             SelectKBest(f_classif, k=10))

# Using Decision Tree with 2 as max_depth instead of the default 1
# Balanced parameter : Automatically compensate for imbalance
base = DecisionTreeClassifier(max_depth=2, 
                              class_weight='balanced')

# First Version of Ada Boost for Prediction

# Auto n_estimator best etc
AdaBoostPred = make_pipeline(preprocessor, 
                          AdaBoostClassifier(estimator=base,
                                             n_estimators=200, # Number of trees
                                             learning_rate=0.05, # Step size shrinkage
                                             random_state=0
                                            )
                          )


#               Dictionnary welcoming future hyperparameters                  #

param = dict() 

#                    All hyperparameters for tuning                           #

# Number of estimators / weak learners 
param['adaboostclassifier__n_estimators'] = [10, 50, 100, 500] 
# Controls the loss function used for calculating the weight of the base models
param['adaboostclassifier__learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
param['pipeline__polynomialfeatures__degree']=[2, 3]
param['pipeline__selectkbest__k'] = list(range(45, 60))

#                   Tuning of the model with hyperparamaters                  #

# Variable grid stocking the best parameters for the model
rand = RandomizedSearchCV(AdaBoostPred, param, scoring='recall', cv=4)

#                       Model training and Prediction                         #
# Training
rand.fit(X_train, y_train)

# Prediction
y_pred = rand.predict(X_test)

#                        Evaluation of the model                              #

def evaluation_v1(model):
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
# Final Evaluation of best parameters
model = evaluation_v1(rand.best_estimator_)


# --------------------------------------------------------------------------- #

import time
import pandas as pd 

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GeonamesCountriesTxtFileReader(object):
 """
  A GeonamesCountriesTxtFileReader 
 """
 def __init__(_self, input_file):
  """Return a GeonamesCountriesTxtFileReader object""" 
  
  _self.input_file = input_file # underscore to not hash it because it's a custom function
  
  
 @st.cache_data
 def get_header_row(_self): # Name of the columns
  return [ 
   "geonameid",
   "name",
   "asciiname",
   "alternatenames",
   "latitude",
   "longitude",
   "feature_class",
   "feature_code",
   "country_code",
   "cc2",
   "admin1_code",
   "admin2_code",
   "admin3_code",
   "admin4_code",
   "population",
   "elevation",
   "dem",
   "timezone",
   "modification_date" ]

 @st.cache_data
 def get_data_types(_self): # The type of data for each columns
  return {
   "geonameid": "int32",
   "name": "string",
   "asciiname": "string",
   "alternatenames": "string",
   "latitude": "float32",
   "longitude": "float32",
   "feature_class": "category",
   "feature_code": "category",
   "country_code": "category",
   "cc2": "category",
   "admin1_code": "category",
   "admin2_code": "category",
   "admin3_code": "category",
   "admin4_code": "category",
   "population": "int32",
   "elevation": "string",
   "dem": "string",
   "timezone": "category",
   "modification_date": "string",
   }

# Got some issues with my first try with the types, that's when I understand
# That I should have looked through it more or maybe, just check.
# But with Sublime Text, I couldn't understand those data.
 @st.cache_data
 def read_csv(_self):
  start = time.time()
  df = pd.read_csv(
   _self.input_file,
   delim_whitespace=False, # specifies whether or not whitespace (e.g. ' ' or '\t') will be used as the sep delimiter.
   sep='\t', # the famous separator
   skiprows=0, # this parameter is use to skip passed rows in new data frame
   encoding='utf-8', # encoding standard
   names=_self.get_header_row(), # pass rows as headers for df
   dtype=_self.get_data_types(), # dtypes standard to dodge dtypes problems and non standardization
   na_values=['none'], # no nan data
   usecols=[
            "name",
            "latitude",
            "longitude",
            "country_code",
            "admin1_code",
        ]) # Only the useful columns less RAM
   # engine = 'python' just in case we get a warning : 
  
  end = time.time()
  logger.info('Read CSV File (path = {}, elapsed-time = {})'.format(_self.input_file, (end - start)))
  return df


# File path
geonames_df = Path(__file__).parent / "data" / "cities15000.txt"

# df reader and transformator
reader1 = GeonamesCountriesTxtFileReader(geonames_df)
df = reader1.read_csv()

@st.cache_data
# Function to found the nearest city, country or state based on lat/lon data
def nearest_city(lat, lon, _df):
    latitudes = _df["latitude"].values
    longitudes = _df["longitude"].values
    
    # Euclidian Distance from lat/lon from original df vs lat/lon from geonames_df
    dist = (latitudes - lat)**2 + (longitudes - lon)**2
    idx = np.argmin(dist)
    
    return _df.iloc[idx][["name","admin1_code","country_code"]]

# -------------------------------- API PART ----------------------------------#

@st.cache_data
def initialization(df2, cities_df):
    
    cities, countries, states, = [], [], []

    for latitudes, longitudes in zip(df2['latitude'], df2['longitude']):
        
        try:
        
            nearest = nearest_city(latitudes, longitudes, cities_df)
            
            cities.append(nearest["name"])
            states.append(nearest["admin1_code"])
            countries.append(nearest["country_code"])
            
        
        except Exception as e:
            print("Error: ", e)
            
            cities.append('error')
            states.append('error')
            countries.append('error')
            
    df2["city"] = cities
    df2["state"] = states
    df2["country"] = countries
    
    df2["state"] = df2["state"].fillna("")
    df2["country"] = df2["country"].sort_values()
    
    return df2


# --------------------------------------------------------------------------- #
#          Tsunami Risk Class For Animated Estimation Dashboard               #

class TsunamiRisk():
    
    def __init__(self, df_data):
        
        # df
        df_copy = df_data
        
        # df with countries
        cities_df = df
        self.df2 = initialization(df_copy, cities_df)
    
    def tsunami_estimation_graph(self, country): # plus tard Mode : Y/M
       
       from plotly.subplots import make_subplots
       import plotly.graph_objects as go
       
       # Filtering for the user's country
       df_country = self.df2[self.df2["country"] == country]
       
       # Sorting Years
       years = sorted(df_country["Year"].unique())
       
       # Figure
       fig_tsu = make_subplots(
                                rows=1,
                                cols=2,
                                specs=[
                                    [{"type": "bar"}, {"type": "pie"}]
                                ]
                            )
       
       # Per year animation
       df0 = df_country[df_country["Year"] == years[0]]
       
       # Bar
       fig_tsu.add_trace(
       go.Bar(x=df0["magnitude"], 
              y=df0["depth"], 
              marker_color=df0["tsunami"], 
              ),
             row=1, col=1,
       )
       
       # Pie
       fig_tsu.add_trace(
       go.Pie(labels=df0['magnitude'], values=df0['depth'],
              marker=dict(colors=df0["tsunami"]), #, showlegend=False
              ),
             row=1, col=2,
       )
       
       # Display
       fig_tsu.update_layout(height=600, width=1400, title_text=f"Earthquake & Tsunami's Risk in {country}")    
       
       # Animation Parameters
       frames = []
       for year in years:
           dfy = df_country[df_country["Year"] == year]
           frames.append (
                        go.Frame(
                            data=[
                                go.Bar(
                                    x=dfy["magnitude"],
                                    y=dfy["depth"],
                                    marker_color=dfy["tsunami"]
                                ),
                                go.Pie(labels=dfy['magnitude'], 
                                       values=dfy['depth'],
                                       marker=dict(colors=df0["tsunami"]), #, showlegend=False
                                       )
                            ],
                            name=str(year)
                        )
                    )
                       
       fig_tsu.frames=frames
         # customize this frame duration according to your data!!!!!
       sliders = [
                   {
                       "pad": {"t": 50},
                       "len": 0.9,
                       "x": 0.1,
                       
                       "steps": [
                           {
                               "args": [
                                            [str(year)],
                                            {"frame": {"duration": 1000}, "mode": "immediate"}
                                        ],
                               "label": str(year),
                               "method": "animate",
                           }
                           for year in years
                       ],
                   }
               ]
       
       
       fig_tsu.update_layout(
                                updatemenus=[{
                                    "type": "buttons",
                                    "direction": "right",
                                    "x": 0.1,
                                    "y": -0.15,   # under slider
                                    "pad": {"t": 0, "r": 10},
                                    "buttons": [
                                        {
                                            "label": "Play",
                                            "method": "animate",
                                            "args": [None, {"frame": {"duration": 3000, "redraw": True}, "fromcurrent": True, "transition": {"duration": 500}}]
                                        },
                                        {
                                            "label": "Pause",
                                            "method": "animate",
                                            "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                                        }
                                    ]
                                }],
                                sliders=sliders,
                                height=600,
                                width=900,
                                title=f"Earthquake & Tsunami Risk Evolution in {country}"
                            )
       
       
       st.plotly_chart(fig_tsu) 
       
       
# --------------------------------------------------------------------------- #
#          Tsunami Risk Pred Class For Prediction Based on Data               #

class TsunamiRiskPred():
    
    def __init__(self, city, year, month, depth, magnitude, df_data):
        
        # Estimator used for the prediction
        self.estimator = rand.best_estimator_ 
        
        # Storing the dataframe with the column country inside
        tsunami_risk = TsunamiRisk(df_data)
        self.df = tsunami_risk.df2
                
        # Recover the latitude and longitude from the country column by using
        # a mask and the function mean to select only one above all
        selected_city = self.df["country"]==city # mask based on the city 
        latitude = self.df[selected_city]["latitude"].mean() # applied to find one latitude
        longitude = self.df[selected_city]["longitude"].mean() # applied to find one longitude
        
        # Dictionnary future dataframe
        self.dict = {
                     'latitude':latitude,
                     'longitude':longitude,
                     'magnitude':magnitude,
                     'depth':depth,
                     'Year':year,
                     'Month':month
                     }
        
        # Transform the dictionnary in a dataframe to be used by the estimator
        self.X = pd.DataFrame(self.dict, index=[0]) # index 0 = 1 row
        self.X = self.X[X_train.columns]
        
        # Result = the predicted result in the function "prediction"
        self.result = self.prediction()
    
    def prediction(self):
        
        result = self.estimator.predict(self.X)
        
        print(int(result[0]))
        return int(result[0]) # because result = [1] and it's a numpy array not an int
    
# --------------------------------------------------------------------------- #    
#                  New Data fitting our Early Warning features                #

# copy from the EDA in case we need the EDA version later   
df2 = df1.copy() 

# Removing useless columns
df2 = df2.drop(['Year', 'Month'], axis=1)

# Slitting the data into : Train set | Test set
from sklearn.model_selection import train_test_split

trainset, testset = train_test_split(df2, test_size=0.2, random_state=0)

def preprocessing(df2):
    
    # Separate target from the rest of the data
    X = df2.drop('tsunami', axis=1)
    y = df2['tsunami']
    
    return X, y

# Train Part
X_train, y_train = preprocessing(trainset)

# Evaluation Part
X_test, y_test = preprocessing(testset)

# --------------------------------------------------------------------------- #    
#                       Model training and Prediction                         #

# Training
rand.fit(X_train, y_train)

# Prediction
y_pred = rand.predict(X_test)

#                        Evaluation of the model                              #

def evaluation_v2(model):
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
    # Using this report to check the model
    print(classification_report(y_test, ypred))
    
    return model
    
    
# Final Evaluation of best parameters
model2 = evaluation_v2(rand.best_estimator_)

# --------------------------------------------------------------------------- #
#                       Function to calculate distance                        #

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Rayon de la Terre en km

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2)**2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# --------------------------------------------------------------------------- #
#   Tsunami Risk Early War Class For Live Alarm Based on GEOJSON from USGS    #

# Import
import json
import requests
import pandas as pd

# Tuning pandas future dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500)

# Class containing user input and the functions for prediction
class TsunamiRiskEW():
    
    def __init__(self, lat, lon):
        
        # Define the data we need for scraping 
        self.start_time = 'now-24hours' # 'now-180days' -> historic of 180 days
        self.min_magnitude = 3
        self.latitude = lat
        self.longitude = lon
        self.max_radius_km = 2000
        
        # Estimator used for the prediction
        self.estimator = rand.best_estimator_ 

        # Get data from a request to the Earthquake USGS API
        url = requests.get(f'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={self.start_time}&minmagnitude={self.min_magnitude}&latitude={self.latitude}&longitude={self.longitude}&maxradiuskm={self.max_radius_km}')
        
        # Storing the data(HTTPS block) as a json (Python Format) : a dictionnary in a json format
        df = url.json()

        # Implement features inside the dataset
        self.features = df['features']
        
        if len(self.features) == 0:
            print("⚠️ No seism around this coordinate")
            
        self.no_earthquake = False
        
        # The features as lists
        self.places = [] # Display for the user places name 
        self.mags = [] # Display magnitude on interface
        self.times = [] # Display real time hour on interface
        self.lats = [] # Display the choosen lat on interface
        self.lons = [] # Dislay the choosen lon on interface
        self.depths = [] # Display the depth of the earthquake on interface
        self.tsunamis = [] # Display the tsunami in real time
        
        # Appending every features inside lists
        for feature in self.features:
            self.places.append(feature['properties']['place'])
            self.mags.append(feature['properties']['mag'])
            self.times.append(pd.to_datetime(feature['properties']['time'], unit='ms').strftime('%y/%m/%d %H:%M:%S'))
            self.lats.append(feature['geometry']['coordinates'][1]) # in the json file it's the second coordinate
            self.lons.append(feature['geometry']['coordinates'][0]) # in the json file it's the first coordinate
            self.depths.append(feature['geometry']['coordinates'][2]) # in the json file it's the third coordinate, which isn't a coordinate though
            self.tsunamis.append(feature['properties']['tsunami']) # in the json file it's the third coordinate, which isn't a coordinate though
        
        # Stocking the result of the prediction to be displayed on interface
        self.result = self.predictionEW()
        
    def build_dataframe(self):
        
        rows = [] # rows for a dataframe
        
        # Adding features 
        for param in self.features:
            rows.append({
                
                "magnitude" : param['properties']['mag'],
                "depth" : param['geometry']['coordinates'][2],
                "latitude" : param['geometry']['coordinates'][1],
                "longitude" : param['geometry']['coordinates'][0]
                
                })
        
        # Transforming Rows into dataframe for EW
        return pd.DataFrame(rows)
    
    def predictionEW(self):
        
        # Stock the build dataframe result in X 
        X = self.build_dataframe()
        
        # If a there's no tsunami, no X probabilities, then it's equal to 0.0
        if X.empty or not {"latitude", "longitude"}.issubset(X.columns): # subset : dans X colonnes
            
            self.no_earthquake = True
            return None
        
        else:
            
            # Adding a distance that will be used later with a filter
            X["distance_km"] = haversine(
                self.latitude, self.longitude, # choosen latitude and longitude
                X["latitude"], X["longitude"]  # second latitude and longitude
            )
            
            # Filtrating within a radius of 1500 km
            X = X[X["distance_km"] <= self.max_radius_km]
        
            X_proba = X[["magnitude", "depth", "latitude", "longitude"]]
            
            # Probabilities of the features to lead to a possible tsunami
            X["proba"] = self.estimator.predict_proba(X_proba)[:,1]
            
            # Since I have a set of seisms, I need a risk_score being the mean of all probabilities
            risk_score = X["proba"].mean()
            
            print("\n 🪐 RISK SCORE HERE :")
            print(risk_score)
            
            # Plus tard je mettrais des précisions avec les lieux où se trouvent réellement
            # chaque probabilités + noms du pays/villes grâce au JSON !
            
            return risk_score







