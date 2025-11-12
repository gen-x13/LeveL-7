# Imports for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOF

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

# 2) Distplots Visualization

# 1- Tsunami / Magnitude relationship
plt.figure()
sns.distplot(data['tsunami'], label='Tsunami')
sns.distplot(data['magnitude'], label='Magnitude')
plt.title("Tsunami / Magnitude Dist Visualization")
plt.legend()
plt.show()

# 2- Depth / Signifiance event relationship
plt.figure()
sns.distplot(data['sig'], label='SIG')
sns.distplot(data['depth'], label='Depth')
plt.title("Depth / Sig Event Dist Visualization")
plt.legend()
plt.show()

# Doesn't seem to have any relation between those columns


#                             Pre-Processing of data                          #


# Copy of the dataframe from EDA to start freshly
df2 = data.copy()
print(df2)

X = df2[["magnitude", "tsunami", "depth", "longitude", "latitude"]]
print("X Shape : ", X.shape)

# Using LOF algorithm to determine underlying anomalies
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# Stock the prediction
pred = clf.fit_predict(X)

# Stock the score of LOF
neg = clf.negative_outlier_factor_

# ground_truth is here to compare with the results in pred
ground_truth = np.ones(len(X), dtype=int)
n_errors = (pred != ground_truth).sum()

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection

def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])

X1 = X.to_numpy()

plt.scatter(X1[:, 0], X1[:, 1], color="k", s=3.0, label="Data points")
# plot circles with radius proportional to the outlier scores
radius = (neg.max() - neg) / (neg.max() - neg.min())
scatter = plt.scatter(
    X1[:, 0],
    X1[:, 1],
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)
plt.axis("auto")
plt.xlim(X1[:, 0].min() - 1, X1[:, 0].max() + 1)
plt.ylim(X1[:, 1].min() - 1, X1[:, 1].max() + 1)
plt.xlabel("prediction errors: %d" % (n_errors))
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)
plt.title("Local Outlier Factor (LOF) Without Zoom")
plt.show()

# Looking for anomalies zooming on the points higher than 7 
# Because in this dataset, major earthquake events have a 
# magnitude higher than 7.

plt.scatter(X1[:, 0], X1[:, 1], color="k", s=3.0, label="Data points")
# plot circles with radius proportional to the outlier scores
radius = (neg.max() - neg) / (neg.max() - neg.min())
scatter = plt.scatter(
    X1[:, 0],
    X1[:, 1],
    s=1000 * radius,
    edgecolors="b",
    facecolors="none",
    label="Outlier scores",
)
plt.axis("auto")
plt.xlim(6.5, 9.5)
plt.ylim(-0.2, 1.2)
plt.xlabel("prediction errors: %d" % (n_errors))
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)
plt.title("Local Outlier Factor (LOF) With Zoom")
plt.show()

# Checking if LOF really predict what I wanted to see since there's some big circles where it should
# have a smaller circle. Probably because of the latitude and longitude : if points are not close geographically
# it may impact the prediction. I will use StandardScaler to standardize this if it's really a problem.


# I will have to standardize it.

# Standardization and Polynomial Features function (nonlinear data)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# In case I need better results !
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Module to create a pipeline
from sklearn.pipeline import make_pipeline

# To see how it learns and if it works really or not
from sklearn.model_selection import learning_curve # Over-fitting / Under-fitting


# Preprocessor which will be used with all the models
preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), 
                             (StandardScaler()))


# Stock the prediction
X_trans = preprocessor.fit_transform(X)

# New prediction with preprocessor
pred2 = clf.fit_predict(X_trans)

# Stock the score of LOF
neg2 = clf.negative_outlier_factor_

# ground_truth is here to compare with the results in pred
ground_truth = np.ones(len(X), dtype=int)
n_errors = (pred2 != ground_truth).sum()


plt.scatter(X1[:, 0], X1[:, 1], color="k", s=3.0, label="Data points")
# plot circles with radius proportional to the outlier scores
radius = (neg2.max() - neg2) / (neg2.max() - neg2.min())
scatter = plt.scatter(
    X1[:, 0],
    X1[:, 1],
    s=1000 * radius,
    edgecolors="b",
    facecolors="none",
    label="Outlier scores",
)
plt.axis("auto")
plt.xlim(6.5, 9.5)
plt.ylim(-0.2, 1.2)
plt.xlabel("prediction errors: %d" % (n_errors))
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)
plt.title("New LOF With Preprocessors")
plt.show()

X3 = X.copy()

X3["lof_score"] = -neg  # Plus grand = plus anormal
X3["pred"] = pred

# Trier par score décroissant
new_X3 = X3.sort_values("lof_score", ascending=False).head(20)[["magnitude", "tsunami", "depth", "longitude", "latitude", "lof_score"]]

print(new_X3)


# Trying gridsearchcv to try with better parameters

"""
params = {
    "n_neighbors": [5, 10, 15, 20, 25, 30],
    "contamination": [0.02, 0.05, 0.1, 0.15],
    "leaf_size": [20, 30, 40]
}

search = GridSearchCV(
    estimator=LocalOutlierFactor(),
    param_grid=params,
    cv=[(slice(None), slice(None))],  # For one and only split
)
search.fit(X_trans)
best_params = search.best_params_
print(best_params)


# New prediction with preprocessor
pred4= clf.fit_predict(best_params)

# Stock the score of LOF
neg4 = clf.negative_outlier_factor_

# ground_truth is here to compare with the results in pred
ground_truth = np.ones(len(X), dtype=int)
n_errors = (pred4 != ground_truth).sum()

plt.scatter(X1[:, 0], X1[:, 1], color="k", s=3.0, label="Data points")
# plot circles with radius proportional to the outlier scores
radius = (neg4.max() - neg4) / (neg4.max() - neg4.min())
scatter = plt.scatter(
    X1[:, 0],
    X1[:, 1],
    s=1000 * radius,
    edgecolors="g",
    facecolors="none",
    label="Outlier scores",
)
plt.axis("auto")
plt.xlim(6.5, 9.5)
plt.ylim(-0.2, 1.2)
plt.xlabel("prediction errors: %d" % (n_errors))
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)
plt.title("New LOF With Preprocessors")
plt.show()

X4 = X.copy()

X4["lof_score"] = -neg  # Plus grand = plus anormal
X4["pred"] = pred

# Trier par score décroissant
new_X4 = X4.sort_values("lof_score", ascending=False).head(20)[["magnitude", "tsunami", "depth", "longitude", "latitude", "lof_score"]]

print(new_X4)

"""

# Since this solution can't work because LOF and GSCV aren't compatible
# LOF = unsupervised learning and don't have normal parameters
# I will create a wrapper around it to use it like any other models :

from sklearn.base import BaseEstimator, OutlierMixin
# Slitting the data into : Train set | Test set
from sklearn.model_selection import train_test_split

trainset, testset = train_test_split(X, 
                                     test_size=0.3, 
                                     random_state=0)

params = {
    "n_neighbors": [5, 10, 15, 20, 25, 30],
    "contamination": [0.02, 0.05, 0.1, 0.15],
    "leaf_size": [20, 30, 40]
}

a = 0
n_error_list = []

from itertools import product # cartesian product iteration (for list, dict, tuples)
for n_neighbors, contamination, leaf_size in product(params['n_neighbors'],
                                                     params['contamination'],
                                                     params['leaf_size']):
    
    a += 1 # To annotate the number of tries
    model = LocalOutlierFactor(n_neighbors=n_neighbors, 
                                    contamination=contamination, 
                                    leaf_size=leaf_size,
                                    novelty=False)
    
    # Stock the prediction
    X_trans = preprocessor.fit_transform(X)
    # New prediction with preprocessor
    pred5 = model.fit_predict(X_trans)
    # Stock the score of LOF
    neg5 = clf.negative_outlier_factor_
    # ground_truth is here to compare with the results in pred
    ground_truth = np.ones(len(X), dtype=int)
    n_errors = (pred5 != ground_truth).sum()
    
    if n_errors < 20:
        plt.scatter(X1[:, 0], X1[:, 1], color="k", s=3.0, label="Data points")
        # plot circles with radius proportional to the outlier scores
        radius = (neg5.max() - neg5) / (neg5.max() - neg5.min())
        scatter = plt.scatter(
            X1[:, 0],
            X1[:, 1],
            s=1000 * radius,
            edgecolors="g",
            facecolors="none",
            label="Outlier scores",
        )
        plt.axis("auto")
        plt.xlim(6.5, 9.5)
        plt.ylim(-0.2, 1.2)
        plt.xlabel("prediction errors: %d" % (n_errors))
        plt.legend(
            handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
        )
        plt.title(f"New LOF With For Loop {a} with small pred error")
        plt.show()
        
        n_error_list.append({
            "try":a,
            "n_neighbors": n_neighbors,
            "contamination": contamination,
            "leaf_size": leaf_size,
            "n_errors" : n_errors
        })
    else:
        print("Too much errors !")

"""

1 5 0.02 20 16
2 5 0.02 30 16
3 5 0.02 40 16
Too much errors !
13 10 0.02 20 16
14 10 0.02 30 16
15 10 0.02 40 16
Too much errors !
25 15 0.02 20 16
26 15 0.02 30 16
27 15 0.02 40 16
Too much errors !
37 20 0.02 20 16
38 20 0.02 30 16
39 20 0.02 40 16
Too much errors !
49 25 0.02 20 16
50 25 0.02 30 16
51 25 0.02 40 16
Too much errors !
61 30 0.02 20 16
62 30 0.02 30 16
63 30 0.02 40 16
Too much errors !

La plus petite erreur possible : 16

"""

# Getting the min of the list -> best param

best_param = min(n_error_list, key=lambda x: x["n_errors"]) # in function of n_errors
print("Best param : ", best_param)

model_param = LocalOutlierFactor(
                                 n_neighbors=best_param["n_neighbors"],
                                 contamination=best_param["contamination"],
                                 leaf_size=best_param["leaf_size"],
                                 novelty=False
                                 )

X_trans = preprocessor.fit_transform(X)
pred_best = model_param.fit_predict(X_trans)

from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Confusion Matrix with Ground Truth -> as a clean dataset because the dataset is anyway clean

label = [1, -1] # inlier vs outlier
label_w = ["inlier", "outlier"]

# First Visualisation : Confusion Matrix to check FN
cf_matrix = confusion_matrix(ground_truth, pred_best, labels=label)
print(cf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                             display_labels=label_w)

# Creating a dist with the matrix
disp.plot(cmap='Blues')
plt.title("Confusion Matrix LOF with Manual Params Dist")
plt.xlabel("Outlier")
plt.ylabel("Ground Truth")
plt.show()

# Creating a heatmap to see TP/FP/TN/FN

# Variables for the annotation
group_names = ['True Neg','False Pos','False Neg','True Pos']
# Counting the groups 
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
# Making a percentage
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
# Store that
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

# Reshape the dimension of labels
labels = np.asarray(labels).reshape(2,2)

# Creating a heatmap with the matrix
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.title("Confusion Matrix LOF with Manual Params Heat")
plt.xlabel("Outlier")
plt.ylabel("Ground Truth")
plt.show()

# Negative outliers
plt.hist(-model_param.negative_outlier_factor_, bins=50)
plt.title("Score Negative Outliers")
plt.xlabel("Suspected Anomalies")
plt.ylabel("Frequency")
plt.show()

# Outliers vs Inliers
plt.scatter(X1[:, 0], X1[:, 1], c=pred_best, s=3.0, cmap='Blues')
plt.title("Outliers vs Inliers")
plt.show()

"""

Best param :  {'try': 1, 
               'n_neighbors': 5, 
               'contamination': 0.02, 
               'leaf_size': 20, 
               'n_errors': np.int64(16)}

Confusion Matrix :
                
               GroundTruth
               inlier     [[766  16]
               outlier     [  0   0]]
                            in    out   Outlier
                            
True Neg = 766, 97.95%
False Pos = 16, 2.05%
False Neg = 0, 0.00%
True Pos = 0, 0.00%

Everything's good !

"""













