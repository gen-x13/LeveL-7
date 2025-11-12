# Imports for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Maybe add a page to test the model with other data only similar to this one
# RandomizedSearchCV will probably get the right perfomances
# Ca c'est la feuille de recherche, puis il y a l'app à base de ça !
# Peut-être une animation de la carte par années/mois !

"""
It was an error... I was supposed to put X_train2 and y_train2...
I have to retry every model since everything is wrong...
Well, it's still a good lesson because I wanted to work on it quickly 
because I was understanding quickly but, I should have looked at my program 
before.
I've separated the dataset with another df to not confuse the df for 
AdaBoost and the one for the Magnitude Estimations, I should have used 
another file for that.
So I'm doing that right now and see what's wrong and what's correct !
"""


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


#                           Magnitude Estimations


# Copy of the dataframe from EDA to start freshly
df2 = data.copy()

# Removing tsunami since it will not help with our current objective
df2.drop(['tsunami'], axis=1)

# Probablement supprimer d'autres colonnes aussi

# Slitting the data into : Train set | Test set
from sklearn.model_selection import train_test_split

trainset, testset = train_test_split(df2, test_size=0.2, random_state=0)

def preprocessing(df2):
    
    # Separate target from the rest of the data
    X = df2.drop('magnitude', axis=1)
    y = df2['magnitude']
    
    return X, y

# Train Part
X_train2, y_train2 = preprocessing(trainset)

# Evaluation Part
X_test2, y_test2 = preprocessing(testset)

# Since magnitude is a continuous quantity that needs to be predicted, 
# I will use Regression Algorithms.

from sklearn.linear_model import LinearRegression # minimize the residual sum of squares between the observed targets
#from sklearn.tree import DecisionTreeRegressor # for the baggingregressor algo
from sklearn.linear_model import SGDRegressor # the gradient of the loss
from sklearn.ensemble import BaggingRegressor # against overfitting / good accuracy
from sklearn.ensemble import AdaBoostRegressor # Same as ABC but for regression
from sklearn.ensemble import RandomForestRegressor # Same as RF but for regression

# Pipeline Maker
from sklearn.pipeline import make_pipeline
# Standardization and Polynomial Features function (nonlinear data)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Module to create a pipeline
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import learning_curve # Over-fitting / Under-fitting


# Preprocessor which will be used with all the models
preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), 
                             (StandardScaler()))

# Pipeline for each model with their basic hyperparameters
LR = make_pipeline(preprocessor, LinearRegression())
ABR = make_pipeline(preprocessor, AdaBoostRegressor(random_state=0)) # Default Estimator
BR = make_pipeline(preprocessor, BaggingRegressor(random_state=0)) # Default Estimator
SGDR = make_pipeline(preprocessor, SGDRegressor(random_state=0))
RFR = make_pipeline(preprocessor, RandomForestRegressor(random_state=0))


# Dictionnary with the models
dict_of_models2 = {
                  'LinearRegression' : LR,
                  'AdaBoostRegressor' : ABR,
                  'BaggingRegressor': BR,
                  'SGDRegressor': SGDR,
                  'RandomForestRegressor': RFR
                 }

# Evaluation function v2 (use to see which model is not or less over-fitting)
def evaluation_v6(model, title):
    
    # Training
    model.fit(X_train2, y_train2)
    
    # Prediction
    ypred2 = model.predict(X_test2)
    
    # Display RMSE, R2, RSE, RMAE
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    # Using Regression Metrics instead of confusion matrix (only for classification)
    mae = mean_absolute_error(y_test2, ypred2)
    mse = mean_squared_error(y_test2, ypred2) # Can't interprete because it's squarred
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test2, ypred2)
    
    print(f"MAE : {mae:.3f}") # lower values = good
    print(f"RMSE : {rmse:.3f}") # Like MAE, lower values indicate a better fit.
    print(f"R²  : {r2:.3f}") # 1 = perfect fit | 0 = performs no better | neg = performs worse
    
   
    # Visualization of the results : if data is over-fitting or under-fitting
    N, train_score, val_score = learning_curve(model, X_train2, y_train2,
                                              cv=4, scoring='r2', # regression scoring
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    # Figure
    plt.figure(figsize=(15, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.title(f"{title}")
    plt.legend()
    plt.show()

# Iteration of the evaluation for all models
for name, model in dict_of_models2.items():
    print(name)
    evaluation_v6(model, name)

# Results :
    
"""
LinearRegression
MAE : 0.204
RMSE : 0.311
R²  : 0.600
AdaBoostRegressor
MAE : 0.239
RMSE : 0.341
R²  : 0.520
BaggingRegressor
MAE : 0.148
RMSE : 0.297
R²  : 0.635
SGDRegressor
MAE : 0.264
RMSE : 0.375
R²  : 0.419
RandomForestRegressor
MAE : 0.140
RMSE : 0.293
R²  : 0.644
"""

# I will test SGDRegressor.
# This model doesn't overfit or underfit, just need some hyperparameter tuning.

# Change parameters such as the learning rate or the penalty : aka regularization term
SGDR2 = make_pipeline(preprocessor, SGDRegressor(
                                                penalty = 'elasticnet', # Ridge + Lasso (L2+L1)
                                                learning_rate='adaptive', # Adjusted based on the progress of the weight updates
                                                eta0=0.01, # Initial learning rate
                                                max_iter=2000, # Iteration
                                                tol=1e-4, # tolerance of research
                                                random_state=0))

# Take a look at the pipeline
print(SGDR2)

# Using this technic again
param2 = dict() 
# Constant that multiplies the regularization term
param2['sgdregressor__alpha'] = [1e-5, 1e-4, 1e-3] 
# Lasso / Ridge or the combination of them
param2['sgdregressor__penalty'] = ['l1', 'l2', 'elasticnet']
# Type of learning rate
param2['sgdregressor__learning_rate'] = ['adaptive', 'constant']
param2['sgdregressor__eta0'] = [0.001, 0.01, 0.1]

# Pipeline hyperparameters
param2['pipeline__polynomialfeatures__degree']=[2, 3]

# Lecture : l'hyperparam gamma appartient à svc (d'où le svc d'abord ou pipeline)

# Variable grid stocking the best parameters for the model
grid2 = GridSearchCV(SGDR2, param2, scoring='r2', cv=4)

# Training
grid2.fit(X_train2, y_train2)

print("Best R²:", grid2.best_score_)
print("Best params:", grid2.best_params_)

def evaluation_v7(model):
    
    # Training
    model.fit(X_train2, y_train2)
    
    # Prediction
    ypred2 = model.predict(X_test2)
    
    # Display RMSE, R2, RSE, RMAE
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    # Using Regression Metrics instead of confusion matrix (only for classification)
    mae = mean_absolute_error(y_test2, ypred2)
    mse = mean_squared_error(y_test2, ypred2) # Can't interprete because it's squarred
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test2, ypred2)
    
    print(f"MAE : {mae:.3f}") # lower values = good
    print(f"RMSE : {rmse:.3f}") # Like MAE, lower values indicate a better fit.
    print(f"R²  : {r2:.3f}") # 1 = perfect fit | 0 = performs no better | neg = performs worse
    
   
    # Visualization of the results : if data is over-fitting or under-fitting
    N, train_score, val_score = learning_curve(model, X_train2, y_train2,
                                              cv=4, scoring='r2', # regression scoring
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    # Figure
    plt.figure(figsize=(15, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.title("SGDRegressor After Hyperparameter Tuning GSCV")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8,8))
    plt.scatter(y_test2, ypred2, alpha=0.6, color='teal')
    plt.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 'r--')
    plt.xlabel("Real values")
    plt.ylabel("Pred values")
    plt.title("SGDRegressor : predicted vs real GSCV")
    plt.show()
    
    
# Evaluation finale des meilleurs paramètres
evaluation_v7(grid2.best_estimator_)

"""
Not good at all ! : train and valid somwhere between 1 and 0, and the
results of valid start from - 4
MAE : 0.223
RMSE : 0.328
R²  : 0.556
"""

# New version with loss, RandomizedSearchCV, etc
SGDR3 = make_pipeline(preprocessor, SGDRegressor(loss='huber',
                                                penalty = 'elasticnet', # Ridge + Lasso (L2+L1)
                                                learning_rate='adaptive', # Adjusted based on the progress of the weight updates
                                                eta0=0.01, # Initial learning rate
                                                max_iter=2000, # Iteration
                                                tol=1e-4, # tolerance of research
                                                random_state=0))

from scipy.stats import loguniform, uniform

param_distributions = {
    
    'pipeline__polynomialfeatures__degree': [2, 3, 4],
    'sgdregressor__alpha': loguniform(1e-5, 1e-1),
    'sgdregressor__eta0': loguniform(1e-4, 0.1),
    'sgdregressor__l1_ratio': uniform(0, 1),
    'sgdregressor__penalty': ['l1', 'l2', 'elasticnet']
}

search = RandomizedSearchCV(
    SGDR3, # model
    param_distributions=param_distributions,
    n_iter=50,            # numbers of iteration for tests
    cv=5,
    scoring='r2',
    random_state=0,
    n_jobs=-1
)

# Training
search.fit(X_train2, y_train2)

print("Best R²:", search.best_score_)
print("Best params:", search.best_params_)

def evaluation_v8(model):
    
    # Training
    model.fit(X_train2, y_train2)
    
    # Prediction
    ypred2 = model.predict(X_test2)
    
    # Display RMSE, R2, RSE, RMAE
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    # Using Regression Metrics instead of confusion matrix (only for classification)
    mae = mean_absolute_error(y_test2, ypred2)
    mse = mean_squared_error(y_test2, ypred2) # Can't interprete because it's squarred
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test2, ypred2)
    
    print("RSCV")
    print(f"MAE : {mae:.3f}") # lower values = good
    print(f"RMSE : {rmse:.3f}") # Like MAE, lower values indicate a better fit.
    print(f"R²  : {r2:.3f}") # 1 = perfect fit | 0 = performs no better | neg = performs worse
    
   
    # Visualization of the results : if data is over-fitting or under-fitting
    N, train_score, val_score = learning_curve(model, X_train2, y_train2,
                                              cv=4, scoring='r2', # regression scoring
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    # Figure
    plt.figure(figsize=(15, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.title("SGDRegressor After Hyperparameter Tuning RSCV")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8,8))
    plt.scatter(y_test2, ypred2, alpha=0.6, color='teal')
    plt.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 'r--')
    plt.xlabel("Real values")
    plt.ylabel("Pred values")
    plt.title("SGDRegressor : predicted vs real RSCV")
    plt.show()
    
    
# Evaluation finale des meilleurs paramètres
evaluation_v8(search.best_estimator_)

"""
RSCV
MAE : 0.189
RMSE : 0.320
R²  : 0.576

Same as previously, we don't know, but it's between 1 and 0
"""

# The blue dots are not aligned with the red line. SGDRegressor isn't enough good for
# the prediction, even with a r² with 0.6 and without overfitting or underfitting.

# Since it's a fail with SGDRegressor, I decided to either tune RandomForestRegressor
# or use GradientBoostingRegressor 


# First, try with GradientBoostingRegressor :
    
from sklearn.ensemble import GradientBoostingRegressor

GBR = make_pipeline(preprocessor, GradientBoostingRegressor(random_state=0))


def evaluation_v9(model):
    
    # Training
    model.fit(X_train2, y_train2)
    
    # Prediction
    ypred2 = model.predict(X_test2)
    
    # Display RMSE, R2, RSE, RMAE
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    # Using Regression Metrics instead of confusion matrix (only for classification)
    mae = mean_absolute_error(y_test2, ypred2)
    mse = mean_squared_error(y_test2, ypred2) # Can't interprete because it's squarred
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test2, ypred2)
    
    print("GradientBoostingR")
    print(f"MAE : {mae:.3f}") # lower values = good
    print(f"RMSE : {rmse:.3f}") # Like MAE, lower values indicate a better fit.
    print(f"R²  : {r2:.3f}") # 1 = perfect fit | 0 = performs no better | neg = performs worse
    
   
    # Visualization of the results : if data is over-fitting or under-fitting
    N, train_score, val_score = learning_curve(model, X_train2, y_train2,
                                              cv=4, scoring='r2', # regression scoring
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    # Figure
    plt.figure(figsize=(15, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.title(f"GradientBoostingRegressor")
    plt.legend()
    plt.show()


evaluation_v9(GBR)

# GBR results :
"""
GradientBoostingR
MAE : 0.166
RMSE : 0.306
R²  : 0.611
"""

# 90% of performance is actually what we need for the magnitude estimations.
# But, I don't like the small overfitting. I will tune the hyperparameters
# without changing the r² score.

GBR2 = make_pipeline(preprocessor, GradientBoostingRegressor(
                                                            #loss=''
                                                            learning_rate=0.01, # Small learning_rate
                                                            n_estimators=2000, # More estimators
                                                            subsample=0.7, # inferior to 1.0 to decrease variance
                                                            max_depth=3,
                                                            random_state=0))


print(GBR2)

# Using this technic again
param3 = dict() 

# GradientBoostingRegressor hyperparameters
param3['gradientboostingregressor__n_estimators'] = [100, 500, 1000]
param3['gradientboostingregressor__learning_rate'] = loguniform(1e-6, 1e-2)
param3['gradientboostingregressor__subsample'] = uniform(0.7, 0.9)
#param3['gradientboostingregressor__max_depth'] = [3, 4, 5, 6, 7]

# Pipeline hyperparameters
param3['pipeline__polynomialfeatures__degree']=[2, 3]


rand4 = RandomizedSearchCV(GBR2, param_distributions=param3, n_iter=50, cv=4, scoring='r2',
                           n_jobs=-1, verbose=2)

# Training
rand4.fit(X_train2, y_train2)

print("Best R²:", rand4.best_score_)
print("Best params:", rand4.best_params_)


def evaluation_v10(model):
    
    # Training
    model.fit(X_train2, y_train2)
    
    # Prediction
    ypred2 = model.predict(X_test2)
    
    # Display RMSE, R2, RSE, RMAE
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    # Using Regression Metrics instead of confusion matrix (only for classification)
    mae = mean_absolute_error(y_test2, ypred2)
    mse = mean_squared_error(y_test2, ypred2) # Can't interprete because it's squarred
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test2, ypred2)
    
    print("GradientBoostingR RSCV New Version")
    print(f"MAE : {mae:.3f}") # lower values = good
    print(f"RMSE : {rmse:.3f}") # Like MAE, lower values indicate a better fit.
    print(f"R²  : {r2:.3f}") # 1 = perfect fit | 0 = performs no better | neg = performs worse
    
   
    # Visualization of the results : if data is over-fitting or under-fitting
    N, train_score, val_score = learning_curve(model, X_train2, y_train2,
                                              cv=4, scoring='r2', # regression scoring
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    # Figure
    plt.figure(figsize=(15, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.title(f"GradientBoostingRegressor RSCV New Version")
    plt.legend()
    plt.show()

evaluation_v10(rand4.best_estimator_)


"""
GradientBoostingR RSCV New Version
MAE : 0.162
RMSE : 0.303
R²  : 0.620
"""

# With max_depth to suppress the overfit

# Using this technic again
param4 = dict() 

# GradientBoostingRegressor hyperparameters
param4['gradientboostingregressor__n_estimators'] = [100, 500, 1000]
param4['gradientboostingregressor__learning_rate'] = loguniform(1e-6, 1e-2)
param4['gradientboostingregressor__subsample'] = uniform(0.7, 0.9)
param4['gradientboostingregressor__max_depth'] = [3, 4, 5, 6, 7]

# Pipeline hyperparameters
param4['pipeline__polynomialfeatures__degree']=[2, 3]


rand5 = RandomizedSearchCV(GBR2, param_distributions=param4, n_iter=50, cv=4, scoring='r2',
                           n_jobs=-1, verbose=2)

# Training
rand5.fit(X_train2, y_train2)

print("Best R²:", rand5.best_score_)
print("Best params:", rand5.best_params_)


def evaluation_v11(model):
    
    # Training
    model.fit(X_train2, y_train2)
    
    # Prediction
    ypred2 = model.predict(X_test2)
    
    # Display RMSE, R2, RSE, RMAE
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    # Using Regression Metrics instead of confusion matrix (only for classification)
    mae = mean_absolute_error(y_test2, ypred2)
    mse = mean_squared_error(y_test2, ypred2) # Can't interprete because it's squarred
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test2, ypred2)
    
    print("GradientBoostingR RSCV New Version 5 (Xtrain problem)")
    print(f"MAE : {mae:.3f}") # lower values = good
    print(f"RMSE : {rmse:.3f}") # Like MAE, lower values indicate a better fit.
    print(f"R²  : {r2:.3f}") # 1 = perfect fit | 0 = performs no better | neg = performs worse
    
   
    # Visualization of the results : if data is over-fitting or under-fitting
    N, train_score, val_score = learning_curve(model, X_train2, y_train2,
                                              cv=4, scoring='r2', # regression scoring
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    # Figure
    plt.figure(figsize=(15, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.title(f"GradientBoostingRegressor RSCV New Version 5")
    plt.legend()
    plt.show()

evaluation_v11(rand5.best_estimator_)


"""

GradientBoostingR RSCV New Version 5 (Xtrain problem)
MAE : 0.157
RMSE : 0.299
R²  : 0.631

"""

# With max_depth, I loose in fit but gain in r² score
# Without, it's the opposite


# Trying with max_depth = 10/15, maybe it was constraint

# Using this technic again
param5 = dict() 

# GradientBoostingRegressor hyperparameters
param5['gradientboostingregressor__n_estimators'] = [100, 500, 1000]
param5['gradientboostingregressor__learning_rate'] = loguniform(1e-6, 1e-2)
param5['gradientboostingregressor__subsample'] = uniform(0.7, 0.9)
param5['gradientboostingregressor__max_depth'] = [5, 10, 15, 20]

# Pipeline hyperparameters
param5['pipeline__polynomialfeatures__degree']=[2, 3]


rand6 = RandomizedSearchCV(GBR2, param_distributions=param5, n_iter=50, cv=4, scoring='r2',
                           n_jobs=-1, verbose=2)

# Training
rand6.fit(X_train2, y_train2)

print("Best R²:", rand6.best_score_)
print("Best params:", rand6.best_params_)


def evaluation_v11(model):
    
    # Training
    model.fit(X_train2, y_train2)
    
    # Prediction
    ypred2 = model.predict(X_test2)
    
    # Display RMSE, R2, RSE, RMAE
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    # Using Regression Metrics instead of confusion matrix (only for classification)
    mae = mean_absolute_error(y_test2, ypred2)
    mse = mean_squared_error(y_test2, ypred2) # Can't interprete because it's squarred
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test2, ypred2)
    
    print("GradientBoostingR RSCV New Version 6")
    print(f"MAE : {mae:.3f}") # lower values = good
    print(f"RMSE : {rmse:.3f}") # Like MAE, lower values indicate a better fit.
    print(f"R²  : {r2:.3f}") # 1 = perfect fit | 0 = performs no better | neg = performs worse
    
   
    # Visualization of the results : if data is over-fitting or under-fitting
    N, train_score, val_score = learning_curve(model, X_train2, y_train2,
                                              cv=4, scoring='r2', # regression scoring
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    # Figure
    plt.figure(figsize=(15, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.title(f"GradientBoostingRegressor RSCV New Version 6")
    plt.legend()
    plt.show()

evaluation_v11(rand6.best_estimator_)


"""
GradientBoostingR RSCV New Version 6
MAE : 0.154
RMSE : 0.303
R²  : 0.621

"""


"""
RandomForestRegressor : train : 0.95, valid : 0.75
MAE : 0.140
RMSE : 0.293
R²  : 0.644
"""









