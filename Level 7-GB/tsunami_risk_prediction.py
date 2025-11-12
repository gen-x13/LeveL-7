# Imports for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

nb_na = (data.isnull().sum().sum() / data.size) * 100

#                          Removing useless data                              #

data = data.drop(['nst', 'dmin', 'gap'], axis=1)

#                           Visualizing the data                              #

import cartopy.crs as ccrs # Map of the world
import matplotlib.pyplot as plt # Visualizing graphics
import cartopy.feature as cfeature # Borders and colors

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



#                                 Modellising the data

# Classification models
from sklearn.svm import SVC
# Module to create a pipeline
from sklearn.pipeline import make_pipeline


# Test with one classification model :

from sklearn.tree import DecisionTreeClassifier

model_0 = DecisionTreeClassifier(random_state=0) 

# Procedure for evaluating the model with data 
from sklearn.metrics import f1_score # harmonic mean of the precision and recall
from sklearn.metrics import confusion_matrix # F-P / T-P / T-N / F-N
from sklearn.metrics import classification_report # Report of confusion, accuracy, f1
from sklearn.model_selection import learning_curve # Over-fitting / Under-fitting

# Evaluation function v1
def evaluation_v1(model, title):
    
    # Training
    model.fit(X_train, y_train)
    
    # Prediction
    ypred = model.predict(X_test)
    
    # Display a report and a confusion matrix from the prediction
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
        
    # Visualization of the results : if data is over-fitting or under-fitting
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='f1', # classification scoring
                                              # From 0.1 to 1 with 10 split
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    # Figure
    plt.figure(figsize=(15, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.title(f"{title}")
    plt.legend()
    plt.show()

# Evaluate the model of DTC
evaluation_v1(model_0, "DecisionTreeClassifier")

# Classification Models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Pipeline Maker
from sklearn.pipeline import make_pipeline
# To select features/data
from sklearn.feature_selection import SelectKBest, f_classif
# Standardization and Polynomial Features function (nonlinear data)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Preprocessor which will be used with all the models
preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), 
                             SelectKBest(f_classif, k=10))

# Pipeline for each model with their basic hyperparameters
RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
# Unlike decision trees, SVM and KNN require a normalized dataset
# hence StandardScaler and KNN.
SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())

# Dictionnary with the models
dict_of_models = {'RandomForest': RandomForest,
                  'AdaBoost' : AdaBoost,
                  'SVM': SVM,
                  'KNN': KNN
                 }

# Evaluation function v2 (use to see which model is not or less over-fitting)
def evaluation_v2(model, title):
    
    # Training
    model.fit(X_train, y_train)
    
    # Prediction
    ypred = model.predict(X_test)
    
    # Display a report and a confusion matrix from the prediction
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
        
    # Visualization of the results : if data is over-fitting or under-fitting
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='f1',
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    # Figure
    plt.figure(figsize=(15, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.title(f"{title}")
    plt.legend()
    plt.show()

# Iteration of the evaluation for all models
for name, model in dict_of_models.items():
    print(name)
    evaluation_v2(model, name)

# I choose to use Ada Boost : ~0.87 for the test set
# Recall : 0.97 and Precision : 0.83


# --------------------------- Optimisation ---------------------------------- #

# Utilisation de GridSearchCV pour Best Estimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Visualizing AdaBoost Hyperparameters
print("AdaBoost Hyperparameters :")
print(AdaBoost)


#                      First Test with GridSearchCV

# Hyperparameter Tuning

# Dictionnary welcoming future hyperparameters
param = dict() 
# Number of estimators / weak learners 
param['adaboostclassifier__n_estimators'] = [10, 50, 100, 500] 
# Controls the loss function used for calculating the weight of the base models
param['adaboostclassifier__learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0] 

# Lecture : l'hyperparam gamma appartient à svc (d'où le svc d'abord ou pipeline)

# Variable grid stocking the best parameters for the model
grid = GridSearchCV(AdaBoost, param, scoring='recall', cv=4)

# Training
grid.fit(X_train, y_train)

print("Best params :")
print(grid.best_params_)

# Prediction
y_pred = grid.predict(X_test)

# Report with the recall and precision
print(classification_report(y_test, y_pred))

def evaluation_v3(model):
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='f1',
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    plt.title("AdaBoost with GridSearchCV")
    plt.show()
    
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    
# Evaluation finale des meilleurs paramètres
evaluation_v3(grid.best_estimator_)


#                      Second Test with RandomizedSearchCV

# Hyperparameter Tuning

# Dictionnary welcoming future hyperparameters
param = dict() 
# Number of estimators / weak learners 
param['adaboostclassifier__n_estimators'] = [10, 50, 100, 500] 
# Controls the loss function used for calculating the weight of the base models
param['adaboostclassifier__learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]

param['pipeline__polynomialfeatures__degree']=[2, 3],
param['pipeline__selectkbest__k'] = range(45, 60)

# Lecture : l'hyperparam gamma appartient à svc (d'où le svc d'abord ou pipeline)

# Variable grid stocking the best parameters for the model
rand = RandomizedSearchCV(AdaBoost, param, scoring='recall', cv=4)

# Training
rand.fit(X_train, y_train)

print("Best params :")
print(rand.best_params_)

# Prediction
y_pred = rand.predict(X_test)

# Report with the recall and precision
print(classification_report(y_test, y_pred))

def evaluation_v4(model):
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='f1',
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    plt.title("AdaBoost with RandomizedSearchCV")
    plt.show()
    
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    
# Evaluation finale des meilleurs paramètres
evaluation_v4(grid.best_estimator_)

# The graph is chaotic, but the results of training and scoring converged
# on 0.850. 



#         Another test using another estimator and other parameters

# Using Decision Tree with 2 as max_depth instead of the default 1
# Balanced parameter : Automatically compensate for imbalance
base = DecisionTreeClassifier(max_depth=2, 
                              class_weight='balanced')

# Second Version of Ada Boost with hyperparameters tuning

# Auto n_estimator best etc

AdaBoost2 = make_pipeline(preprocessor, 
                          AdaBoostClassifier(estimator=base,
                                             n_estimators=200, # Number of trees
                                             learning_rate=0.05, # Step size shrinkage
                                             random_state=0
                                            )
                          )


print(AdaBoost2)

# Dictionnary welcoming future hyperparameters
param = dict() 
# Number of estimators / weak learners 
param['adaboostclassifier__n_estimators'] = [10, 50, 100, 500] 
# Controls the loss function used for calculating the weight of the base models
param['adaboostclassifier__learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]

param['pipeline__polynomialfeatures__degree']=[2, 3]
param['pipeline__selectkbest__k'] = list(range(45, 60))

# Lecture : l'hyperparam gamma appartient à svc (d'où le svc d'abord ou pipeline)

# Variable grid stocking the best parameters for the model
rand = RandomizedSearchCV(AdaBoost2, param, scoring='recall', cv=4)

# Training
rand.fit(X_train, y_train)

print("Best params :")
print(rand.best_params_)

# Prediction
y_pred = rand.predict(X_test)

# Report with the recall and precision
print(classification_report(y_test, y_pred))

def evaluation_v5(model):
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='f1',
                                              train_sizes=np.linspace(0.1, 1, 10))
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    plt.title("AdaBoost with RandomizedSearchCV & Balanced param")
    plt.show()
    
    cf_matrix = confusion_matrix(y_test, ypred)
    print(cf_matrix)
    
    # Visualization of the confusion matrix with heatmap
    
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
    plt.title("Confusion Matrix AdaBoost")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()
    
    print(classification_report(y_test, ypred))
    
# Evaluation finale des meilleurs paramètres
evaluation_v5(rand.best_estimator_)

# Results of confusion matrix
# TN : 69, 43.95% | TP : 60, 38.22% | FP : 28, 17.83% | FN : 0, 0.00%

# AdaBoost2 with its great precision, can be used for Early Warning System and
# Tsunami risk prediction

# Since we have a model for two/four objectives, I will use other models for 
# the other statements.

# For the magnitude estimations objective, the new target is magnitude itself.

class TsunamiRisk():
    
    def __init__(self):
        
        self.create_globe_tsunami() # Tsunami
        
        # Un test pour les risques de tsunami
        

class TsunamiRiskPred():
    
    def __init__(self):
        
        self.create_globe_tsunami() # Tsunami
        
        # tests sur des vrais données 
        # (nettoyées au préalable ou que j'aurais trouvé)































