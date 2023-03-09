# EDA
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np 
import plotly.express as px

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from datasist.structdata import detect_outliers
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import category_encoders as ce
import re 

# Modelado
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from sklearn.metrics import classification_report 
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Opciones extra
sns.set(rc={'figure.figsize': [14, 7]}, font_scale=1.2) # Standard figure size for all 
np.seterr(divide='ignore', invalid='ignore', over='ignore') ;

import warnings 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")


#Leemos los datos
data = pd.read_csv('G:/Mi unidad/Data Science/repo_clase/MAD_PT_DS_Sep22_Mar23/03-Machine_Learning/Final Project/proyecto_gonzalomartin/src/data/raw_files/train.csv')

#Primero, preprocesamos el Target

m = {"Poor": 0,
     "Standard": 1, 
     "Good": 2
     }
data['Credit_Score'] = data['Credit_Score'].map(m)

#Preprocesamos la col Occupation

data['Occupation'].value_counts()

data = pd.get_dummies(data, columns = ['Occupation'], drop_first=True)

#Arrgelamos la columna Type Of Loan

for i in data['Type_of_Loan'].value_counts().head(9).index[1:] : 
    data[i] = data['Type_of_Loan'].str.contains(i)
    
del data['Type_of_Loan']

#Arrgelamos la columna Credit Mix

data['Credit_Mix'].value_counts()

p = {'Bad': 0,
     'Standard': 1,
     'Good': 2
     }


data['Credit_Mix'] = data['Credit_Mix'].map(p)

#Arrgelamos la columna Payment of Min Amount

data['Payment_of_Min_Amount'].replace("NM", "No", inplace = True)

z = {'No': 0, 
     'Yes': 1
     }

data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].map(z)

#Arrgelamos la columna Payment_Behaviour

data['Payment_Behaviour'].value_counts()

data = pd.get_dummies(data, columns=['Payment_Behaviour'], drop_first=True )

#Data Sampling

X,y = data.drop("Credit_Score", axis = 1), data["Credit_Score"]

#Eliminamos las columnas que a priori no van a servirnos para el modelo

X = X.drop(columns=["ID", "Customer_ID", "Name", "SSN"], axis = 1)

# Convertimos cada columna uint8 a tipo float

uint8_cols = X.select_dtypes(include=['uint8']).columns.tolist()


for col in uint8_cols:
    X[col] = X[col].astype('float')
    

# Convertimos cada columna bool a tipo float

bool_cols = X.select_dtypes(include=['bool']).columns.tolist()


for col in bool_cols:
    X[col] = X[col].astype('float')
    
#Dividimos en train y test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

#Escalamos
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


#Entrenamos el modelo que vamos a utilizar

bagging = BaggingClassifier(n_jobs=-1)
extraTrees = ExtraTreesClassifier(max_depth=10, n_jobs=-1)
randomForest = RandomForestClassifier(n_jobs=-1)
histGradientBoosting = HistGradientBoostingClassifier()
XGB = XGBClassifier(n_jobs=-1)

model = StackingClassifier([
    ('bagging', bagging),
    ('extraTress', extraTrees),
    ('randomforest', randomForest),
    ('histGradientBoosting', histGradientBoosting),
    ('XGB', XGB)
], n_jobs=-1)

model.fit(X, y)

#Lo guardamos

import pickle
with open('modelo_xgb.pickle', 'wb') as f:
    pickle.dump(model, f)
