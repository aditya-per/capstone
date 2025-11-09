#Creating the file to define,build and tune model
# for data manipulation
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow


api = HfApi()
#Define the location of train and test data from the Hugging Face data space
Xtrain_path = "hf://datasets/adityasharma0511/Predictive_Maintenance/Xtrain.csv"  
Xtest_path = "hf://datasets/adityasharma0511/Predictive_Maintenance/Xtest.csv"
ytrain_path = "hf://datasets/adityasharma0511/Predictive_Maintenance/ytrain.csv"
ytest_path = "hf://datasets/adityasharma0511/Predictive_Maintenance/ytest.csv"

#Read the data into relevant datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)
