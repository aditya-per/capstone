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

#Using ngroc link from previous step to send stats to MLflow running on local machine
mlflow.set_tracking_uri("https://endopoditic-miller-unveraciously.ngrok-free.dev")
mlflow.set_experiment("mlops-training-experiment3")

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

# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define base XGBoost model
xgb_model =xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 75, 100, 125,150],    # number of tree to build
    'max_depth': [2, 3, 4],    # maximum depth of each tree
    'colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]
        
        #Following lines can be used to log all the parameter combination but if we run it at once we'll hit the ngrok limit.
        #If we add delay, it takes many hours to finish
        ## Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)
            # Adding 1 second sleep as ngroc has a rate limit of 120 calls per miniute
            time.sleep(1) 

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Using a low classification threshold to get better recall score
    classification_threshold = 0.425

    #For best model, Calculating the Model prediction for train data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    #For best model, Calculating the Model prediction for test data
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    #Making classification report 
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    
    # Save best model
    joblib.dump(best_model, "best_predict_model.joblib")

    # Log the model artifact
    mlflow.log_artifact("best_predict_model.joblib", artifact_path="model")
    print(f"Model saved as artifact at: best_predict_model.joblib")

    # Upload to Hugging Face
    repo_id = "adityasharma0511/predictive-maintenance-model"
    repo_type = "model"

    api = HfApi(token=os.getenv("HF_TOKEN"))

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_predict_model.joblib",
        path_in_repo="best_predict_model.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
