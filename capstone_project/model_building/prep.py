# Load the dataset from the Hugging Face data space and perform data cleaning
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define constants for the dataset which is located at huggingface
DATASET_PATH = "hf://datasets/adityasharma0511/Predictive_Maintenance/engine_data.csv"
data = pd.read_csv(DATASET_PATH)  #Read the datasset
print("Dataset loaded successfully.") 


# Define predictor matrix (X) using selected numeric and categorical features
X = data.drop(["Engine Condition"], axis=1)  #Capture all the data except the Target

# Define target variable
y = data["Engine Condition"]


# Split the cleaned dataset into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

#Checking the dataset balance t find if there is imbalance in the train and test set distribution.

print("Shape of Training set : ", Xtrain.shape)
print("Shape of test set : ", Xtest.shape)
print("Percentage of classes in training set:")
print(ytrain.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(ytest.value_counts(normalize=True))


# Save training and testing datasets locally.
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


# Upload the resulting train and test datasets back to the Hugging Face data space.
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="adityasharma0511/Predictive_Maintenance",
        repo_type="dataset",
    )
