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
