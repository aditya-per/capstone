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
