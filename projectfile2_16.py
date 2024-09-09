# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:24:23 2024

@author: shash
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score




file_path = 'D:/Submission of BTP Final Python code and its Datasets/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    df = pickle.load(file)
    
    
    
def normalize_column(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

# Prepare the dataset
features = df.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1)  # Remove 'tilt/twist' and 'file_name' columns
target = df['tilt/twist']

# Encode target labels
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)  # 'tilt' -> 0, 'twist' -> 1 (or vice versa)

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

# Normalize the training data
train_min = X_train.min()
train_max = X_train.max()
X_train_normalized = X_train.apply(lambda x: normalize_column(x, train_min[x.name], train_max[x.name]))

# Normalize the test data using training min/max
X_test_normalized = X_test.apply(lambda x: normalize_column(x, train_min[x.name], train_max[x.name]))


# XGBoost model
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 0.1, 0],
}

# Randomized search with cross-validation
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42,
    scoring='accuracy'
)

# Fit the model
random_search.fit(X_train_normalized, y_train)

# Best parameters
best_params = random_search.best_params_
print("Best parameters found: ", best_params)

# Predictions
y_pred = random_search.predict(X_test_normalized)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
