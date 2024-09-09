# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:58:48 2024

@author: shash
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score


file_path = 'D:/Submission of BTP Final Python code and its Datasets/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    loaded_df = pickle.load(file)

# Verify that the DataFrame is loaded correctly
print(loaded_df)



# Encode the target variable
loaded_df['tilt/twist'] = loaded_df['tilt/twist'].map({'tilt': 0, 'twist': 1})

# Assign features to X and target to y
X=loaded_df.drop(columns=["file_name","ATOMS_id","HydrostaticStress_GPa","type","tilt/twist"], axis=1) # Features
y = loaded_df['tilt/twist']  # Target variable

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Calculate the min and max values for each column in the training set
train_min = X_train.min()
train_max = X_train.max()

# Function to normalize a dataset based on provided min and max values
def normalize(loaded_df, min_vals, max_vals):
    return (loaded_df - min_vals) / (max_vals - min_vals)

# Apply normalization to the training set
X_train_normalized = normalize(X_train, train_min, train_max)

# Apply the same normalization to the testing set using the training set's min and max values
X_test_normalized = normalize(X_test, train_min, train_max)


# Define the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 500],          # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],          # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],          # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],            # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]                # Whether bootstrap samples are used when building trees
}

# Randomized search with cross-validation
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=10,                # Number of different combinations to try
    cv=3,                     # 3-fold cross-validation
    n_jobs=-1,                # Use all available cores
    verbose=2,                # Detailed logging
    random_state=42,
    scoring='accuracy'        # Use accuracy as the scoring metric
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