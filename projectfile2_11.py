# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:53:52 2024

@author: shash
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pickle

file_path = 'D:/Submission of BTP Final Python code and its Datasets/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    loaded_df = pickle.load(file)

# Remove 'file_name' and 'tilt/twist' columns
features = loaded_df.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1)
target = loaded_df['tilt/twist']

def normalize_train_test(X_train, X_test):
    X_train_normalized = X_train.copy()
    X_test_normalized = X_test.copy()
    
    for column in X_train.columns:
        min_val = X_train[column].min()
        max_val = X_train[column].max()
        
        X_train_normalized[column] = (X_train[column] - min_val) / (max_val - min_val)
        X_test_normalized[column] = (X_test[column] - min_val) / (max_val - min_val)
    
    return X_train_normalized, X_test_normalized

def remove_variables_and_evaluate(features, target, variables_to_remove):
    # Remove specified variables
    features_filtered = features.drop(columns=variables_to_remove)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_filtered, target, test_size=0.25, random_state=42)
    
    # Normalize the data using custom formula
    X_train_normalized, X_test_normalized = normalize_train_test(X_train, X_test)
    
    # Train SVM model
    svm_model = SVC(C=1000, kernel='linear', random_state=42)
    svm_model.fit(X_train_normalized, y_train)
    
    # Evaluate the model
    y_pred = svm_model.predict(X_test_normalized)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy after removing {variables_to_remove}: {accuracy}")
    print(f"Confusion Matrix after removing {variables_to_remove}:")
    print(conf_matrix)
    
    return accuracy, conf_matrix

# List of variables to test removal
variables = list(features.columns)

# Example: Remove 'vonMisesDeviotricStress_GPa' and 'HydrostaticStress_GPa' and evaluate
remove_variables_and_evaluate(features, target, ['HydrostaticStress_GPa', 'vonMisesDeviotricStress_GPa'])

# Optionally, loop through all combinations of variables and evaluate
from itertools import combinations

results = {}
for r in range(1, len(variables) + 1):
    for combo in combinations(variables, r):
        accuracy, conf_matrix = remove_variables_and_evaluate(features, target, list(combo))
        results[combo] = {'accuracy': accuracy, 'conf_matrix': conf_matrix}

# Print results
print("Results after removing each combination of variables:")
for combo, result in results.items():
    print(f"{combo}: Accuracy = {result['accuracy']}")
    print(f"Confusion Matrix:\n{result['conf_matrix']}\n")