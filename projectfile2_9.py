# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:50:25 2024

@author: shash
"""


import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the combined SVM model
with open('svm_model_combined.pkl', 'rb') as file:
    combined_model_data = pickle.load(file)
    combined_svm_model = combined_model_data['model']
    combined_min_vals = combined_model_data['min_vals']
    combined_max_vals = combined_model_data['max_vals']

# Load the tilt SVM model
with open('svm_model_tilt.pkl', 'rb') as file:
    tilt_model_data = pickle.load(file)
    tilt_svm_model = tilt_model_data['model']
    tilt_min_vals = tilt_model_data['min_vals']
    tilt_max_vals = tilt_model_data['max_vals']

# Load the twist SVM model
with open('svm_model_twist.pkl', 'rb') as file:
    twist_model_data = pickle.load(file)
    twist_svm_model = twist_model_data['model']
    twist_min_vals = twist_model_data['min_vals']
    twist_max_vals = twist_model_data['max_vals']
    



file_path = 'D:/Submission of BTP Final Python code and its Datasets/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    loaded_df = pickle.load(file)
    
    

# Encode 'tilt/twist' and 'file_name' columns
label_encoder = LabelEncoder()
loaded_df['tilt/twist'] = label_encoder.fit_transform(loaded_df['tilt/twist'])
loaded_df['file_name'] = label_encoder.fit_transform(loaded_df['file_name'])

# Function to normalize the data with given min and max values
def normalize_with_params(loaded_df, min_vals, max_vals):
    return (loaded_df - min_vals) / (max_vals - min_vals)

# Normalize the test data using the combined model parameters
X_test_combined = loaded_df.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1)
X_test_combined_normalized = normalize_with_params(X_test_combined, combined_min_vals, combined_max_vals)

# Predict 'tilt/twist' using the combined model
tilt_twist_predictions = combined_svm_model.predict(X_test_combined_normalized)

# Initialize lists to store the actual and predicted 'file_name'
actual_file_name = []
predicted_file_name = []

# Loop through the predictions to use the appropriate model for 'file_name' classification
for i, prediction in enumerate(tilt_twist_predictions):
    if prediction == 0:  # Tilt
        X_test_tilt_normalized = normalize_with_params(X_test_combined.iloc[i:i+1], tilt_min_vals, tilt_max_vals)
        file_name_prediction = tilt_svm_model.predict(X_test_tilt_normalized)
    else:  # Twist
        X_test_twist_normalized = normalize_with_params(X_test_combined.iloc[i:i+1], twist_min_vals, twist_max_vals)
        file_name_prediction = twist_svm_model.predict(X_test_twist_normalized)
    
    actual_file_name.append(loaded_df.iloc[i]['file_name'])
    predicted_file_name.append(file_name_prediction[0])

# Calculate accuracy
accuracy = accuracy_score(actual_file_name, predicted_file_name)
print(f"Accuracy of the final model: {accuracy}")

# Print confusion matrix
conf_matrix = confusion_matrix(actual_file_name, predicted_file_name)
print("Confusion Matrix for the final model:")
print(conf_matrix)
