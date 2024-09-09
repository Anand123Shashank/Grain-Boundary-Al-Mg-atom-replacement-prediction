# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:13:57 2024

@author: shash
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle



file_path = 'D:/Submission of BTP Final Python code and its Datasets/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    loaded_df = pickle.load(file)



# Encode 'tilt/twist' and 'file_name' columns
label_encoder = LabelEncoder()
loaded_df['tilt/twist'] = label_encoder.fit_transform(loaded_df['tilt/twist'])
loaded_df['file_name'] = label_encoder.fit_transform(loaded_df['file_name'])

# Function to normalize the data
def normalize(loaded_df, train=True, min_vals=None, max_vals=None):
    if train:
        min_vals = loaded_df.min()
        max_vals = loaded_df.max()
    df_normalized = (loaded_df - min_vals) / (max_vals - min_vals)
    return df_normalized, min_vals, max_vals

# Train SVM model for combined 'tilt/twist' classification
X_combined = loaded_df.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1)
y_combined = loaded_df['tilt/twist']

X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.20, random_state=42)

X_train_combined_normalized, min_vals_combined, max_vals_combined = normalize(X_train_combined)
X_test_combined_normalized, _, _ = normalize(X_test_combined, train=False, min_vals=min_vals_combined, max_vals=max_vals_combined)

svm_model_combined = SVC(C=1000)
svm_model_combined.fit(X_train_combined_normalized, y_train_combined)

# Save the combined SVM model and normalization parameters
with open('svm_model_combined.pkl', 'wb') as file:
    pickle.dump({'model': svm_model_combined, 'min_vals': min_vals_combined, 'max_vals': max_vals_combined}, file)

print("Combined SVM model saved successfully.")


# Train SVM model for 'file_name' classification on 'tilt' subset
tilt_subset = loaded_df[loaded_df['tilt/twist'] == 0]
X_tilt = tilt_subset.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1)
y_tilt = tilt_subset['file_name']

X_train_tilt, X_test_tilt, y_train_tilt, y_test_tilt = train_test_split(X_tilt, y_tilt, test_size=0.20, random_state=42)

X_train_tilt_normalized, min_vals_tilt, max_vals_tilt = normalize(X_train_tilt)
X_test_tilt_normalized, _, _ = normalize(X_test_tilt, train=False, min_vals=min_vals_tilt, max_vals=max_vals_tilt)

svm_model_tilt = SVC(C=1000)
svm_model_tilt.fit(X_train_tilt_normalized, y_train_tilt)

# Save the tilt SVM model and normalization parameters
with open('svm_model_tilt.pkl', 'wb') as file:
    pickle.dump({'model': svm_model_tilt, 'min_vals': min_vals_tilt, 'max_vals': max_vals_tilt}, file)

print("Tilt SVM model saved successfully.")

# Train SVM model for 'file_name' classification on 'twist' subset
twist_subset = loaded_df[loaded_df['tilt/twist'] == 1]
X_twist = twist_subset.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1)
y_twist = twist_subset['file_name']

X_train_twist, X_test_twist, y_train_twist, y_test_twist = train_test_split(X_twist, y_twist, test_size=0.20, random_state=42)

X_train_twist_normalized, min_vals_twist, max_vals_twist = normalize(X_train_twist)
X_test_twist_normalized, _, _ = normalize(X_test_twist, train=False, min_vals=min_vals_twist, max_vals=max_vals_twist)

svm_model_twist = SVC(C=1000)
svm_model_twist.fit(X_train_twist_normalized, y_train_twist)

# Save the twist SVM model and normalization parameters
with open('svm_model_twist.pkl', 'wb') as file:
    pickle.dump({'model': svm_model_twist, 'min_vals': min_vals_twist, 'max_vals': max_vals_twist}, file)

print("Twist SVM model saved successfully.")








