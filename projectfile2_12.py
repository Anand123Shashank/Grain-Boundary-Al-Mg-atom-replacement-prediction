# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:02:27 2024

@author: shash
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



file_path = 'D:/Submission of BTP Final Python code and its Datasets/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    loaded_df = pickle.load(file)
    

# Remove 'file_name' and 'tilt/twist' and other columns
features = loaded_df.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1)


# Split data into training and test sets (80-20 split)
train_features, test_features = train_test_split(features, test_size=0.2, random_state=42)


def normalize(X):
    X_normalized = X.copy()
    for column in X.columns:
        min_val = X[column].min()
        max_val = X[column].max()
        X_normalized[column] = (X[column] - min_val) / (max_val - min_val)
    return X_normalized

def standardize(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return X_standardized, scaler

# Normalize the training data using custom formula
normalized_train_features = normalize(train_features)

# Standardize the normalized training data
standardized_train_features, scaler = standardize(normalized_train_features)

# Standardize the normalized test data using the same scaler fitted on training data
normalized_test_features = normalize(test_features)
standardized_test_features = scaler.transform(normalized_test_features)

# Perform PCA on the standardized training data
pca = PCA()
pca.fit(standardized_train_features)

# Eigenvalues (explained variance)
eigenvalues = pca.explained_variance_

# Normalize eigenvalues to get the percentage of variance explained
explained_variance_ratio = eigenvalues / eigenvalues.sum() * 100

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Eigenvectors (principal components)
eigenvectors = pca.components_

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='b')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.grid()
plt.show()

# Print eigenvalues, explained variance ratio, and eigenvectors
print("Eigenvalues:\n", eigenvalues)
print("Explained Variance Ratio (%):\n", explained_variance_ratio)
print("Cumulative Explained Variance (%):\n", cumulative_explained_variance)
print("Eigenvectors (Principal Components):\n", eigenvectors)

# Select the first 5 eigenvectors
first_5_eigenvectors = eigenvectors[:5]

# Perform the dot product with all 10 variables in the training data
train_result = standardized_train_features @ first_5_eigenvectors.T

# Create a DataFrame with the training result
train_result_df = pd.DataFrame(train_result, columns=[f'PC{i+1}' for i in range(5)])


print("\nTransformed Training Data (First 5 rows):")
print(train_result_df.head())


# Perform the dot product with all 10 variables in the test data
test_result = standardized_test_features @ first_5_eigenvectors.T

# Create a DataFrame with the test result
test_result_df = pd.DataFrame(test_result, columns=[f'PC{i+1}' for i in range(5)])

print("\nTransformed Test Data (First 5 rows):")
print(test_result_df.head())



train_df = pd.DataFrame(train_result, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
test_df = pd.DataFrame(test_result, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

# Step 2: Add the 'tilt/twist' and 'file_name' labels to each DataFrame
train_df['tilt/twist'] = y_train.reset_index(drop=True)
test_df['tilt/twist'] = y_test.reset_index(drop=True)
train_df['file_name'] = train_file_names.reset_index(drop=True)
test_df['file_name'] = test_file_names.reset_index(drop=True)

# Step 3: Combine the DataFrames without distinguishing between train and test data
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Save the train_result and test_result using pickle
with open('train_result.pkl', 'wb') as train_file:
    pickle.dump(train_result, train_file)

with open('test_result.pkl', 'wb') as test_file:
    pickle.dump(test_result, test_file)


