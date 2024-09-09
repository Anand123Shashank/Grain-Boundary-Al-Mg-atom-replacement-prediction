# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:09:47 2024

@author: shash
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


file_path = 'D:/Submission of BTP Final Python code and its Datasets/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    df = pickle.load(file)

# Define features and target
X=df.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1) # Features
y = df['tilt/twist']
file_names = df['file_name']



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data using the given formula
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

# Normalize train and test features separately
normalized_X_train = normalize(X_train)
normalized_X_test = (X_test - X_train.min()) / (X_train.max() - X_train.min())

# Standardize the normalized data
scaler = StandardScaler()
standardized_X_train = scaler.fit_transform(normalized_X_train)
standardized_X_test = scaler.transform(normalized_X_test)

# Perform PCA on the standardized training data
pca = PCA(n_components=5)  # Get the first 5 principal components
pca.fit(standardized_X_train)

# Get the first 5 eigenvectors (principal components)
first_5_eigenvectors = pca.components_[:5]

# Perform the dot product with all 10 variables in the training dataset
train_result = standardized_X_train @ first_5_eigenvectors.T

# Perform the dot product with all 10 variables in the test dataset
test_result = standardized_X_test @ first_5_eigenvectors.T

# 3D plotting of the first 3 principal components
fig = plt.figure(figsize=(14, 7))

# Training data plot
ax_train = fig.add_subplot(121, projection='3d')
for label in np.unique(y_train):
    indices = y_train == label
    ax_train.scatter(train_result[indices, 0], train_result[indices, 1], train_result[indices, 2], label=label)

ax_train.set_title('3D PCA - Training Data')
ax_train.set_xlabel('PC1')
ax_train.set_ylabel('PC2')
ax_train.set_zlabel('PC3')
ax_train.legend()

# Test data plot
ax_test = fig.add_subplot(122, projection='3d')
for label in np.unique(y_test):
    indices = y_test == label
    ax_test.scatter(test_result[indices, 0], test_result[indices, 1], test_result[indices, 2], label=label)

ax_test.set_title('3D PCA - Test Data')
ax_test.set_xlabel('PC1')
ax_test.set_ylabel('PC2')
ax_test.set_zlabel('PC3')
ax_test.legend()

plt.show()


# Split file_names corresponding to your X_train and X_test
train_file_names = file_names.loc[X_train.index].reset_index(drop=True)
test_file_names = file_names.loc[X_test.index].reset_index(drop=True)

# Create DataFrames for the PCA results
train_df = pd.DataFrame(train_result, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
test_df = pd.DataFrame(test_result, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

# Add the 'tilt/twist' and 'file_name' labels to each DataFrame
train_df['tilt/twist'] = y_train.reset_index(drop=True)
test_df['tilt/twist'] = y_test.reset_index(drop=True)
train_df['file_name'] = train_file_names
test_df['file_name'] = test_file_names

# Combine the DataFrames without distinguishing between train and test data
combined_df = pd.concat([train_df, test_df], ignore_index=True)


# Path to save the pickle file
pickle_file_path = 'combined_df.pkl'


# Saving the combined DataFrame
with open(pickle_file_path, 'wb') as file:
    pickle.dump(combined_df, file)


