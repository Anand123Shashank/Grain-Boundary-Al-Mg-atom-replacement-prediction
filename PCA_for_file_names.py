# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 08:13:04 2024

@author: shash
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import numpy as np


file_path = 'D:/Grain Boundary Al-Mg atom replacement prediction/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    df = pickle.load(file)



# Encode the target variable
label_encoder=LabelEncoder()
df['file_name']=label_encoder.fit_transform(df['file_name'])

# Assign features to X and target to y
X= df.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1) # Features
y = df['file_name']  # Target variable

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Calculate the min and max values for each column in the training set
train_min = X_train.min()
train_max = X_train.max()

# Function to normalize a dataset based on provided min and max values
def normalize(df, min_vals, max_vals):
    return (df - min_vals) / (max_vals - min_vals)

# Apply normalization to the training set
X_train_normalized = normalize(X_train, train_min, train_max)

# Apply the same normalization to the testing set using the training set's min and max values
X_test_normalized = normalize(X_test, train_min, train_max)

# Step 4: Perform PCA for 2D Visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_normalized)

# 2D scatter plot for the training data
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train.astype('category').cat.codes, cmap='viridis', edgecolor='k', s=50)

# Customizing the color bar to show all 18 labels
plt.colorbar(scatter, ticks=np.arange(18))
plt.title('2D PCA Scatter Plot of Training Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# Ensure axis ticks for each label are displayed
plt.xticks(np.arange(min(X_train_pca[:, 0]), max(X_train_pca[:, 0]), (max(X_train_pca[:, 0]) - min(X_train_pca[:, 0])) / 17))  # Adjust range and steps
plt.yticks(np.arange(min(X_train_pca[:, 1]), max(X_train_pca[:, 1]), (max(X_train_pca[:, 1]) - min(X_train_pca[:, 1])) / 17))  # Adjust range and steps

plt.show()

print(df['file_name'].nunique())  # Check how many unique file names are present
print(y_train.nunique())  # Check unique file names in training set
print(y_test.nunique())   # Check unique file names in test set