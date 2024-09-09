# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:23:52 2024

@author: shash
"""



import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


file_path = 'D:/Submission of BTP Final Python code and its Datasets/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    loaded_df = pickle.load(file)

# Verify that the DataFrame is loaded correctly
print(loaded_df)



# Encode the target variable
loaded_df['tilt/twist'] = loaded_df['tilt/twist'].map({'tilt': 0, 'twist': 1})

# Assign features to X and target to y
X=loaded_df.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1) # Features
y = loaded_df['tilt/twist']  # Target variable

# Split the dataset into 75% training and 25% testing
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

# Function to train and evaluate SVM with a given value of C
def evaluate_svm(C, X_train, y_train, X_test, y_test):
    svm = SVC(C=C, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

# Hyperparameter tuning for C
C_values = [1000]
accuracies = []
y_preds = []

for C in C_values:
    accuracy, y_pred = evaluate_svm(C, X_train_normalized, y_train, X_test_normalized, y_test)
    accuracies.append(accuracy)
    y_preds.append(y_pred)
    print(f"Accuracy for C={C}: {accuracy}")

# Find the index of the best accuracy
best_index = accuracies.index(max(accuracies))
best_C = C_values[best_index]
best_accuracy = accuracies[best_index]
best_y_pred = y_preds[best_index]

print(f"\nBest C value: {best_C} with accuracy: {best_accuracy}")



# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, best_y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Create a DataFrame for the confusion matrix
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Tilt', 'Actual Twist'], columns=['Predicted Tilt', 'Predicted Twist'])

print("\nConfusion Matrix Table:")
print(conf_matrix_df)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

