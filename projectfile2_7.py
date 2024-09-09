# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:07:13 2024

@author: shash
"""
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


file_path = 'D:/Submission of BTP Final Python code and its Datasets/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    loaded_df = pickle.load(file)

# Verify that the DataFrame is loaded correctly
print(loaded_df)



# Function to normalize a dataset based on provided min and max values
def normalize(loaded_df, min_vals, max_vals):
    return (loaded_df - min_vals) / (max_vals - min_vals)

# Function to train and evaluate SVM with a given value of C
def evaluate_svm(C, X_train, y_train, X_test, y_test):
    svm = SVC(C=C, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

# Hyperparameter tuning for C
C_values = [1000]

# Process for each subset: 'tilt' and 'twist'
results = {}
for label in ['tilt', 'twist']:
    subset = loaded_df[loaded_df['tilt/twist'] == label]
    
    X = subset.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1)  # Features
    y = subset['file_name']  # Target variable
    
    # Encode the target variable 'file_name'
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Split the dataset into 75% training and 25% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Calculate the min and max values for each column in the training set
    train_min = X_train.min()
    train_max = X_train.max()
    
    # Apply normalization to the training set
    X_train_normalized = normalize(X_train, train_min, train_max)
    
    # Apply the same normalization to the testing set using the training set's min and max values
    X_test_normalized = normalize(X_test, train_min, train_max)
    
    # Perform SVM and store results
    accuracies = []
    y_preds = []

    for C in C_values:
        accuracy, y_pred = evaluate_svm(C, X_train_normalized, y_train, X_test_normalized, y_test)
        accuracies.append(accuracy)
        y_preds.append(y_pred)
        print(f"Accuracy for C={C} on '{label}' subset: {accuracy}")
    
    # Find the index of the best accuracy
    best_index = accuracies.index(max(accuracies))
    best_C = C_values[best_index]
    best_accuracy = accuracies[best_index]
    best_y_pred = y_preds[best_index]
    

    print(f"\nBest C value for '{label}' subset: {best_C} with accuracy: {best_accuracy}")

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, best_y_pred)
    print(f"Confusion Matrix for '{label}' subset:")
    print(conf_matrix)
    
    # Create a DataFrame for the confusion matrix with appropriate labels
    conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(f"\nConfusion Matrix Table for '{label}' subset:")
    print(conf_matrix_df)
    
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {label} Subset')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    results[label] = {
        'best_C': best_C,
        'best_accuracy': best_accuracy,
        'conf_matrix': conf_matrix,
        'conf_matrix_df': conf_matrix_df
    }
    
    

# Print accuracy array for both subsets
print("Accuracies for each C value on 'tilt' subset:")
print(accuracies)

print("Accuracies for each C value on 'twist' subset:")
print(accuracies)



