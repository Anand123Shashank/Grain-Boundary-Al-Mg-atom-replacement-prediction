# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:20:00 2024

@author: shash
"""

import pandas as pd
import pickle

file_path = 'D:/Submission of BTP Final Python code and its Datasets/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    loaded_df = pickle.load(file)

# Verify that the DataFrame is loaded correctly
print(loaded_df)

numFeatures = 10

#Dropping columns ATOMS_id,type,file_name,tilt/twist

df=loaded_df.drop(columns=["file_name","ATOMS_id","type"], axis=1)


#Splitting train and test dataset and normalize it
from sklearn.model_selection import train_test_split


# Split the dataset into 75% training and 25% testing
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

# Normalize the train dataset
train_min = train_df.iloc[:,:numFeatures].min()
train_max = train_df.iloc[:,:numFeatures].max()

# Function to normalize a dataset based on min and max values
def normalize(df, min_vals, max_vals):
    df_norm= (df.iloc[:,:numFeatures] - min_vals) / (max_vals - min_vals)
    df_norm['tilt/twist']=df['tilt/twist']
    return df_norm

# Apply normalization to the training set
normalized_train_df = normalize(train_df, train_min, train_max)

# Apply the same normalization to the testing set using min and max from training set
normalized_test_df = normalize(test_df, train_min, train_max)

# Check the results
print("Normalized Training Data:")
print(normalized_train_df.head())

print("\nNormalized Testing Data:")
print(normalized_test_df.head())














