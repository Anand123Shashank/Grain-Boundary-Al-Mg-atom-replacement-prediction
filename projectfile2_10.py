# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:10:01 2024

@author: shash
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

file_path = 'D:/Submission of BTP Final Python code and its Datasets/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    loaded_df = pickle.load(file)
    
df=loaded_df.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1)

# Calculate Pearson correlation
pearson_corr = df.corr(method='pearson')

# Calculate Spearman correlation
spearman_corr = df.corr(method='spearman')

# Plot heatmaps
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation Heatmap')

plt.subplot(1, 2, 2)
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Correlation Heatmap')

plt.tight_layout()
plt.show()
