# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:01:44 2024

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


#Dropping columns ATOMS_id,type,file_name,tilt/twist

table1=loaded_df.drop(columns=["tilt/twist", "file_name","ATOMS_id","type"], axis=1)
labels = loaded_df['tilt/twist']

#Standardize the dataset in the dataframe

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
scaled_data = scaler.fit_transform(table1)

#PCA analysis
pca = PCA(n_components=10)  # Here, we specify 3 components for PCA
principal_components = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(10)])


# Access the principal directions (eigenvectors)
principal_directions = pca.components_

# Print the principal directions
print("Principal Directions (Eigenvectors):")
print(principal_directions)

# Access the eigenvalues (explained variance)
eigenvalues = pca.explained_variance_

# Print the eigenvalues
print("Eigenvalues of the PCA:")
print(eigenvalues)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



# Plot points with different colors based on 'tilt/twist'
pca_df['tilt/twist'] = labels.reset_index(drop=True)

colors = {'tilt': 'blue', 'twist': 'red'}
for label in colors:
    indices = pca_df['tilt/twist'] == label
    ax.scatter(pca_df.loc[indices, 'PC1'], pca_df.loc[indices, 'PC2'], pca_df.loc[indices, 'PC3'], 
               c=colors[label], label=label)
    
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA Projection')
ax.legend()

plt.show()


#Dropping columns ATOMS_id,type,file_name,tilt/twist

X=loaded_df.drop(columns=["tilt/twist", "file_name","ATOMS_id","type"], axis=1)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

Y = loaded_df['tilt/twist']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Perform LDA
lda = LDA(n_components=1)  # Reduce to 2 dimensions for visualization
X_lda = lda.fit_transform(X_scaled, Y)

# Step 4: Separate tilt and twist
tilt_indices = Y == 'tilt'
twist_indices = Y == 'twist'

# Step 5: Plot the results
plt.figure(figsize=(10, 7))
plt.scatter(X_lda[tilt_indices], np.zeros(sum(tilt_indices)), c='red', label='Tilt', s=150, marker='^', alpha=0.5)  # Larger, triangle-shaped markers for Tilt
plt.scatter(X_lda[twist_indices], np.zeros(sum(twist_indices)), c='blue', label='Twist', s=50, marker='o', alpha=0.5)  # Default-sized, circular markers for Twist
plt.title('LDA of Tilt/Twist')
plt.xlabel('LDA Component 1')
plt.legend()
plt.show()











