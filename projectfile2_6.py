# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:59:51 2024

@author: shash
"""


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle


file_path = 'D:/Grain Boundary Al-Mg atom replacement prediction/Combined_table_list/combined_table_list.pkl'

# Load the DataFrame from the pickle file
with open(file_path, 'rb') as file:
    loaded_df = pickle.load(file)



# Encode the target variable
label_encoder=LabelEncoder()
loaded_df['file_name']=label_encoder.fit_transform(loaded_df['file_name'])

# Assign features to X and target to y
X=loaded_df.drop(columns=["file_name","ATOMS_id","type","tilt/twist"], axis=1) # Features
y = loaded_df['file_name']  # Target variable

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

# Step 4: Apply Kernel SVM
# Initialize SVM with RBF kernel
svm_model = SVC(kernel='poly', C=2, degree=3, gamma='scale')  # You can adjust 'C' and 'gamma' as needed
svm_model.fit(X_train_normalized, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test_normalized)


accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Testing Data')
plt.show()