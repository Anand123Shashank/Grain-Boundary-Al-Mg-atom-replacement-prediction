# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:04:51 2024

@author: shash
"""

#for tilt

import pandas as pd
import glob
import os

path=os.chdir(r"D:\Submission of BTP Final Python code and its Datasets\Bi-Crystal\Bi-Crystal\CSL GBs Al_Al\tilt")

my_files = glob.glob("*")



dataframes = []

# Iterate over the text files and create a dataframe for each one
for text_file in my_files:
    # Read the text file into a dataframe
    df = pd.read_table(text_file,delimiter=" ")
    df['file_name']=text_file.split("/")[-1]
    df['tilt/twist']='tilt'                                
    
    # Append the dataframe to the list of dataframes
    dataframes.append(df)
    
    
# Concatenate the dataframes into a single dataframe
df = pd.concat(dataframes)

# Print the dataframe
print(df)


#for twist

import pandas as pd
import glob
import os

path2=os.chdir(r"D:\Submission of BTP Final Python code and its Datasets\Bi-Crystal\Bi-Crystal\CSL GBs Al_Al\twist")

my_files2 = glob.glob("*")


dataframes2 = []

# Iterate over the text files and create a dataframe for each one
for text_file2 in my_files2:
    # Read the text file into a dataframe
    df2 = pd.read_table(text_file2,delimiter=" ")
    df2['file_name']=text_file2.split("/")[-1]
    df2['tilt/twist']='twist'                                
    
    # Append the dataframe to the list of dataframes
    dataframes2.append(df2)
    
    
# Concatenate the dataframes into a single dataframe
df2 = pd.concat(dataframes2)

# Print the dataframe
print(df2)


# Combined_table list for tilt and twist

combined_table=pd.concat([df,df2])


# Print combined_table dataframe
print(combined_table)


import pickle

folder_path = 'D:\Submission of BTP Final Python code and its Datasets\Combined_table_list'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    

file_path = os.path.join(folder_path, 'combined_table_list.pkl')
with open(file_path, 'wb') as file:
    pickle.dump(combined_table, file)






























