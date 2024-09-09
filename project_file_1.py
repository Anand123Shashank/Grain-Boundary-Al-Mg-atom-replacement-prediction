# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:42:53 2024

@author: shash
"""

import pandas as pd
file_path=('D:/Submission of BTP Final Python code and its Datasets/Bi-Crystal/Bi-Crystal/CSL GBs Al_Al/tilt/input_G2-2-3_[110]_csl17')

df=pd.read_table(file_path,delimiter = " ")
df['tilt/twist']='tilt'
df['file_name']='input_G2-2-3_[110]_csl17'

print(df)

file_path1=('D:/Submission of BTP Final Python code and its Datasets/Bi-Crystal/Bi-Crystal/CSL GBs Al_Al/tilt/input_G-112_[110]_csl3')

df1=pd.read_table(file_path1,delimiter = " ")
df1['tilt/twist']='tilt'
df1['file_name']='input_G-112_[110]_csl3'

print(df1)

file_path2=('D:/Submission of BTP Final Python code and its Datasets/Bi-Crystal/Bi-Crystal/CSL GBs Al_Al/tilt/input_G-2-10_[001]_csl5')

df2=pd.read_table(file_path2,delimiter=" ")
df2['tilt/twist']='tilt'
df2['file_name']='input_G-2-10_[001]_csl5'

file_path3=('D:/Submission of BTP Final Python code and its Datasets/Bi-Crystal/Bi-Crystal/CSL GBs Al_Al/tilt/input_G140_[001]_csl17')

df3=pd.read_table(file_path3,delimiter = " ")
df3['tilt/twist']='tilt'
df3['file_name']='input_G140_[001]_csl17'

file_path4=('D:/Submission of BTP Final Python code and its Datasets/Bi-Crystal/Bi-Crystal/CSL GBs Al_Al/tilt/input_G250_[001]_csl29')

df4=pd.read_table(file_path4,delimiter = " ")
df4['tilt/twist']='tilt'
df4['file_name']='input_G250_[001]_csl29'

file_path5=('D:/Submission of BTP Final Python code and its Datasets/Bi-Crystal/Bi-Crystal/CSL GBs Al_Al/tilt/input_G-12-1_[111]_csl3')

df5=pd.read_table(file_path5,delimiter = " ")
df5['tilt/twist']='tilt'
df5['file_name']='input_G-12-1_[111]_csl3'

file_path6=('D:/Submission of BTP Final Python code and its Datasets/Bi-Crystal/Bi-Crystal/CSL GBs Al_Al/tilt/input_G-47-3_[111]_csl37')

df6=pd.read_table(file_path6,delimiter = " ")
df6['tilt/twist']='tilt'
df6['file_name']='input_G-47-3_[111]_csl37'

file_path7=('D:/Submission of BTP Final Python code and its Datasets/Bi-Crystal/Bi-Crystal/CSL GBs Al_Al/tilt/input_G-312_[111]_csl7')

df7=pd.read_table(file_path7,delimiter = " ")
df7['tilt/twist']='tilt'
df7['file_name']='input_G-312_[111]_csl7'



# Import Module 
import os 

# Folder Path 
path = "D:\Submission of BTP Final Python code and its Datasets\Bi-Crystal\Bi-Crystal\CSL GBs Al_Al\tilt"

# Change the directory 
os.chdir(path) 

# Read text File 


def read_text_file(file_path): 
	with open(file_path, 'r') as f: 
		print(f.read()) 


# iterate through all file 
for file in os.listdir(): 
	# Check whether file is in text format or not 

		file_path = f"{path}\{file}"

		# call read text file function 
		read_text_file(file_path) 


import glob
import os

os.chdir(r"")

my_files = glob.glob("*.txt")

print(my_files)















