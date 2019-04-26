# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:29:06 2019

@author: eslam
"""
# importing os module 
import os
import pandas as pd 
import codecs
import random


ham = [[]]
spam = [[]]

emails = [[]]

# Ham File
for filename in os.listdir("enron4/ham/"):  
    file1 = open("enron4/ham/" + filename,"rb")
    ham.append([1,file1.read()])
    
# Spam File
for filename in os.listdir("enron4/spam/"):  
    with codecs.open("enron4/spam/" + filename, 'rb',
                 errors='ignore') as fdata:
                    spam.append([0,fdata.read()])

# Merge two Lists
emails = ham + spam

# Shuffl List
random.shuffle(emails) 
    
# Create the pandas DataFrame 
df = pd.DataFrame(emails, columns = ['v1', 'v2'])

# Convert DF to CSV
df.to_csv (r'DataSets/spam-ham.csv', index = None, header=True)

