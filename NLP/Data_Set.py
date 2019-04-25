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
for filename in os.listdir("enron3/enron3/ham/"):  
    file1 = open("enron3/enron3/ham/" + filename,"r+")
    ham.append([file1.read(),"ham"])
    
# Spam File
for filename in os.listdir("enron3/enron3/spam/"):  
    with codecs.open("enron3/enron3/spam/" + filename, 'r', encoding='utf-8',
                 errors='ignore') as fdata:
                    spam.append([fdata.read(),"spams"])

# Merge two Lists
emails = ham + spam

# Shuffl List
random.shuffle(emails) 
    
# Create the pandas DataFrame 
df = pd.DataFrame(emails, columns = ['text', 'label'])

# Convert DF to CSV
df.to_csv (r'C:\Users\Eslam\Desktop\NLP\spam-ham1.csv', index = None, header=True)

