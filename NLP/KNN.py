# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:31:48 2019

@author: Eman

"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import seaborn as sns
from collections import Counter
import matplotlib as mpl
from sklearn import feature_extraction
from sklearn.model_selection import cross_validate
from scipy import sparse
from scipy.sparse import csr_matrix
import nltk
import random
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

datasetname = "DataSets/dataset-1.csv"

data = pd.read_csv(datasetname,encoding='latin-1')
data.head()

if(datasetname == "DataSets/dataset-2.csv"):
   data=data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

data=data.rename(columns={"v1":"class", "v2":"text"})
data = data.dropna(subset=['class'])
data = data.dropna(subset=['text'])
data.head()
data['length']=data['text'].apply(len)
data.head()

def pre_process(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i)) + " "
    return words


#Prprocessing for Data to find the best Acc and Vectorizing the data to be able to be learned
textFeatures = data['text'].copy()
textFeatures = textFeatures.apply(pre_process)
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(textFeatures)

#Split Data after Preprocessing to Training set & Testing set with it's Labels
features_train, features_test, labels_train, labels_test = train_test_split(features, data['class'], test_size=0.3, random_state=111)

############ Intializing Variables ############
k = 1 # K nearest neighbors
fold = 5 # Number of folds
k_scores = {}

# Creating Knn classifier with k nearest neighbors
knn = KNeighborsClassifier(n_neighbors = k)

# Fit training data to classifier
knn.fit(features_train, labels_train)

while k != 5:
    score = cross_validate(knn,features_train,labels_train, cv=fold)
    k_scores[k] = np.mean(score['test_score'])
    k += 1
    knn.n_neighbors = k

MaxScore = max(k_scores, key=k_scores.get)
print("The Maximum Score on the training set is : " + str(k_scores[MaxScore]) + ", The Best K value is: " + str(MaxScore))

# Test set Prediction
test_prediction = knn.predict(features_test)

print("The Predicted output: " + str(test_prediction))
print("The Real output: " + str(labels_test))

final_score = accuracy_score(labels_test, test_prediction)

print("The Accuracy of prediction: " + str(final_score))

print("EVALUATION ON TESTING DATA")
print(classification_report(labels_test, test_prediction))

#Save Model traning Result
filename = 'finalized_model_Spam_Ham_KNN.sav'
pickle.dump(knn, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

#def pre_process(text):
#    
#    text = text.translate(str.maketrans('', '', string.punctuation))
#    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
#    words = ""
#    for i in text:
#            stemmer = SnowballStemmer("english")
#            words += (stemmer.stem(i))+" "
#    return words
#
## GUI Window to Enter Email to Detect it
#from tkinter import *
#root=Tk()
#root.resizable(width=False, height=False)
#root.geometry("200x130")
#
#Label1 = Label(root, text = "Enter Email") 
#EmailField = Entry(root,text = "Email...", width = 100)
#DetectButton = Button(root, text = "Detect")
#
#def leftClick(event):
#    Email = "Your free ringtone is waiting to be collected. Simply text the password \MIX\" to 85069 to verify. Get Usher and Britney. FML	 PO Box 5249 MK17 92H. 450Ppw 16"	
#    
#    #Prprocessing for Data to find the best Acc and Vectorizing the data to be able to be learned
#    textFeatures = Email
#    textFeatures = pre_process(textFeatures)
#    vectorizer = TfidfVectorizer("english")
#    x = [textFeatures]
#    NEmailfeatures = vectorizer.fit_transform(x)
#    
#    NEmailfeatures = NEmailfeatures.toarray()
#
#    row = []    
#    col = []
#    data = []
#    (fd,sd) = NEmailfeatures.shape
#    
#    for x in range(sd):
#        row.append(0)
#        
#    for y in range(sd,8037):
#        row.append(1)
#    
#    for z in range(8037):
#        col.append(z)
#        
#    for f in range(sd):
#        data.append(NEmailfeatures[0][f])
#        
#    for f in range(sd,8037):
#        data.append(2)
#        
#    EmailEntry = csr_matrix((data, (row, col)), shape=(2, 8037))
#    
#    Nprediction = loaded_model.predict(EmailEntry)
#    print(Nprediction[0])
#
#Label1.pack()
#EmailField.pack(ipady=30)
#DetectButton.pack()
#DetectButton.bind("<Button-1>", leftClick)
#
#root.mainloop()