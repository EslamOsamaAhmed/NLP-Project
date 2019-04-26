# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:31:48 2019

@author: Eslam

"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import seaborn as sns
from collections import Counter
import matplotlib as mpl
from sklearn import feature_extraction
from scipy import sparse
from scipy.sparse import csr_matrix
import nltk
import random
from sklearn.metrics import classification_report

datasetname = "DataSets/Dataset-1Charts/dataset-1.csv"

data = pd.read_csv(datasetname,encoding='latin-1')
data.head()

if(datasetname == "DataSets/Dataset-2Charts/dataset-2.csv"):
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

#five Batches to find the Acc of Training set, Testing Set for the ML Model
for x in range (1, 6):
    #Activation Function Sigmoid
    svc = SVC(kernel='sigmoid', gamma=1)
    
    #Train Model to Classify between Ham and Spam Emails by Fit()
    svc.fit(features_train, labels_train)
    prediction = svc.predict(features_test)
    
    #Accuracy of Testing Set
    print("Accuracy Test", accuracy_score(labels_test,prediction))
    
    #Accuracy of Training Set
    print("Accuracy Traning", svc.score(features_train, labels_train))
    
    print("EVALUATION ON TESTING DATA")
    print(classification_report(labels_test, prediction))

#########################################################################

#Save Model traning Result
filename = 'finalized_model_Spam_Ham_SVM.sav'
pickle.dump(svc, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))