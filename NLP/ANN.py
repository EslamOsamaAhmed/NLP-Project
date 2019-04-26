# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:31:48 2019

@author: Heba

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
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam 
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras import backend as k
import scipy.io as io 
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

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = int(features.shape[0]), init = 'uniform', activation = 'relu', input_dim = features.shape[1]))

# Adding the second hidden layer
classifier.add(Dense(output_dim = int(features.shape[0] / 2), init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(features_train, labels_train, batch_size = 5000, nb_epoch = 2)

classifier.summary()
classifier.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])

score = classifier.evaluate(features_test, labels_test, verbose = 0)

# Predicting the Test set results
y_pred = classifier.predict(features_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, y_pred)

print("EVALUATION ON TESTING DATA")
print(classification_report(labels_test, y_pred))


#########################################################################

#Save Model traning Result
filename = 'finalized_model_Spam_Ham_ANN.sav'
pickle.dump(classifier, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
