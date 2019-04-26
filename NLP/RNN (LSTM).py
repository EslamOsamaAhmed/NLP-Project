# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 23:28:36 2019

@author: eslam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report


datasetname = "DataSets/Dataset-2Charts/dataset-2.csv"

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

X_train,X_test,Y_train,Y_test = train_test_split(data["text"],data["class"],test_size=0.15)
max_words = 1500
max_len = 200
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

inputs = Input(name='inputs',shape=[max_len])
layer = Embedding(max_words,50,input_length=max_len)(inputs)
layer = LSTM(64)(layer)
layer = Dense(256,name='FC1')(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(1,name='out_layer')(layer)
layer = Activation('sigmoid')(layer)
model = Model(inputs=inputs,outputs=layer)

model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.3,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

#Save Model traning Result
filename = 'finalized_model_Spam_Ham_ANN.sav'
pickle.dump(model, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
