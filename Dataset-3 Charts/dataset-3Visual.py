# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:31:49 2019

@author: Eman

"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
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

datasetname = "dataset-3.csv"

data = pd.read_csv(datasetname,encoding='latin-1')
data.head()

data=data.rename(columns={"v1":"class", "v2":"text"})
data = data.dropna(subset=['class'])
data = data.dropna(subset=['text'])
data.head()
data['length']=data['text'].apply(len)
data.head()

#Bar Chart Plot
count_class=pd.value_counts(data['class'], sort=True)
count_class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.savefig('Bar chart', dpi=100)
plt.show()

#Pie Chart Plot
count_class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.savefig('Pie chart', dpi=100)
plt.show()

#Categorize The Data to Ham, Spam
ham =data[data['class'] == 1]['text'].str.len()
sns.distplot(ham, label='Ham')
spam = data[data['class'] == 0]['text'].str.len()

#Histogram to visualize categorized Data
sns.distplot(spam, label='Spam')
plt.title('Distribution by Length')
plt.savefig('Histogram', dpi=100)
plt.legend()


#find Most 30 Common Word in Spam and Ham Email
count1 = Counter(" ".join(data[data['class']==1]["text"]).split()).most_common(30)

data1 = pd.DataFrame.from_dict(count1)

#find Most 30 Common Word in Ham Email
data1 = data1.rename(columns={0: "words of ham", 1 : "count"})

count2 = Counter(" ".join(data[data['class']==0]["text"]).split()).most_common(30)

data2 = pd.DataFrame.from_dict(count2)

#find Most 30 Common Word in Spam Email
data2 = data2.rename(columns={0: "words of spam", 1 : "count_"})

data1.plot.bar(legend = False, color = 'purple',figsize = (20,15))

#Top 30 words of ham
y_pos = np.arange(len(data1["words of ham"]))
plt.xticks(y_pos, data1["words of ham"])
plt.title('Top 30 words of ham')
plt.xlabel('words')
plt.ylabel('number')
plt.savefig('Top 30 words of ham', dpi=100)
plt.show()

#Top 30 words of spam
data2.plot.bar(legend = False, color = 'green', figsize = (20,17))
y_pos = np.arange(len(data2["words of spam"]))
plt.xticks(y_pos, data2["words of spam"])
plt.title('Top 30 words of spam')
plt.xlabel('words')
plt.ylabel('number')
plt.savefig('Top 30 words of spam', dpi=100)
plt.show()


