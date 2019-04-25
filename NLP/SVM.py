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


data = pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data=data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data=data.rename(columns={"v1":"class", "v2":"text"})
data.head()
data['length']=data['text'].apply(len)
data.head()

#Bar Chart Plot
count_class=pd.value_counts(data['class'], sort=True)
count_class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.show()

#Pie Chart Plot
count_class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()

#Categorize The Data to Ham, Spam
ham =data[data['class'] == 'ham']['text'].str.len()
sns.distplot(ham, label='Ham')
spam = data[data['class'] == 'spam']['text'].str.len()

#Histogram to visualize categorized Data
sns.distplot(spam, label='Spam')
plt.title('Distribution by Length')
plt.legend()


#find Most 30 Common Word in Spam and Ham Email
count1 = Counter(" ".join(data[data['class']=='ham']["text"]).split()).most_common(30)

data1 = pd.DataFrame.from_dict(count1)

#find Most 30 Common Word in Ham Email
data1 = data1.rename(columns={0: "words of ham", 1 : "count"})

count2 = Counter(" ".join(data[data['class']=='spam']["text"]).split()).most_common(30)

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
plt.show()

#Top 30 words of spam
data2.plot.bar(legend = False, color = 'green', figsize = (20,17))
y_pos = np.arange(len(data2["words of spam"]))
plt.xticks(y_pos, data2["words of spam"])
plt.title('Top 30 words of spam')
plt.xlabel('words')
plt.ylabel('number')
plt.show()

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

#########################################################################

#Save Model traning Result
filename = 'finalized_model_Spam_Ham.sav'
pickle.dump(svc, open(filename, 'wb'))


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