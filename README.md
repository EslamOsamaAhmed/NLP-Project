
# Project Overview :
	We Have A Binary Classification Problem of defining whether the E-mail Is a spam or ham Email.

# DataSets :
	1) " dataset-1.csv " -> IS A labeled dataset consists of 5175  Record (or "spam and ham E-mails observations ").
	2) " dataset-2.csv " -> IS A labeled dataset consists of 5573  Record (or "spam and ham E-mails observations ").
	3) " dataset-3.csv " -> IS A labeled dataset consists of 30000 Record (or "spam and ham E-mails observations ").

# DataSets Preprocessing :
	Our Three Datasets were four folders ('enron1 , enron2 , enron3 , enron4') each separated to spam file consists of spams.txt ('spam emails observations in txt files format') and the same for hams.txt ('ham emails observations  in txt files format') , so we made some processing steps on those folders for converting it to ('spam-ham.csv') file .
	=》 Firstly , we Imported the necessary libraris we have used :

# Prerequests:
    install numpy, keras, sklearn, pandas, matplotlib
  
# DataSets Preprocessing :
    **
    import os
    import pandas as pd 
    import codecs        # For Opening .txt files 
    import random
    **

	=》 Then , defining A set for each class :

**
	ham = [[]]
	spam = [[]]
	emails = [[]]
**
	=》 Then , by two 'for' loops we have iterated to open those text files to read and append them respectevely at thier pre-defined lists :

**
	# Ham File
	for filename in os.listdir("enron4/enron4/ham/"):  
   	   file = open("enron4/enron4/ham/" + filename,"r+")
   	   ham.append([file.read(),"ham"])

	# Spam File
	for filename in os.listdir("enron4/enron4/spam/"):  
           with codecs.open("enron4/enron4/spam/" + filename, 'r', encoding='utf-8',errors='ignore') as fdata:
           spam.append([fdata.read(),"spam"])
**
	
	=》 Then , we have merged them togather at one defined list ('emails'):

**
	# Merge two Lists
	emails = ham + spam
**

	=》 Then , we shuffled this final list ('emails') :

**
	# Shuffl List
 	random.shuffle(emails)
**

	=》 Then , We converted the final shuffled list to A DataFrame By pandas package :

**
	# Create the pandas DataFrame 
	df = pd.DataFrame(emails, columns = ['text', 'label'])
**

	=》 Finally , we have to convert this dataframe to a ('csv') file and give A path to the converter function :
**
 	# Convert DF to CSV
	df.to_csv (r"E:\Heba's college\Level four\Semester Two\NLP\NLP-project\Data Sets\AnnDataset.csv", index = None, header=True)
**

*************************************************************************************************************************************
=》 Data Pre-Processing Before getting into Machine\Deep Learning Models :

	=》 for each Dataset , we loaded and dropped un-necessary columns and get some plots ('Bar chart , pie chart , histogram ') , adding additional column ('length') which identify the leanth of each e-mail observation , ploting the most thirty words common words in each (spam , ham), and encoding the two categorical classes to zeros and ones .
	=》 Then , we defined ('pre-processing') function wjich takes a text ('E-mail') and doing some processing upon like ('translate') function which takes a string and ('Panctuate') it that have to find all punctuation signs on this text and remove it ,then make sure the this words aren't from the language stopwords finally ('stemming') the rest words to get its origin without any grammatecal additions .

**
       def pre_process(text):
   	 text = text.translate(str.maketrans('', '', string.punctuation))
   	 text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    	 words = ""
    	 for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i)) + " "
    	return words
**

	=》 Then , getting to vectorize words what rests from previous function steps to be able to be learnt by machine\deep learning algorithms by ('tfidfvectorizer').

** 
	#Prprocessing for Data to find the best Acc and Vectorizing the data to be able to be learned
	textFeatures = data['text'].copy()
	textFeatures = textFeatures.apply(pre_process)
	vectorizer = TfidfVectorizer("english")
	features = vectorizer.fit_transform(textFeatures)
**
	
	=》 Then , we have to split our resulted Data after those all previous steps into training and test Data to learn the model and testing by the same dataset :

**
 	#Split Data after Preprocessing to Training set & Testing set with it's Labels
	features_train, features_test, labels_train, labels_test = train_test_split(features, data['class'], test_size=0.3,random_state=111)
**

	=》 Then , getting into learning several models to classify right our binary classification problem ( spam-or-ham ) all over datsets we had used .

*************************************************************************************************************************************

# Introduction : 
    K-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:

    In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

    In k-NN regression, the output is the property value for the object. This value is the average of the values of its k nearest neighbors. (Reference Wikipedia)

*******************************

Problems faced when optimizing KNN code:

The main concern with optimizing the KNN classifier is to select the right number of neighbors K and the distance function to be considered.


******************************


#Conclusion :

    In terms of values of K When we tried picking very small values for K that is 1 or 2 then the knn classifier was over fitting the dataset. Because it was trying to build individual data points with their own individual decision boundaries. So, the classifier was performing better on training dataset that is was giving better accuracies on it whereas on test dataset the accuracy was getting reduced.

    When we tried picking very large value for K that is 50 or more then the knn classifier was under fitting the dataset. Because, it is not fitting the input data properly. So, the classifier was not performing better on train as well as test dataset.


*********************************


#requirments : 

    Python 3.x
    numpy
    scikit-learn
    scipy

 *******************************

#Packages for visualization :

    import numpy as np
    %matplotlib inline
    import matplotlib.pyplot as plt

*************************************

#Dataset Format : 

    CSV (Comma Separated Values) format.
    Attributes can be integer or real values.
    List attributes first, and add response as the last parameter in each row.
    E.g. [4.5, 7, 2.6, "Orange"], where the first 3 numbers are values of attributes and "Orange" is one of the response classes.
    Another example can be [1.2, 4.3, 3], in this case there are 2 attributes while the response class is the integer 3.
    The square brackets are shown for convenience in reading, don't put them in your CSV file.
    Responses can be integer, real or categorical.

***************************************


#Algorithm :

    It generates k * c new features, where c is the number of class labels. The new features are computed from the distances between the observations and their k nearest neighbors inside each class, as follows:

    The first test feature contains the distances between each test instance and its nearest neighbor inside the first class.
    The second test feature contains the sums of distances between each test instance and its 2 nearest neighbors inside the first class.
    The third test feature contains the sums of distances between each test instance and its 3 nearest neighbors inside the first class.
    And so on.
    This procedure repeats for each class label, generating k * c new features. Then, the new training features are generated using a n-fold CV approach, in order to avoid overfitting.

************************************

#Overview of the different implementations (Python/R/Matlab) :

    Of the three implementations provided here, the Python implementation is the most thoroughly tested and the fastest. However, all implementations run reasonably fast - typically on the order of seconds or minutes for datasets containing < 5,000 cells. For larger datasets, we recommend using the Python implementation. The Python implementation also provides a command-line interface (see below), which makes it easy to use for non-Python users.

    We strive to ensure the correctness of all implementations and to make them all as consistent as possible. However, due to differences in terms of how the randomized PCA is implemented in each language, there are currently small differences in the exact results produced by each implementation. We appreciate any reports of inconsistencies or suggestions for improvements.

*********************************

#Notes :

    Keep the data set files in the working directory of project as defined by the IDE configuration.
    When running in stand alone mode (E.g. command line), keep the data sets in the same directory as the script.

***************************** 

# We Used ANN Classifier For Train A Type of Deep learning Models upon our Dataset " spam-or-ham classification problem " .
       =»Firstly , Importing necessary modules from keras libirary :
**
              from keras.models import Sequential
              from keras.layers import Dense 
**
       =»Then , we create An object of the sequential Model " responsible for building a model consists of sequence of layers " 
**          
              classifier = Sequential()
**
         =» we Built our sequential model from 3 Layers :
              =»First Hidden Layer with " Input_dim = 8035 , and output_dim = 3900 " and we initialize the network weights of " uniform " distribution , in this case , weights are random numbers between 0 and 0.05 because that is the default uniform weight initialization in Keras , The Activation function of rectifier ('relu').
             =»Second Hidden Layer with " output_dim = 1950 , The Activation function ('relu') .
             =» Third Layer is The prediction output layer with output_dim of 1 neuron for predicting the class of text ( spam or ham ), Activation function is ('segmoid') to ensure our network output is between 0 and 1 and easy to map to either a probability of classes.
**
            classifier.add(Dense(output_dim = 3900, init = 'uniform', activation = 'relu', input_dim = 8035)) 
           classifier.add(Dense(output_dim = 1950, init = 'uniform', activation = 'relu')) 
            classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
**
      =» Then , We comiled our Network with logarithmic loss function (' binary_crossentropy') which for binary classification problem Defined in Keras , the efficient gradient descent Optimizer “adam” also predefined in keras which used to search through different weights , and The Accuracy Metric .

**
       classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

**
        =» Then , Fit the network by our splitted Dataset by " fit " function with our training observations " features_train " and its labels " labels_train" , defining the number of "epochs" (Iterations for spesific number of instances with different weights values ) and the : "batch_size" ( the number of instancez that are evaluated before a weight update in the network ) .
**
       classifier.fit(X, Y, epochs=195, batch_size=20)
**
         =» Then , Evaluating model on our training set , This will generate a prediction for each input and output pair and collect scores, including the average loss and any metrics you have configured, such as accuracy.
**
        score = classifier.evaluate(featurs_train , labels_train , verbose=0 )
**
        =» Then , predicting some labels of miss_labeled test data we had splited before .
**
          Y_pred = classifier.predict (features_test)
          y_pred = (y_pred > 0.5)
**
        =» Finally , Making the confusion Matrix (TP, TN , FP , FN).
**
          from sklearn.metrics import confusion_matrix
         cm = confusion_matrix(labels_test, y_pred)
** 
**** END Of Ann Model *****
=» Using Another Machine Learning Algorithm for Binary classification problem , we have used " support vector machine " non-linear classifier to separately deviding our to classes " Spam , Ham " by A decision boundary .
      =» Using " Sklearn " library we import The class of SVM called ('svc').
**
        from sklearn.svm import SVC 
**
      =»Then define our model with svc object The non-linear kernel of ('segmoid') which makes a non-linear decision boundary  between the two classes (sapm , ham) with probabilstic predction And fit our model with our training data (obsrrvations and its labels ).
**
        svc = SVC(kernel='sigmoid').
        svc.fit(features_train, labels_train).
**
      =» Then Predict the labels of test observations we have splitted before.
**
      prediction = svc.predict(features_test).
**
     =» Finally saving The accuracy of deffering the predections and the real test labels using sklearn.metrics.
**
     accuracy_score(labels_test,prediction).
**

*****************************

#RNN (Recurrent Neural Network):

    **A recurrent neural network (RNN)** is a class of artificial neural network where connections between nodes form a directed graph along a temporal sequence.
    This allows it to exhibit temporal dynamic behavior. Unlike feedforward neural networks, RNNs can use their internal state (memory) to process sequences of inputs. This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition.

    The term "recurrent neural network" is used indiscriminately to refer to two broad classes of networks with a similar general structure, where one is finite impulse and the other is infinite impulse. Both classes of networks exhibit temporal dynamic behavior. A finite impulse recurrent network is a directed acyclic graph that can be unrolled and replaced with a strictly feedforward neural network, while an infinite impulse recurrent network is a directed cyclic graph that can not be unrolled.

    (LSTM)
    ** Both finite impulse and infinite impulse recurrent networks can have additional stored state, 
    and the storage can be under direct control by the neural network.
    The storage can also be replaced by another network or graph, if that incorporates time delays or has feedback loops.** 
    Such controlled states are referred to as gated state or gated memory, and are part of long short-term memory networks (LSTMs) and gated recurrent units.
    Long short-term memory (LSTM) networks were discovered by Hochreiter and Schmidhuber in 1997 and set accuracy records in multiple applications domains.

    Around 2007, LSTM started to revolutionize speech recognition, outperforming traditional models in certain speech applications.

    LSTM broke records for improved machine translation, Language Modeling and Multilingual Language Processing.
    LSTM combined with convolutional neural networks (CNNs) improved automatic image captioning.

    max_words = 1500
    max_len = 200
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# RNN Tockenize:
    the input sentence to have the important words of the inputs to get it to the layers 
    with putting the maximum length of the word and the maximum words that will be as input in the Mail.
    And fit these on Traning Mails to fit the tokenz on it using tok the convert it to using 
    **texts_to_sequences** tok.texts_to_sequences(X_train) 

inputs = Input(name='inputs',shape=[max_len])
layer = Embedding(max_words,50,input_length=max_len)(inputs)
layer = LSTM(64)(layer)
layer = Dense(256,name='FC1')(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(1,name='out_layer')(layer)
layer = Activation('sigmoid')(layer)
model = Model(inputs=inputs,outputs=layer)

**RNN** Using LSTM Layers beside Dense Layers to match the training on the sequence of the seneteces after convert it using texts_to_sequences and then apply LSTM Layern on it with its Hyper Parameters and Activation Functions then fit the training sentences (Training Sequences) to the Training Labels.

*****************************

ALL Accuracy of Models (KNN, SVM, RNN, ANN) are plotted and saved in Models File as Results of 
Training Acc & Testing Acc of Models and the Learning of Models are Saved in .sav files to be loaded any time.

ALL Visualizations of Datasets (dataset-1, dataset-2, dataset-2) are saved as graphs and plots in DataSets file with its Charts.

# Authors

    Eslam Osama Ahmed
    Heba Gamal Aldin
    Eman Hesham

# License
    This project is licensed under the FCIH License




