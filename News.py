import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


#Read the data
data=pd.read_csv('news.csv')
data.shape
data.head()

#Get the lable from data
lables=data.label
print(lables.head())

#Split the dataset inti trainig and testing
x_train, x_test, y_train, y_test = train_test_split(data['text'], lables, test_size=0.2, random_state=7)

#Initialize the TfidVector
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transform test/set with tfidf_vectorizer
tfid_train=tfidf_vectorizer.fit_transform(x_train)
tfid_test=tfidf_vectorizer.transform(x_test)


#Initialize the PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)

#Fit and transform test/set with PassiveAggressiveClassifier
pac.fit(tfid_train, y_train)

#predict on test set and calulate accuraccy
y_pred = pac.predict(tfid_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy:{round(score*100,2)}%')

#Build confusion matrix
print(confusion_matrix(y_test, y_pred,labels=['FAKE','REAL']))

