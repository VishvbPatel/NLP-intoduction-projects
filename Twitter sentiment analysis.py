#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[19]:


import nltk                            #importing the nltk library
import random                          #importing random library
from nltk.corpus import movie_reviews  #importing movie_reviews library from nltk
from nltk.classify.scikitlearn import SklearnClassifier    #importing scikitlearn library
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB #importing MultinomialNB, GaussianNB and BernoulliNB classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
import io
from nltk.tokenize import word_tokenize


# In[2]:


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# In[22]:


short_pos = io.open("/Users/vishvmac/Machine Learning/Machine learning projects/NLP projects/NLTK introductory projects/Sentimental analysis on twitter/positive.txt","r", encoding='latin-1')
short_neg = io.open("/Users/vishvmac/Machine Learning/Machine learning projects/NLP projects/NLTK introductory projects/Sentimental analysis on twitter/negative.txt","r")


# In[23]:


documents = []
for r in short_pos.split('\n'):
    documents.append((r,"pos"))
for r in short_neg.split('\n'):
    documents.append((r,"neg"))


# In[ ]:


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())
for w in short_neg_words:
    all_words.append(w.lower())    
    


# In[ ]:





# In[6]:


all_words = nltk.FreqDist(all_words) #making a frequency distribution of all the words


# In[7]:


word_features = list(all_words.keys())[:5000] #making list of the first 3000 high frequency words 


# In[8]:


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features                    #making a list of words with the tag whether they are present in the word_features list or not


# In[9]:


featuresets = [(find_features(rev), category) for (rev, category) in documents] #making a set for training and testing with categories


# In[10]:


training_set = featuresets[:10000]  #training set
testing_set = featuresets[10000:]   #testing set


# In[11]:


classifier = nltk.NaiveBayesClassifier.train(training_set)   #making classifier object with naive bayes
print("Original Naive Bayes Algo Accuray percent:", (nltk.classify.accuracy(classifier, testing_set))*100) #prining the accuracy of testing set
classifier.show_most_informative_features(15) #getting first 15 most informative features


# In[ ]:


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Algo Accuray percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


# In[ ]:


BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier Algo Accuray percent:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)


# In[ ]:


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier Algo Accuray percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


# In[ ]:


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier Algo Accuray percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


# In[ ]:


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier Algo Accuray percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)


# In[ ]:


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier Algo Accuray percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


# In[ ]:


voted_classifier = VoteClassifier(classifier, MNB_classifier, BNB_classifier, LogisticRegression_classifier, LinearSVC_classifier, 
                                 SGDClassifier_classifier, NuSVC_classifier)
print("voted_classifier Algo Accuray percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


# In[ ]:


print("classification:", voted_classifier.classify(testing_set[0][0]), "confidence%:", voted_classifier.confidence(testing_set[0][0])*100)
print("classification:", voted_classifier.classify(testing_set[1][0]), "confidence%:", voted_classifier.confidence(testing_set[1][0])*100)
print("classification:", voted_classifier.classify(testing_set[2][0]), "confidence%:", voted_classifier.confidence(testing_set[2][0])*100)
print("classification:", voted_classifier.classify(testing_set[3][0]), "confidence%:", voted_classifier.confidence(testing_set[3][0])*100)
print("classification:", voted_classifier.classify(testing_set[4][0]), "confidence%:", voted_classifier.confidence(testing_set[4][0])*100)
print("classification:", voted_classifier.classify(testing_set[5][0]), "confidence%:", voted_classifier.confidence(testing_set[5][0])*100)


# In[ ]:




