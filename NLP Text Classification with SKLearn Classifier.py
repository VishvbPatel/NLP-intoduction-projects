#!/usr/bin/env python
# coding: utf-8

# # NLP Text Classification with SKLearn Classifier

# Here, the main idea behind this project is to make an classification model with the help of naive bayes from nltk and different other classifiers from sklearn library to classify the movie reviews in positives and negatives. So, first, all the required libraries are imported. 

# In[15]:


import nltk                            #importing the nltk library
import random                          #importing random library
from nltk.corpus import movie_reviews  #importing movie_reviews library from nltk
from nltk.classify.scikitlearn import SklearnClassifier    #importing scikitlearn library
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB #importing MultinomialNB, GaussianNB and BernoulliNB classifiers


# Then, making a list of all the words in the movie reviews with their labels like positive and negative. And then shuffling all the reviews. After that, making another list of words in lower case and then making an frequency distribution of all the words. Then, making another list of first 3000 high frequency words.

# In[4]:


documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)] #making list of all the words in movie reviews with lables


# In[5]:


random.shuffle(documents) #shuffling all the reviews


# In[6]:


all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())  #making a list of all the words in the movie reviews in lower case


# In[7]:


all_words = nltk.FreqDist(all_words) #making a frequency distribution of all the words


# In[8]:


word_features = list(all_words.keys())[:3000] #making list of the first 3000 high frequency words 


# Then, making a list based on whether they are present in the first 3000 high frequency list or not and then based on that again making set with categories to make a final training and testing set for the naive bayes classifier. 

# In[9]:


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features                  #making a list of words with the tag whether they are present in the word_features list or not


# In[10]:


featuresets = [(find_features(rev), category) for (rev, category) in documents] #making a set for training and testing with categories


# In[11]:


training_set = featuresets[:1900]  #training set
testing_set = featuresets[1900:]   #testing set


# Finally, making a classifier object with naive bayes to train and test the dataset. Also printing the first 15 most informative features. Also, using the GaussianNB, MultinomialNB and BernoulliNB classifiers from sklearn library.

# In[16]:


classifier = nltk.NaiveBayesClassifier.train(training_set)   #making classifier object with naive bayes
print("Original Naive Bayes Algo Accuray percent:", (nltk.classify.accuracy(classifier, testing_set))*100) #prining the accuracy of testing set
classifier.show_most_informative_features(15) #getting first 15 most informative features


# In[17]:


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Algo Accuray percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


# In[19]:


BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier Algo Accuray percent:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)


# In[ ]:


GNB_classifier = SklearnClassifier(GaussianNB())
GNB_classifier.train(training_set)
print("GNB_classifier Algo Accuray percent:", (nltk.classify.accuracy(GNB_classifier, testing_set))*100)

