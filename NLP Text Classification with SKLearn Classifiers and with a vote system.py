#!/usr/bin/env python
# coding: utf-8

# # NLP Text Classification with SKLearn Classifiers and with a vote system

# Here, the main idea behind this project is to make an classification model with the help of naive bayes from nltk and different other classifiers from sklearn library to classify the movie reviews in positives and negatives. So, first, all the required libraries are imported. 



import nltk                            #importing the nltk library
import random                          #importing random library
from nltk.corpus import movie_reviews  #importing movie_reviews library from nltk
from nltk.classify.scikitlearn import SklearnClassifier    #importing scikitlearn library
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB #importing MultinomialNB, GaussianNB and BernoulliNB classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

#making class for the vote system and to define confidence percentage

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


# Then, making a list of all the words in the movie reviews with their labels like positive and negative. And then shuffling all the reviews. After that, making another list of words in lower case and then making an frequency distribution of all the words. Then, making another list of first 3000 high frequency words.



documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)] #making list of all the words in movie reviews with lables



random.shuffle(documents) #shuffling all the reviews



all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())  #making a list of all the words in the movie reviews in lower case



all_words = nltk.FreqDist(all_words) #making a frequency distribution of all the words



word_features = list(all_words.keys())[:3000] #making list of the first 3000 high frequency words 


# Then, making a list based on whether they are present in the first 3000 high frequency list or not and then based on that again making set with categories to make a final training and testing set for the naive bayes classifier. 



def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features                  #making a list of words with the tag whether they are present in the word_features list or not



featuresets = [(find_features(rev), category) for (rev, category) in documents] #making a set for training and testing with categories



training_set = featuresets[:1900]  #training set
testing_set = featuresets[1900:]   #testing set


# Finally, making a classifier object with naive bayes to train and test the dataset. Also printing the first 15 most informative features. Also, using the GaussianNB, MultinomialNB and BernoulliNB classifiers from sklearn library.


classifier = nltk.NaiveBayesClassifier.train(training_set)   #making classifier object with naive bayes
print("Original Naive Bayes Algo Accuray percent:", (nltk.classify.accuracy(classifier, testing_set))*100) #prining the accuracy of testing set
classifier.show_most_informative_features(15) #getting first 15 most informative features

#then 6 other classifier has been made to classify.

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Algo Accuray percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)



BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier Algo Accuray percent:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)



LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier Algo Accuray percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)



LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier Algo Accuray percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)



SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier Algo Accuray percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)



NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier Algo Accuray percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

#making the voted_classifier to get the classifier with best results

voted_classifier = VoteClassifier(classifier, MNB_classifier, BNB_classifier, LogisticRegression_classifier, LinearSVC_classifier, 
                                 SGDClassifier_classifier, NuSVC_classifier)
print("voted_classifier Algo Accuray percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

#printing the confidence percentage for defferent classifiers

print("classification:", voted_classifier.classify(testing_set[0][0]), "confidence%:", voted_classifier.confidence(testing_set[0][0])*100)     #printing the confidence for the classifier
print("classification:", voted_classifier.classify(testing_set[1][0]), "confidence%:", voted_classifier.confidence(testing_set[1][0])*100)
print("classification:", voted_classifier.classify(testing_set[2][0]), "confidence%:", voted_classifier.confidence(testing_set[2][0])*100)
print("classification:", voted_classifier.classify(testing_set[3][0]), "confidence%:", voted_classifier.confidence(testing_set[3][0])*100)
print("classification:", voted_classifier.classify(testing_set[4][0]), "confidence%:", voted_classifier.confidence(testing_set[4][0])*100)
print("classification:", voted_classifier.classify(testing_set[5][0]), "confidence%:", voted_classifier.confidence(testing_set[5][0])*100)

