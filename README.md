# NLP-intoduction-projects
This repository contains projects related to the NLP basics and preprocessing.
Here, in the "NLP word tokenizing and sentence tokenizing" file, NLTK library is used in word tokenizing and sentence tokenizing operation. 
And in, "NLP removing stop words from sentence" file, the operation of removing stop words has been done with the help of stopwords in NLTK library.
Both of the files contains basic preprocessing operations which are essential in NLP.
So, in the "stemming of word" file the operation of stemming has been performed on the sample words with the help of "PorterStemmer". 
Then in the "Part of Speech tagging", the speech tagging operation has been done with the help of "state_union" library from nltk.corpus and "PunktSentenceTokenizer" from nltk.tokenize library. Then sample_text and train_text has been added. "PunktSentenceTokenizer" has been applied to train_text and then the sample_text has been tokenized with the help of trained train_text. Then, with the help of word_tokenize and pos_tag libraries the words and tags has been seperated and printed.
Then, two codes of Chunking and Chinking has been added, the chunking is the process of grouping of the words on the basis of their tags in terms of nouns or verbs and chinking is the process of removal of something that is not required. So, first the libraries has been added and then  the train text and sample text has been added. First of all the tagging of the words has been done with the help of "PunktSentenceTokenizer" and "pos_tag". After successfully tagging, the chunking has been done with the help of "RegexpParser". And chinking is not a seperate process, it's just the not required words of phrases are defined that will not be included in the chunk And then printing the chunked words.
The named entity recognition is used to find named entities in the text like, organization, person, location, date, time, money, percent, facility and geo-political entitiy. Here, the nltk library is used to find out the named entities. In other words, the named entity recognition is ued to find the general understanding of what the text is about. The nltk library has also option to find the general named entity rather than finding it in different categories like, organization, person, location, etc. The general named entity can be gained by adding "binary = True" in the attributes of "nltk.ne_chunk" along with "tagged". 
Here the Lemmatizing is similar to stemming, which is to find the actual meaning of the words despite of their different prounciation like, run, running, etc. The main difference between stemming and lemmatizing is that, the stemming might not use the actual word while, the lemmatizing uses an actual word. Here, the "WordNetLemmatizer" is used from nltk library. And in the limmatizing, the different meaning of the words based on their type, whether they are noun, verb or adjective, different meaning of the words can be achieved.
wordnet
In, text classification with the naive bayes model, the main idea behind it, is to make an classification model with the help of naive bayes to classify the movie reviews in positives and negatives. So, first, all the required libraries are imported. Then, making a list of all the words in the movie reviews with their labels like positive and negative. And then shuffling all the reviews. After that, making another list of words in lower case and then making an frequency distribution of all the words. Then, making another list of first 3000 high frequency words. Then, making a list based on whether they are present in the first 3000 high frequency list or not and then based on that again making set with categories to make a final training and testing set for the naive bayes classifier. Finally, making a classifier object with naive bayes to train and test the dataset. Also printing the first 15 most informative features.
