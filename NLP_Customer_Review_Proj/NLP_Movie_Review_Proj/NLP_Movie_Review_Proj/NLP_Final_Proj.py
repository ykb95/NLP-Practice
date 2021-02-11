# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:38:04 2018

@author: Yogendra
"""

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import re
import pandas as pd
import numpy as np


with open('C:\\Users\\IBM_ADMIN\\Downloads\\python_Lecs\\NLP_Movie_Review_Proj\\yelp_labelled.txt', 'r') as f:
    review_lines = f.readlines()
    
print(type(review_lines))

print(len(review_lines))

for one_line in review_lines[0:10]:
    print(one_line)

reviews = []
review_labels = []

for line in review_lines: #Splitting reviews into text and labels
    line_list = line.split("\t")
    review = line_list[0]
    review = review.lower()
    reviews.append(review)
    label = line_list[1]
    label = int(label)
    review_labels.append(label)

print(len(reviews))
print(len(review_labels))
print(reviews[0])
print(review_labels[0])

#Tokenize

#Creating a single string of all the reviews

review_string = " ".join(reviews)
print(review_string)
tknzr = TweetTokenizer()
review_tokens = tknzr.tokenize(review_string)
print(len(review_tokens))

#Removing punctuation
punctuation = re.compile(r'[-.?!,":;()|0-9]')
review_tokens2 = []

for token in review_tokens:
    word = punctuation.sub("", token)
    if len(word)>0:
        review_tokens2.append(word)
        
print(len(review_tokens2))

#Removing stopwords

review_tokens3 = []
stp_words = set(stopwords.words('english'))

for token in review_tokens2: # remove stop words
    token = token.lower()
    if token not in stp_words:
        review_tokens3.append(token)
        
print(len(review_tokens3))

#Finding Frequency Distribution

fdist = FreqDist()
for word in review_tokens3:
    fdist[word]+=1
    
print(len(fdist))

fdist_top20 = fdist.most_common(20) #top 20 tokens with highest frequency

#Finding Parts of speech tags

pos_list = []
for token in review_tokens3: # List after stopwords and punctuation
    pos_list.append(nltk.pos_tag([token]))# The word should be inside [], else tagger will take it as string
    
print(len(pos_list))

print(pos_list[1])

pos_set = set()

for pos in pos_list: # Finding all different Part of Speech Tags
    pos_set.add(pos[0][1])
    
print(len(pos_set))

print(pos_set)

#Finding Adjectives
pos_JJ = []

for each_POS in pos_list:
    if each_POS[0][1] in ["JJ","JJR","JJS"]:
        pos_JJ.append(each_POS[0][0])
        
print(len(pos_JJ))

fdist_JJ = FreqDist()
for word in pos_JJ:
    fdist_JJ[word] += 1
    
print(len(fdist_JJ))

fdist_JJ_top20 = fdist_JJ.most_common(20) #top 20 tokens with highest frequencies

print(fdist_JJ_top20)

#Lemmatizing words

word_lem = WordNetLemmatizer()
lem_ADJ = []
lem_ADV = []
lem_VERB = []

for word in review_tokens3:
    word_pos =nltk.pos_tag([word])
    
    if word_pos[0][1] in ["JJ", "JJR", "JJS"]:
        lem_ADJ.append((word_pos[0][0], word_lem.lemmatize(word, wordnet.ADJ)))
        
    if word_pos[0][1] in ["RB","RBR","RBS"]:
        lem_ADV.append((word_pos[0][0], word_lem.lemmatize(word, wordnet.VERB)))
        
    if word_pos[0][1] in ["VB","VBD","VBG","VBN","VBZ"]:
        lem_VERB.append((word_pos[0][0], word_lem.lemmatize(word, wordnet.VERB)))


print(len(lem_ADJ))    

print(lem_ADJ[:10])

print(lem_ADV)

print(lem_ADV[:10])

print(lem_VERB)

print(lem_VERB[:10])

# Checking most frequent n-grams

#Segregating all positive and negative reviews

positive_reviews = []
negative_reviews = []

for line in review_lines: # Splitting reviews into text and labels
    line_list = line.split("\t")
    
    label = line_list[1]
    label = int(label)
    
    if label==1:
        pos_review = line_list[0]
        pos_review = pos_review.lower()
        positive_reviews.append(pos_review)
        
    elif label==0:
        neg_review = line_list[0]
        neg_review = neg_review.lower()
        negative_reviews.append(neg_review)
        
print(len(positive_reviews))
print(len(negative_reviews))

#Working on positive reviews

pos_review_string = " ".join(positive_reviews)

pos_rev_tokens = nltk.word_tokenize(pos_review_string)

pos_rev_trigrams = list(nltk.trigrams(pos_rev_tokens))

print(len(pos_rev_trigrams))

print(pos_rev_trigrams[0])

fdist_pos_rev = nltk.FreqDist(pos_rev_trigrams)

print(fdist_pos_rev.most_common(10))

#Working on Negative Reviews
    
neg_review_string = " ".join(negative_reviews)

neg_rev_tokens = nltk.word_tokenize(neg_review_string)

neg_rev_trigrams = list(nltk.trigrams(neg_rev_tokens))

print(len(neg_rev_trigrams))

print(neg_rev_trigrams[0])

fdist_neg_rev = nltk.FreqDist(neg_rev_trigrams)

print(fdist_neg_rev.most_common(10))    

#Predicting reviews using machine learning

#Creating features and labels

review_list = []

for line in review_lines: # Splitting reviews into text and labels
    line_list = line.split("\t")
    review = line_list[0]
    review = review.lower()
    review_list.append(review)
    
print(len(review_list))

print(review_list[1])

tf_vect = TfidfVectorizer(min_df = 2, lowercase = True, stop_words = 'english')

X_TFIDF = tf_vect.fit_transform(review_list)

print(type(X_TFIDF))

print(X_TFIDF.shape)

X_TFIDF_names = tf_vect.get_feature_names()

X_tf = pd.DataFrame(X_TFIDF.toarray(), columns = X_TFIDF_names)

print(X_tf.head())

y = pd.Series(review_labels)

print(type(y))

print(y.shape)

print(y.head())

#Train-test Split

X_TF_train, X_TF_test, y_TF_train, y_TF_test = train_test_split(X_tf, y, test_size=0.2, random_state=5)

print(X_TF_train.shape)
print(X_TF_test.shape)
print(y_TF_train.shape)
print(y_TF_test.shape)
    
#Using Multinomial Naive Bayes
clf_TF = MultinomialNB()
clf_TF.fit(X_TF_train, y_TF_train)

y_TF_pred = clf_TF.predict(X_TF_test)

print(type(y_TF_pred))

print(y_TF_pred.shape)

print(metrics.accuracy_score(y_TF_test, y_TF_pred))

score_clf_TF = confusion_matrix(y_TF_test, y_TF_pred)

print(score_clf_TF)

TP = score_clf_TF[0][0]
FP = score_clf_TF[0][1]
FN = score_clf_TF[1][0]
TN = score_clf_TF[1][1]

print('True Positive: ',TP)
print('False Positive: ', FP)
print('False Negative: ', FN)
print('True Negative: ', TN)

print("Correctly Identified: ", TP+TN)
print("Wrongly Identified: ", FP+FN)

#Checking Classifier on new Data

def find_sentiments (sentence):
    sent_list = [sentence]
    sent_vect = tf_vect.transform(sent_list)
    sent_pred = clf_TF.predict(sent_vect)
    print("Sentiment: ",sent_pred[0])
    return

find_sentiments("I liked the food a lot")
find_sentiments("The food was very bad")
find_sentiments("The place was good but the food was bad")    
    