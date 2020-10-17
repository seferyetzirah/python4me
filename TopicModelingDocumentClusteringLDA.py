# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:47:19 2020

@author: Ian Saltzman is721863@sju.edu
Predictive Analytics: Machine Learning
Sentiment Analysis: Text Mining: 
    Tokenization, Stemming, Lemmatization
    Latent Dirichlet Allocation
"""
#adv token, stemming, lemmatization
import spacy
import nltk
# Load spacy's English-language models
en_nlp = spacy.load("en_core_web_sm")
# instantiate nltk's Porter stemmer
stemmer = nltk.stem.PorterStemmer()
# define function to compare lemmatization in spacy with stemming in nltk
def compare_normalization(doc):
 # tokenize document in spacy
 doc_spacy = en_nlp(doc)
 # print lemmas found by spacy
 print("Lemmatization")
 print([token.lemma_ for token in doc_spacy])
 # print tokens found in Porter stemmer
 print("Stemming")
 print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])
 
compare_normalization(u"Our meeting today was worse than yesterday, "
 "I'm scared of meeting the clients tomorrow.")

#latent dirichlet allocation
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import mglearn as mglearn
import matplotlib.pyplot as plt
vect = CountVectorizer(max_features=1000, max_df=.15)

X = vect.fit_transform(text_train)
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10, learning_method="batch", max_iter=25, random_state=0)
# We build the model and transform the data in one step
# Computing transform takes some time,
# and we can save time by doing both at once
document_topics = lda.fit_transform(X)
print("lda.components_.shape: {}".format(lda.components_.shape))
# For each topic (a row in the components_), sort the fetaures (ascending)
# Invert rows with [:, ::-1] to make sort descending
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# Get the feature names from the vectorizer
feature_names = np.array(vect.get_feature_names())
# Print out the 10 topics
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names, sorting=sorting,
 topics_per_chunk=5, n_words=10)
# For each topic (a row in the components_), sort the fetaures (ascending)
# Invert rows with [:, ::-1] to make sort descending
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# Get the feature names from the vectorizer
feature_names = np.array(vect.get_feature_names())
lda100 = LatentDirichletAllocation(n_components=100, learning_method="batch",
 max_iter=25, random_state=0)
document_topics100 = lda100.fit_transform(X)
topics = np.array([7, 16, 24, 25, 28, 36, 37, 45, 51, 53, 54, 63, 89, 97])
sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=topics, feature_names=feature_names,
 sorting=sorting, topics_per_chunk=5, n_words=20)
# sort by weight of "music" topic 45
music = np.argsort(document_topics100[:, 45])[::-1]
# print the five documents where the topic is most important
for i in music[:10]:
 # show first two sentences
 print(b".".join(text_train[i].split(b".")[:2])+b".\n")
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
topic_names = ["{:>2} ".format(i) + " ".join(words)
for i, words in enumerate(feature_names[sorting[:, :2]])]
# two column bar chart:
for col in [0, 1]:
 start = col * 50
 end = (col + 1) * 50
 ax[col].barh(np.arange(50), np.sum(document_topics100, axis=0)[start:end])
 ax[col].set_yticks(np.arange(50))
 ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
 ax[col].invert_yaxis()
 ax[col].set_xlim(0, 2000)
 yax = ax[col].get_yaxis()
 yax.set_tick_params(pad=130)
plt.tight_layout()