#!/usr/bin/env python
# coding: utf-8

# ### Question 1

# In[22]:


#import libraries
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[6]:


#Clean the “Tweet Content” column by removing non-text data and stop words.

data= pd.read_csv('twitter_sample.csv')
data.head()
    


# In[20]:


corpus = ['The sky is blue and beautiful.',
          'Love this blue and beautiful sky!',
          'The quick brown fox jumps over the lazy dog.',
          "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
          'I love green eggs, ham, sausages and bacon!',
          'The brown fox is quick and the blue dog is lazy!',
          'The sky is very blue and the sky is very beautiful today',
          'The dog is lazy but the brown fox is quick!'    
]
labels = ['weather', 'weather', 'animals', 'food', 'food', 'animals', 'weather', 'animals']

corpus = np.array(corpus)
corpus_df = pd.DataFrame({'twitter_sample.csv': corpus, 
                          'Category': labels})
corpus_df = corpus_df[['twitter_sample.csv', 'Category']]
corpus_df


# In[26]:


wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)


# ## Point 2

# In[24]:


#Filtering only tweets (not re-tweets) use your class from part one of this exercise
#to build BOW and TF-IDF Vectorizer representations of the text; 
#print your results.

# get unique words as feature names
unique_words = list(set([word for doc in [doc.split() for doc in norm_corpus] 
                         for word in doc]))
def_feature_dict = {w: 0 for w in unique_words}
print('tweets:', unique_words)
print('Default Feature Dict:', def_feature_dict)


# ## Point 3

# In[25]:


#Find one or more documents (each tweet is a document) that are similar to each other
#using Cosine Similarity; print your results. 

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)
similarity_df


# In[ ]:




