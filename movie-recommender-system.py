#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# ## Preprocessing ##

# In[3]:


movies = movies.merge(credits,on='title')


# In[4]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[5]:


movies.dropna(inplace=True)


# In[6]:


movies.duplicated().sum()


# In[7]:


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[8]:


movies['genres'] = movies['genres'].apply(convert)


# In[9]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[10]:


def convert3(obj):
    l=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            l.append(i['name'])
            counter+=1
        else:
            break
    return l


# In[11]:


movies['cast'] = movies['cast'].apply(convert)


# In[12]:


def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director' :
            l.append(i['name'])
            break
    return l


# In[13]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[14]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[15]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[16]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[17]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[18]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])


# In[19]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))


# In[20]:


new['tags'] = new['tags'].apply(lambda x:x.lower())


# ## Apply Stemming ##

# In[21]:


ps = PorterStemmer()


# In[22]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[23]:


new['tags'] = new['tags'].apply(stem)


# ## Vectorization- Bag of Words ##

# In[24]:


cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()


# In[25]:


cv.get_feature_names()


# In[27]:


similarity = cosine_similarity(vector)


# In[28]:


similarity


# In[29]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

