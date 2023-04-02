#!/usr/bin/env python
# coding: utf-8

# In[520]:


import pandas as pd
import numpy as np
credits_df=pd.read_csv(r'C:\Users\anton\Downloads\credits.csv')
movies_df=pd.read_csv(r'C:\Users\anton\Downloads\movies.csv')
credits_df


# In[521]:


movies_df


# In[522]:


movies_df=movies_df.merge(credits_df,on="title")


# In[523]:


movies_df


# In[524]:


movies_df.info


# In[525]:


movies_df=movies_df[['movie_id','cast','crew','title','keywords','genres','overview']]


# In[526]:


movies_df.head()


# movies_df.isnull.sum()

# In[527]:


movies_df.isnull().sum()


# In[528]:


movies_df.dropna(inplace=True)


# In[529]:


movies_df.head()


# In[530]:


movies_df.duplicated().sum()


# In[531]:


movies_df.genres.iloc[0]

import ast


# In[532]:


def convert(x):
    l=[]
    for i in  ast.literal_eval(x):
        l.append(i["name"])
    return l
    


# In[533]:


movies_df["genres"]=movies_df["genres"].apply(convert)


# In[534]:


movies_df.head()


# In[535]:


movies_df["keywords"]=movies_df["keywords"].apply(convert)


# In[536]:


movies_df.head()


# In[537]:


movies_df['cast'][0]


# In[538]:


def convert3(x):
   l=[]
   count =0
   for i in  ast.literal_eval(x):
       if count != 3 :
           l.append(i["name"])
           count=count+1
       else:
           break
   return l
   


# In[539]:


movies_df["cast"]=movies_df["cast"].apply(convert3)


# In[540]:


movies_df["crew"][0]


# In[541]:


def convert4(x):
    l=[]
    count =0
    for i in  ast.literal_eval(x):
        if i["job"]== 'Director' :
            l.append(i["name"])
            break
    
    return l


# In[542]:


movies_df["crew"]=movies_df["crew"].apply(convert4)


# In[543]:


movies_df.head()


# In[544]:



movies_df['tilte']=movies_df['title'].apply(lambda x:[i.replace(" ","")for i in x])
movies_df['crew']=movies_df['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies_df['cast']=movies_df['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies_df['keywords']=movies_df['keywords'].apply(lambda x:[i.replace(" ","")for i in x])




# In[545]:


movies_df.head()


# In[546]:


movies_df["tags"]= movies_df["genres"] + movies_df["keywords"] + movies_df["cast"] + movies_df["crew"]


# In[547]:


movies_df['overview'] = movies_df["overview"].apply(lambda x: x[1:-1].split(','))


# In[548]:


movies_df["tags"]= movies_df["genres"] + movies_df["keywords"] + movies_df["cast"] + movies_df["crew"]+movies_df["overview"]


# In[549]:


new_df=movies_df[["tags","movie_id","title",]]


# In[550]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)    
new_df["tags"].apply(stem)


# movies_df.head()

# In[551]:


new_df.head()


# In[552]:


new_df["tags"][0]


# In[553]:


new_df["tags"]=new_df["tags"].apply(lambda x:'  '.join(x))


# In[554]:


new_df["tags"]=new_df["tags"].apply(lambda x:x.lower())


# In[555]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words="english")


# In[556]:


vector=cv.fit_transform(new_df["tags"]).toarray()


# In[557]:


from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vector)


# In[558]:


new_df[new_df["title"] == "Avatar"].index[0]


# In[570]:


def recommend(k):
    p=new_df[new_df["title"] == k ].index[0]
    distances = sorted(list(enumerate(similarity[p])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)     
               
    


# In[571]:


recommend('Avatar')


# In[ ]:




