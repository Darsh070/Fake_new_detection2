#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf


# In[2]:


data = pd.read_csv(r"E:\kaggle_dataset\train.csv")


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.tail()


# In[10]:


X = data.drop('label',axis=1)


# In[11]:


X


# In[12]:


X.shape


# In[14]:


y=data['label']


# In[15]:


y


# In[16]:


y.shape


# In[17]:


y.head()


# In[70]:


y.value_counts()
tf.__version__


# In[71]:


data.shape


# In[72]:



from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout


# In[73]:


### Vocabulary size
voc_size=5000


# In[74]:


messages=X.copy()


# In[75]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer


# In[76]:


data.isnull().mean()


# In[77]:


data = data.dropna()


# In[78]:


data.shape


# In[79]:


data.head(10)


# In[80]:


messages=data.copy()


# In[81]:


messages.reset_index(inplace=True)#


# In[82]:


messages.head(10)


# In[83]:


import nltk
import re
from nltk.corpus import stopwords


# In[84]:


nltk.download('stopwords')


# In[85]:


from nltk.stem.porter import PorterStemmer


# In[86]:


ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[87]:


corpus


# In[53]:


onehot_repr=[one_hot(words,voc_size)for words in corpus] 
onehot_repr


# In[88]:


sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


# In[98]:



embedded_docs[0]


# In[101]:


len(embedded_docs)


# In[100]:


y.shape


# In[90]:


## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[91]:



## Creating model
embedding_vector_features=40
model1=Sequential()
model1.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dropout(0.3))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())


# In[102]:


y=data.dropna()


# In[114]:


y=data['label']
y.shape


# In[115]:


len(embedded_docs),y.shape


# In[116]:


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


# In[117]:


X_final.shape,y_final.shape


# In[118]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


# In[119]:



### Finally Training
model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


# In[120]:


X_final.shape,y_final.shape


# In[128]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[133]:


y_pred = model1.predict_classes(X_test)


# In[134]:


print(confusion_matrix(y_test,y_pred))


# In[135]:


print(accuracy_score(y_test,y_pred))


# In[137]:


print(classification_report(y_test,y_pred))


# In[ ]:




