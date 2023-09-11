#!/usr/bin/env python
# coding: utf-8

import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import seaborn as sn


# In[8]:


a=np.arange(4).reshape(2,2)
a


# In[27]:





# In[1]:


import sklearn as sl
from sklearn import tree as tr
features=[[240,0],[245,1],[230,0],[240,1]]
labels = [0,1,0,1]
clf=tr.DecisionTreeClassifier()
clf=clf.fit(features,labels)
clf.predict([[245,0]])


# In[ ]:





# In[ ]:




