#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[5]:


def fn(x):
    print(x/x.sum())


# In[7]:


def softmax(x):
    e = np.exp(x)
    print(e)
    print('========')
    print(e/e.sum())


# In[10]:


arr = np.array([2.0, 1.0, 0.1])
# fn(arr)
softmax(arr)

