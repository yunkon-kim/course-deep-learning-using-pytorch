#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


x_data = [1,2,3]
y_data = [1,2,3]


# In[3]:


def cost(x,y,w):
    c=0
    for i in range (len(x)):
        hx = w*x[i]
        c = c + (hx-y[i])**2
    return c/len(x)


# In[10]:


def gradient(x,y,w):
    c=0
    for i in range (len(x)):
        hx = w*x[i]
        c = c + (hx-y[i])*x[i]
    return c/len(x)


# In[11]:


w = 10
for epoch in range(200):
    c = cost (x_data, y_data, w)
    print('epoch:', epoch, 'cost=', c, 'w=', w)
    g = gradient(x_data, y_data, w)
    w = w - 0.1 * g
print('최종 w: ', w)

