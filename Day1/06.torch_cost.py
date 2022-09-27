#!/usr/bin/env python
# coding: utf-8

# In[18]:


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# In[3]:


x_data = torch.FloatTensor([1,2,3])
y_data = torch.FloatTensor([1,2,3])


# In[9]:


def cost(x, y, w):
    hx = w*x # hx = torch.add(w, x)
    s = torch.sum((hx-y)**2) # torch.square((hx.sub(y))
    return s/len(x)


# In[23]:


def cost1(x, y, w):
    hx = w*x # hx = torch.add(w, x)
    return F.mse_loss(hx,y)


# In[32]:


def cost2(x, y, w):
    hx = w*x # hx = torch.add(w, x)
    loss_fn = torch.nn.MSELoss()
    return loss_fn(hx, y)


# In[33]:


def gradient(x, y, w):
    hx = w * x
    s = torch.sum((hx-y)*2*x)
    return s/len(x)


# In[34]:


print(cost1(x_data,y_data, -1))


# In[35]:


print(cost2(x_data,y_data, -1))


# In[11]:


print(cost(x_data,y_data, -1))
print(cost(x_data,y_data, 0))
print(cost(x_data,y_data, 1))
print(cost(x_data,y_data, 2))


# In[12]:


w = 10
for epoch in range(200):
    c = cost (x_data, y_data, w)
    print('epoch:', epoch, 'cost=', c, 'w=', w)
    g = gradient(x_data, y_data, w)
    w = w - 0.1 * g
print('최종 w: ', w)

