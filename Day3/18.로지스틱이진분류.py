#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential, Sigmoid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F


# In[2]:


x = torch.FloatTensor([[1,3],[2,2],[3,1],[4,6],[5,5],[6,4]]) # 공부한시간, 출석일수
y = torch.FloatTensor([[0],[0],[0],[1],[1],[1]]) # 합격여부


# In[5]:


w = torch.empty([2,1], requires_grad=True)
b = torch.empty(1, requires_grad=True)
torch.nn.init.uniform_(w)
torch.nn.init.uniform_(b)


# In[9]:


def cost():
    z = torch.matmul(x,w) + b
    cost_i = F.binary_cross_entropy_with_logits(z, y)
    c = torch.mean(cost_i)
    return c


# In[12]:


def cost1():
    z = torch.matmul(x,w) + b
    hx = Sigmoid(z)
    cost_i = F.binary_cross_entropy(hx, y)
    c = torch.mean(cost_i)
    return c


# In[14]:


optimizer = Adam([w,b], lr=0.01)
for epoch in range(1000):
    c = cost()
    optimizer.zero_grad()
    c.backward()
    optimizer.step()
    print('cost=', c.item())


# In[15]:


w


# In[31]:


def hxFn(xd):
    xd = torch.FloatTensor(xd)
    z = torch.matmul(xd, w) + b
    hx = torch.sigmoid(z)
#     return z, hx
#     return ((hx > 0.5) + 0).detach().item() # True + 1 = 2 , True + 0 = 1, False + 1 = 1, False + 0 = 0
    return ((hx > 0.5) + 0).detach().numpy()


# In[33]:


hxFn([[4,5]])


# In[35]:


pred = hxFn(x)
pred


# In[39]:


(y.numpy() == pred).mean() # 정확도

