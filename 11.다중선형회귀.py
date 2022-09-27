#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


x1 = torch.FloatTensor([1,0,3,4,5]) # 공부한 시간
x2 = torch.FloatTensor([0,1,2,3,4]) # 출석 일수
y = torch.FloatTensor([1,2,3,4,5]) # 출석 점수


# In[3]:


w1 = torch.empty(1, requires_grad=True)
w2 = torch.empty(1, requires_grad=True)
b = torch.empty(1, requires_grad=True)
torch.nn.init.uniform_(w1)
torch.nn.init.uniform_(w2)
torch.nn.init.uniform_(b)


# In[4]:


def cost():
    hx = w1*x1 + w2*x2 +b
    c = torch.mean((hx-y)**2)
    return c


# In[7]:


optimizer = Adam([w1, w2, b], lr = 0.01)
for epoch in range(1000):
    c = cost()
    optimizer.zero_grad()
    c.backward()
    optimizer.step()
    print(epoch, c.item())


# In[9]:


w1


# In[10]:


w2 


# In[11]:


b


# In[12]:


w1*5 + w2*5 + b

