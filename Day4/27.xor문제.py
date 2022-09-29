#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential, Sigmoid, Softmax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sb

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# In[2]:


x_data =[[0,0],[0,1],[1,0],[1,1]]
y_data =[[0],[1],[1],[0]]


# In[3]:


x = torch.FloatTensor(x_data)
y = torch.FloatTensor(y_data)


# In[4]:


w = torch.empty([2,1], requires_grad=True)
b = torch.empty([1], requires_grad=True)
torch.nn.init.uniform_(w)
torch.nn.init.uniform_(b)


# In[5]:


def cost():
    z = torch.matmul(x,w)+b
    cost_i = F.binary_cross_entropy_with_logits(z,y)
    c = torch.mean(cost_i)
    return c


# In[6]:


optimizer = Adam( [w,b], lr=0.01)


# In[7]:


hist = []
for epoch in range(2000):
    c = cost()
    optimizer.zero_grad()
    c.backward()
    optimizer.step()
    print(c.item())
    hist.append(c.item())


# In[8]:


plt.plot(hist)
plt.show()


# In[9]:


def hxFn(xd):
    xd = torch.FloatTensor(xd)
    z = torch.matmul(xd, w) + b
    hx = torch.sigmoid(z)
    return (hx.detach().numpy() > 0.5) + 0


# In[10]:


hxFn(x)


# In[ ]:




