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

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# In[2]:


softFn = Softmax(dim=1) # 행별 확률값
softFn1 = Softmax(dim=0) # 컬럼별 확률값


# In[3]:


r = torch.randn(2,3) # normalized 
r


# In[4]:


softFn(r)


# In[5]:


softFn1(r)


# In[6]:


df = pd.read_csv('../data/softmax.txt', sep=' ', header=None)
df


# onehot encoding 0 0 1 => A,  0 1 0 => B, 1 0 0 => C

# In[30]:


x_data = df.iloc[:,0:2].values
y_data = df.iloc[:,2:].values


# In[31]:


x = torch.FloatTensor(x_data)
y = torch.FloatTensor(y_data)


# In[32]:


w = torch.empty([2,3], requires_grad=True) # [특성데이터 갯수, 라벨의 갯수]
b = torch.empty([3], requires_grad=True) # 라벨의 겟수
torch.nn.init.uniform_(w)
torch.nn.init.uniform_(b)


# In[39]:


y


# In[33]:


def cost():
    z = torch.matmul(x, w) + b
#     hx = F.softmax(z, dim=1)
    cost_i = F.cross_entropy(z, y) # cross_entropy에는 softmax가 포함되어 있음
    # xxxx c= torch.mean((hx-y)**2) 
    c = torch.mean(cost_i)
    return c


# In[34]:


optimizer = Adam([w,b], lr=0.01)
for epoch in range(1000):
    c = cost()
    optimizer.zero_grad()
    c.backward()
    optimizer.step()
    print('cost: ', c.item())


# In[35]:


def hxFn(xd):
    xd = torch.FloatTensor(xd)
    z = torch.matmul(xd, w) + b
    hx = torch.softmax(z, dim=1)
    return hx.argmax(dim=1), hx
#     return z, hx


# In[36]:


hxFn([[2,3]])


# In[38]:


## 6,6 2,2

hxFn([[6,6], [2,2]])

