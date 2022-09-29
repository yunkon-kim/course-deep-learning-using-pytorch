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


## 수치 맞추는 것 잘 볼것
## 아래처럼 딥하고 와이드하게 잡는 방법이 있음
## 얼마나 딥하고 와이드하게 할지는 튜닝해봐야함
w1 = torch.empty([2,20], requires_grad=True)
w2 = torch.empty([20,1], requires_grad=True)
# w1 = torch.empty([2,100], requires_grad=True)
# w2 = torch.empty([100,50], requires_grad=True)
# w3 = torch.empty([50,1], requires_grad=True)
b1 = torch.empty([20], requires_grad=True)
b2 = torch.empty([1], requires_grad=True)
torch.nn.init.uniform_(w1)
torch.nn.init.uniform_(w2)
torch.nn.init.uniform_(b1)
torch.nn.init.uniform_(b2)


# In[5]:


def cost():
    z1 = torch.matmul(x, w1) + b1
    hx1 = torch.sigmoid(z1)
    
    z = torch.matmul(hx1, w2) + b2
    hx = torch.sigmoid(z)
    cost_i = F.binary_cross_entropy(hx, y)
    
#     cost_i = F.binary_cross_entropy_with_logits(z2,y)    
    c = torch.mean(cost_i)
    return c


# In[6]:


hist = []
optimizer = Adam( [w1,w2,b1,b2], lr=0.01)
for epoch in range(2000):
    c = cost()
    optimizer.zero_grad()
    c.backward()
    optimizer.step()
    print(c.item())
    hist.append(c.item())


# In[7]:


plt.plot(hist)
plt.show()


# In[8]:


w1


# In[9]:


w2


# In[10]:


b1


# In[11]:


b2


# In[12]:


def hxFn(xd):
    xd = torch.FloatTensor(xd)
    z1 = torch.matmul(xd, w1) + b1
    hx1 = torch.sigmoid(z1)
    
    z = torch.matmul(hx1, w2) + b2
    hx = torch.sigmoid(z)
    return (hx.detach().numpy() > 0.5) + 0


# In[13]:


hxFn(x) # 4x2 2x20 (w2) = 4x20 20x1 = 4x1

