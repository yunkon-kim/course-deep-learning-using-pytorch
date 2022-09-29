#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential, Sigmoid, Softmax, ReLU
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


# In[7]:


x_data =[[0,0],[0,1],[1,0],[1,1]]
y_data =[[0],[1],[1],[0]]


# In[8]:


x = torch.FloatTensor(x_data)
y = torch.FloatTensor(y_data)


# In[24]:


model = Sequential()
model.add_module('nn1', Linear(2,20)) # w1: 2x20 b1: 20 (특성갯수, 라벨 갯수)
model.add_module('sig1', Sigmoid()) # 활성함수
#model.add_module('sig1', ReLU()) # 활성함수
model.add_module('nn2', Linear(20,1)) # w1: 20x1 b1: 1 (특성갯수, 라벨 갯수)
model.add_module('sig2', Sigmoid()) # 활성함수
loss_fn = torch.nn.BCELoss()


# In[25]:


list(model.parameters())


# In[29]:


hist=[]
optimizer = Adam(model.parameters(), lr=0.01)
for epoch in range(2000):
    optimizer.zero_grad()
    hx = model.forward(x) 
    # z = torch.matmul(x,w)+b
    # hx = Softmax(z)
    cost = loss_fn(hx,y)
    cost.backward()
    optimizer.step()
    print(cost.item())
    hist.append(cost.item())


# In[30]:


plt.plot(hist)
plt.show()


# In[32]:


model[0].weight


# In[33]:


model[2].weight


# In[34]:


# matmul(x,w1) + b1 => Sigmoid() => matmul(hx1,w2) +b2 => Sigmoid()
# 4x2 2x20 -> 4x20 -> Sig => 4x20 20x1 -> 4x1 -> Sig
(model.forward(x)  > 0.5 ) + 0


# In[ ]:




