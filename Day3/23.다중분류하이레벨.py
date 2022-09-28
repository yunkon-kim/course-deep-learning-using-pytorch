#!/usr/bin/env python
# coding: utf-8

# In[23]:


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


# In[24]:


df = pd.read_csv('../data/softmax.txt', sep=' ', header=None)
df


# In[25]:


x_data = df.iloc[:,0:2].values
y_data = df.iloc[:,2:].values


# In[26]:


x = torch.FloatTensor(x_data)
y = torch.FloatTensor(y_data)


# In[27]:


model = Sequential()
model.add_module('nn1', Linear(2,3)) # w: 2x3, b: 3
model.add_module('softmax', Softmax(dim=1)) # 할성함수
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)


# In[28]:


for epoch in range(2000):
    optimizer.zero_grad()
    hx = model.forward(x) 
    # z = torch.matmul(x,w)+b
    # hx = Softmax(z)
    cost = loss_fn(hx,y)
    cost.backward()
    optimizer.step()
    print(epoch, cost.item())


# In[29]:


model[0].weight


# In[30]:


model[0].bias


# In[31]:


p = model.forward(torch.FloatTensor([[5,6]]))
p.argmax(dim=1)


# In[34]:


p = model.forward(torch.FloatTensor([[6,6], [2,2]]))
p.argmax(dim=1)

