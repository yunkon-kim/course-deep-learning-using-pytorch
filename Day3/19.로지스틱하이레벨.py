#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[8]:


model = Sequential()
model.add_module('nn1', Linear(2,1)) # w:[2,1] b:[1]
model.add_module('sig1', Sigmoid()) # 활성함수
loss_fn = torch.nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.1)


# In[9]:


for epoch in range(1000):
    optimizer.zero_grad()
    hx = model.forward(x) 
    # z = torch.matmul(x,w)+b
    # hx = Sigmoid(z)
    cost = loss_fn(hx,y)
    cost.backward()
    optimizer.step()
    print('cost=',cost.item())


# In[10]:


model.forward(torch.FloatTensor([[5,5]]))


# In[15]:


pred = (model.forward(x) > 0.5) + 0
pred

