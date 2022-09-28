#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('../data/test.csv', header=None) # Quiz1 , Quiz2, Midterm, Final
df.columns = ['q1', 'q2', 'mid', 'final']
df


# In[8]:


#df.iloc[행, 열]
x = torch.FloatTensor(df.iloc[:,:-1].values)
y = torch.FloatTensor(df.iloc[:,[-1]].values)


# In[13]:


# w = torch.empty([행,열], requires_grad=True) # [특성데이터 갯수, 라벨의 갯수]
w = torch.empty([3,1], requires_grad=True)
b = torch.empty(1, requires_grad=True)
torch.nn.init.uniform_(w)
torch.nn.init.uniform_(b)


# In[17]:


def cost():
    hx = torch.matmul(x, w) + b
    c = torch.mean((hx-y)**2)
    return c


# In[18]:


optimizer = Adam([w, b], lr = 0.01)
for epoch in range(1000):
    c = cost()
    optimizer.zero_grad()
    c.backward()
    optimizer.step()
    print(epoch, c.item())


# In[26]:


def hxFn(xd):
    xd = torch.FloatTensor(xd)
    hx = torch.matmul(xd, w) + b
    return hx.detach().numpy()


# In[27]:


w


# In[28]:


b


# q1: 80 q2: 90 mid: 90 최종점수를 예측하시오

# In[30]:


hxFn([[80,90,90]])


# q1: 80 q2: 90 mid: 90   
# q1: 70 q2: 50 mid: 50   
# 최종점수를 예측하시오   

# In[42]:


hxFn([[80,90,90], [70,50,50]])


# In[45]:


pred = hxFn(x)
pred


# In[51]:


x_axis = torch.arange(0, len(df['q1']))
plt.scatter(x_axis, df['final'])
plt.plot(x_axis, pred, 'r--') # r-- : Red, dashed line
plt.show()

