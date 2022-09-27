#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


# In[2]:


x = torch.FloatTensor( [1,2,3,4,5] )
y = torch.FloatTensor( [5,8,11,14,17] )


# In[3]:


w=10
b=20
epoch = 2000
learning_rate=0.01
n=5


# In[4]:


for i in range(epoch):
    hx = w*x + b
    cost = torch.sum((hx-y)**2) / n
    gradientw = torch.sum((hx-y)*2*x) / n
    gradientb = torch.sum((hx-y)*2) / n
    w=w-learning_rate*gradientw
    b=b-learning_rate*gradientb
    print('cost:', cost, 'w=', w, 'b=', b)
print('최종w', w, '최종b', b)


# In[5]:


w


# In[8]:


b


# In[6]:


def function(x):
    hx=w*x+b
    return hx


# In[7]:


function(7)


# In[10]:


pred = function(x)
pred


# In[12]:


plt.xlim(0,10)
plt.ylim(0,30)
plt.scatter( x, y, color='c')
plt.plot(x,pred, color='b')
plt.show()


# y = (x+3)^3 미분:
# 
# t = (x+3)
# y = t^3
# 
# dy/dx = dy/dt * dt/dx
#     = 3t^2 * 1
#     = 3(x+3)^2
