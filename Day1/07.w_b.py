#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# In[2]:


x_data = torch.FloatTensor([1,2,3,4,5])
y_data = torch.FloatTensor([3,5,7,9,11])


# In[5]:


plt.xlim(0, 15)
plt.ylim(0, 10)
plt.scatter(x_data, y_data)
plt.show()


# In[13]:


w = 10 
b = 10
epochs = 3000
learning_rate = 0.01
n = 5


# In[11]:


x = torch.FloatTensor([1,2,3,4,5])
y = torch.FloatTensor([3,5,7,9,11])


# In[14]:


for i in range(epochs):
    hx = w*x + b
    cost = torch.sum((hx-y)**2)/n
    gradientW = torch.sum((hx-y)*2*x)/n
    gradientB = torch.sum((hx-y)*2)/n
    w = w - learning_rate * gradientW
    b = b - learning_rate * gradientB
    print('cost: ', cost, 'w=', w, 'b=',b)
    
print('최종w: ', w, '최종b: ', b)

