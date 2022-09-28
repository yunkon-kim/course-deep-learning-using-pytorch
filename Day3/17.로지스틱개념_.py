#!/usr/bin/env python
# coding: utf-8

# In[27]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F


# In[41]:


# x_data = torch.FloatTensor([1,2,3]) # 출석일수
# y_data = torch.FloatTensor([1,2,3]) # 합격여부
x_data = torch.FloatTensor([1,2,3,4,5,6,7,8,9,10]) # 출석일수
y_data = torch.FloatTensor([0,0,0,0,0,1,1,1,1,1]) # 합격여부


# In[45]:


def cost(x, y, w):
#     hx = w*x + b
    z = w*x
    hx = torch.sigmoid(z)
#     cost_i = -(y*torch.math.log(hx)+(1-y)*torch.math.log(1-hx))
#     cost_i = F.binary_cross_entropy(hx,y)
    cost_i = F.binary_cross_entropy_with_logits(z,y) # sigmoid가 안에 들어가 있음
    c = torch.mean( cost_i )
#    c = torch.mean( (hx-y)**2 )
    return c


# In[46]:


cost (x_data, y_data, 3)


# In[47]:


for w in np.linspace(-3,5,50):
    c = cost(x_data, y_data, w)
    plt.plot(w,c,'ro')
plt.show()


# In[ ]:




