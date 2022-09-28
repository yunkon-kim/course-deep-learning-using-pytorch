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

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# In[2]:


iris = load_iris()
iris


# In[3]:


iris.keys()


# In[4]:


print(iris['DESCR'])


# In[5]:


df = pd.DataFrame( iris['data'] )
df.columns = iris['feature_names']
df['species'] = iris['target']
df


# In[6]:


df.corr()


# In[7]:


sb.heatmap( df.corr(), vmin=-1, vmax=1, annot=True,
           linewidths=0.2, cmap='seismic' )
plt.show()


# In[8]:


y_onehot = pd.get_dummies(df['species'])
y_onehot


# In[9]:


x_data = df.iloc[:,:-1].values
y_onehot = pd.get_dummies(df['species'])
y_data = y_onehot.values


# In[10]:


x = torch.FloatTensor(x_data)
y = torch.FloatTensor(y_data)


# In[11]:


model = Sequential()
model.add_module('nn1', Linear(4,3)) # (특성갯수, 라벨 갯수) w: 특성갯수 x 라벨 갯수, b: 라벨 갯수
model.add_module('softmax', Softmax(dim=1)) # 할성함수
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)


# In[12]:


hist=[]
for epoch in range(2000):
    optimizer.zero_grad()
    hx = model.forward(x) 
    # z = torch.matmul(x,w)+b
    # hx = Softmax(z)
    cost = loss_fn(hx,y)
    cost.backward()
    optimizer.step()
    print(epoch, cost.item())
    hist.append(cost.item())


# In[13]:


plt.plot(hist)
plt.show()


# In[15]:


# p = model.forward(x[0].reshape(1,4)) # vector to matrix (1행 4열) 여기서 4는 특성 갯수 
p = model.forward(x[0].reshape(1,-1)) # vector to matrix (1행 4열) 여기서 -1을 쓰면 알아서 맞춰줌
p


# In[18]:


idx = p.argmax(dim=1)


# In[20]:


targetName = iris['target_names']
targetName[idx]

