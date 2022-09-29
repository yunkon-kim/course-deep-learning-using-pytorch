#!/usr/bin/env python
# coding: utf-8

# 1. 정규화
# 2. Train set, test set
# 3. 딥러닝학습
# 4. train f1_score, test f1_score

# In[1]:


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


# In[2]:


df = pd.read_csv('../data/pima-indians-diabetes.data.csv')
df


# In[3]:


plt.figure(figsize=(10,10))
sb.heatmap( df.corr(), vmin=-1, vmax=1, annot=True,
           linewidths=0.2, cmap='seismic' )
plt.show()


# In[4]:


x_data = df.iloc[:,:-1]
y_data = df.iloc[:,[-1]].values


# In[5]:


scaleF = MinMaxScaler()
x_dataN = scaleF.fit_transform( x_data )

x_train, x_test, y_train, y_test = train_test_split(x_dataN, y_data, 
                                                    test_size=0.3, stratify=y_data) # stratify=y_data  label값이 골고루 분포되도록


# In[6]:


# y_test


# In[7]:


x_train.shape
feature_len = x_train.shape[1]


# In[8]:


x = torch.FloatTensor(x_train)
y = torch.FloatTensor(y_train)


# In[9]:


model = Sequential()
model.add_module('nn1', Linear(feature_len, 64)) # w:[8,1] b:[1]
model.add_module('sig1', ReLU()) # 활성함수
model.add_module('nn2', Linear(64, 32)) # w:[8,1] b:[1]
model.add_module('sig2', ReLU()) # 활성함수
model.add_module('nn3', Linear(32,1)) # w:[8,1] b:[1]
model.add_module('sig3', Sigmoid()) # 활성함수
loss_fn = torch.nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.01)


# In[10]:


hist=[]
for epoch in range(1000):
    optimizer.zero_grad()
    hx = model.forward(x) 
    # z = torch.matmul(x,w)+b
    # hx = Sigmoid(z)
    cost = loss_fn(hx,y)
    cost.backward()
    optimizer.step()
    print('cost=',cost.item())
    hist.append(cost.item())


# In[11]:


plt.plot(hist)
plt.show()


# In[12]:


train_real = y_train
# train_real


# In[13]:


train_pred = ( model.forward(x) > 0.5 ) + 0
# train_pred


# In[14]:


f1_score(train_real, train_pred)


# In[15]:


test_real = y_test
# test_real


# In[16]:


test_pred = ( model.forward(torch.FloatTensor(x_test)) > 0.5 ) + 0
# test_pred


# In[17]:


f1_score(test_real, test_pred)


# In[18]:


y_test[0]


# In[19]:


(model.forward(torch.FloatTensor(x_test[0].reshape(1,8))) > 0.5) + 0
# 8x64 64x32 32x1
# 1x8 8x64  1x64 64x32  1x32 32x1  1x1


# In[ ]:




