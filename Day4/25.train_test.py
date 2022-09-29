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


iris = load_iris()


# In[3]:


df = pd.DataFrame( iris['data'] )
df.columns = iris['feature_names']
df['species'] = iris['target']
df


# In[4]:


x_data = df.iloc[:,:-1].values
y_onehot = pd.get_dummies(df['species']) # onehot encoding
y_data = y_onehot.values


# In[5]:


x_data.shape


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
                                                    test_size=0.3, stratify=y_data) # stratify=y_data  label값이 골고루 분포되도록


# In[7]:


x_train.shape


# In[8]:


x_test.shape


# In[9]:


y_train.shape


# In[10]:


y_test.shape


# In[11]:


x = torch.FloatTensor(x_train)
y = torch.FloatTensor(y_train)


# In[12]:


model = Sequential()
model.add_module('nn1', Linear(4,3)) # (특성갯수, 라벨 갯수) w: 특성갯수 x 라벨 갯수, b: 라벨 갯수
model.add_module('softmax', Softmax(dim=1)) # 활성함수
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.1)


# In[13]:


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


# In[14]:


plt.plot(hist)
plt.show()


# In[15]:


r = y.argmax(dim=1).numpy()


# In[16]:


pred = model.forward(x).argmax(dim=1).detach().numpy()


# In[17]:


f1_score(r, pred, average='macro')


# In[18]:


treal = y_test.argmax(axis=1)


# In[19]:


tpred = model(torch.FloatTensor(x_test)).argmax(dim=1).detach().numpy()


# In[20]:


f1_score(treal, tpred, average='macro')


# train ==(유사) test : 적합
# 
# train > test: 과적합 (학습횟수 줄임, early stop, 파라미터 튜닝)
#     
# train < test: 과소적합 (학습횟수 늘림, 파라미터 튜닝)
