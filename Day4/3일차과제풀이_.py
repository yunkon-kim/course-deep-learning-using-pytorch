#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# In[3]:


iris = load_iris()
iris


# In[4]:


iris.keys()


# In[5]:


print(iris['DESCR'])


# In[6]:


df = pd.DataFrame( iris['data'] )
df.columns = iris['feature_names']
df['species'] = iris['target']
df


# In[7]:


df.corr()


# In[8]:


sb.heatmap( df.corr(), vmin=-1, vmax=1, annot=True,
           linewidths=0.2, cmap='seismic' )
plt.show()


# In[9]:


x_data = df.iloc[:,:-1].values
y_onehot = pd.get_dummies(df['species'])
y_data = y_onehot.values


# In[10]:


scaleF = MinMaxScaler()
x_dataN = scaleF.fit_transform( x_data )


# In[11]:


x = torch.FloatTensor(x_dataN)
y = torch.FloatTensor(y_data)


# In[13]:


model = Sequential()
model.add_module('nn1', Linear(4,3)) # (특성갯수, 라벨 갯수) w: 특성갯수 x 라벨 갯수, b: 라벨 갯수
model.add_module('softmax', Softmax(dim=1)) # 활성함수
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.1)


# In[14]:


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


# In[15]:


plt.plot(hist)
plt.show()


# In[17]:


p = model.forward(x)


# In[18]:


y_pred = p.argmax(dim=1).numpy()
y_pred


# In[19]:


y = y.argmax(dim=1).numpy()
y


# In[21]:


con_mat = confusion_matrix(y, y_pred)


# In[23]:


sb.heatmap( con_mat, annot=True, fmt='d', linewidths=0.2)
plt.show()


# In[24]:


accuracy_score(y, y_pred)


# In[25]:


f1_score(y, y_pred, average='macro')

