#!/usr/bin/env python
# coding: utf-8

# ### 과제
# 1. 상관관계, heatmap
# 2. 정규화후 학습
# 3. cost 값 hist plot 차트로
# 4. 검정
# - confusion matrix, heatmap
# - 정확도
# - f1 score

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


# ### 1. 상관관계 Heatmap

# In[7]:


sb.heatmap( df.corr(), vmin=-1, vmax=1, annot=True,
           linewidths=0.2, cmap='seismic' )
plt.show()


# In[8]:


x_data = df.iloc[:,:-1].values
y_onehot = pd.get_dummies(df['species'])
y_data = y_onehot.values


# ### 2. 정규화

# In[9]:


scaleF = MinMaxScaler()
x_dataN = scaleF.fit_transform( x_data )


# In[10]:


x = torch.FloatTensor(x_dataN)
y = torch.FloatTensor(y_data)


# In[11]:


model = Sequential()
model.add_module('nn1', Linear(4,3)) # (특성갯수, 라벨 갯수) w: 특성갯수 x 라벨 갯수, b: 라벨 갯수
model.add_module('softmax', Softmax(dim=1)) # 활성함수
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


# ### 3. cost 값 hist plot 차트로

# In[13]:


plt.plot(hist)
plt.show()


# In[14]:


p = model.forward(x)


# In[15]:


y_pred = p.argmax(dim=1).numpy()
y_pred


# In[16]:


y_true = df['species'].values
y_true


# #### 4. 검정
# - confusion matrix, heatmap

# In[17]:


con_mat = confusion_matrix(y_true, y_pred)


# In[18]:


iris['target_names']


# In[19]:


y_labels = iris['target_names']
x_labels = ['pred_setosa', 'pred_versicolor', 'pred_virginica']


# In[20]:


sb.heatmap( con_mat, annot=True, fmt='d', linewidths=0.2, 
           yticklabels=y_labels, xticklabels=x_labels )
plt.show()


# - 정확도

# In[21]:


accuracy_score(y_true, y_pred)


# - f1 score

# In[22]:


f1_score(y_true, y_pred, average='macro')

