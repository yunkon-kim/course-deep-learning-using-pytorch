#!/usr/bin/env python
# coding: utf-8

# In[25]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential, Sigmoid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sb

from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


# 0번째 행값에 대해 당뇨유무 에측하시요.

# In[4]:


x_data = df.iloc[:,:-1]
y_data = df.iloc[:,[-1]].values


# In[5]:


x_data


# In[6]:


y_data


# In[7]:


scaleF = MinMaxScaler()
x_dataN = scaleF.fit_transform( x_data )
x = torch.FloatTensor( x_dataN )
y = torch.FloatTensor( y_data )


# In[8]:


x.shape
feature_len = x.shape[1]
feature_len


# In[9]:


model = Sequential()
model.add_module('nn1', Linear(feature_len,1)) # w:[8,1] b:[1]
model.add_module('sig1', Sigmoid()) # 활성함수
loss_fn = torch.nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.1)


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


p = model.forward(x[0].reshape(1,feature_len))
d = ( p > 0.5 ) + 0
d


# In[13]:


pred = (model.forward(x) > 0.5) + 0
pred


# In[14]:


(y.numpy() == pred.numpy()).mean() # 정확도


# ### 검증

# In[30]:


from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, accuracy_score, precision_score


# In[21]:


con_mat = confusion_matrix(y.numpy(), pred.numpy())
con_mat


# In[29]:


sb.heatmap( con_mat, annot=True, fmt='d', linewidths=0.2, 
           yticklabels=['당뇨아님','당뇨'], xticklabels=['당뇨아님예측','당뇨예측'] )
plt.show()


# #### 정확도

# In[34]:


accuracy_score(y.numpy(), pred.numpy())


# In[36]:


y.numpy()


# In[37]:


pred.numpy()


# In[32]:


f1_score(y.numpy(), pred.numpy())

