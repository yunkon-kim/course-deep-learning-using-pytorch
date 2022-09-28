#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_boston


# In[2]:


boston = load_boston()
boston


# In[3]:


df = pd.DataFrame( boston['data'] )
df.columns = boston['feature_names']
df['medv'] = boston['target']
df


# In[4]:


df.corr()


# In[5]:


import seaborn as sb


# ### 상관관계 히트맵

# In[6]:


plt.figure(figsize=(10,10))
sb.heatmap( df.corr(), vmin=-1, vmax=1, annot=True,
           linewidths=0.2, cmap='seismic' )
plt.show()


# In[7]:


x_data = df.iloc[:,:-1]
y_data = df.iloc[:,[-1]].values


# ## 정규화

# In[8]:


scaleF = MinMaxScaler()
x_dataN = scaleF.fit_transform( x_data )


# In[9]:


x= torch.FloatTensor( x_dataN )
y= torch.FloatTensor( y_data )


# In[10]:


x.shape


# In[11]:


model = Sequential()
model.add_module('nn1',Linear(13,1) )
loss_fn = MSELoss()
optimizer = Adam( model.parameters() , lr=0.1)


# In[12]:


hist=[]
for epoch in range(1000):
    optimizer.zero_grad()
    hx = model(x) #matmul( x,w) + b
    cost = loss_fn( hx,y)
    cost.backward()
    optimizer.step()
    print('cost', cost.item() )
    hist.append( cost.item() )


# In[13]:


x_data.values[0]


# In[14]:


x_dataN[0]


# ### 0번째행 예측값확인

# In[15]:


model( torch.FloatTensor(x_dataN[0].reshape(1,13)) )


# In[16]:


from sklearn.metrics import r2_score


# In[17]:


x_dataN.shape


# In[18]:


pred = model( torch.FloatTensor(x_dataN) ).detach().numpy()


# In[19]:


r2_score(y_data, pred)


# #### cost 값 plot차트

# In[20]:


plt.plot(hist)
plt.show()


# 보스톤 
# 1) 상관관계 heatmap
# 2) 정규화후에 학습( 로우레벨, 하이레벨)
# 3) 0번째 행의 예측값을 구하시요
# 4) 결정계수값확인
# 5) cost 의 plot 차트를 그리시요
