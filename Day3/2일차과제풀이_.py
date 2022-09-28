#!/usr/bin/env python
# coding: utf-8

# In[47]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score


# In[48]:


boston = load_boston()
boston


# In[49]:


df = pd.DataFrame(boston['data'])
df.columns = boston['feature_names']
df['medv'] = boston['target']
df


# ### 상관관계

# In[50]:


plt.figure(figsize=(10, 10))
sb.heatmap(df.corr(), vmin = -1, vmax = 1, annot=True, 
           linewidths=0.2, cmap='seismic')
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
plt.show()


# In[51]:


x_data = df.iloc[:,:-1]
y_data = df.iloc[:,[-1]].values


# ### 정규화

# In[52]:


scaleF = MinMaxScaler()
x_dataN = scaleF.fit_transform(x_data)


# In[53]:


x = torch.FloatTensor(x_dataN)
y = torch.FloatTensor(y_data)


# In[54]:


x.shape


# In[55]:


model = Sequential()
model.add_module('nn1', Linear(13, 1))
loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=0.1)


# In[56]:


hist=[]
for step in range(1000):
    optimizer.zero_grad()
    hx = model.forward(x) # matmul (x, w) + b # model(x) 가능
    cost = loss_fn(hx, y)
    cost.backward()
    optimizer.step()
    print(step, cost.item())
    hist.append(cost.item())


# ### Cost의 Plot 차트

# In[57]:


plt.plot(hist)
plt.show()


# #### 정규화 전 데이터

# In[25]:


x_data.values[0]


# #### 정규화 후 데이터

# In[26]:


x_dataN[0]


# ### 0번째 행 예측값

# In[35]:


model(torch.FloatTensor(x_dataN[0].reshape(1,13)))


# ### 결정계수 값

# In[58]:


x_dataN


# In[36]:


pred = model(torch.FloatTensor(x_dataN)).detach().numpy()


# In[38]:


r2_score(y_data, pred)

