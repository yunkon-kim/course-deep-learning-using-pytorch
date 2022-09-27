#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston


# ### Multi feature
# 1) 상관 관계: corr (상관 관계가 낮은 컬림은 제거)   
# 2) 다중공선성: statesmodel (혈중 알콜 농도와 음주운전 처럼 매우 높은 상관 관계를 갖는것 중 하나 선택)   
# 3) **정규화(무조건)**   
# 4) 검정: 연속데이터(결정계수), 분류(주로 f1 score(그 외 정확도), ROC 커브, confusion matrix)   

# In[3]:


data = [[828, 920, 1234567, 1020, 1111],
            [824, 910, 2345612, 1090, 1234],
            [880, 900, 3456123, 1010, 1000],
            [870, 990, 2312123, 1001, 1122],
            [860, 980, 3223123, 1008, 1133],
            [850, 970, 2432123, 1100, 1221]]
df = pd.DataFrame( data )
df


# In[56]:


df.corr()


# In[58]:


get_ipython().system('pip install seaborn')


# In[59]:


import seaborn as sb


# In[65]:


sb.heatmap(df.corr(), vmin = -1, vmax = 1, annot=True, 
           linewidths=0.2, cmap='seismic')
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
plt.show()


# In[7]:


x_data = df.iloc[:,:-1]
y_data = df.iloc[:,[-1]]


# In[9]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[29]:


scaleF = MinMaxScaler()
x_dataN = scaleF.fit_transform(x_data)


# In[30]:


x_dataN


# In[31]:


scaleL = MinMaxScaler()
y_dataN = scaleL.fit_transform(y_data)


# In[32]:


y_dataN


# In[51]:


x = torch.FloatTensor(x_dataN)
y = torch.FloatTensor(y_dataN)
w = torch.empty([4,1], requires_grad=True)
b = torch.empty(1, requires_grad=True)
torch.nn.init.uniform_(w)
torch.nn.init.uniform_(b)


# In[52]:


def cost():
    hx = torch.matmul(x, w) + b
    c = torch.mean((hx-y)**2)
    return c


# In[53]:


hist = []
optimizer = Adam([w, b], lr = 0.01)
for epoch in range(1000):
    c = cost()
    optimizer.zero_grad()
    c.backward()
    optimizer.step()
    print(epoch, c.item())
    hist.append(c.item())


# In[54]:


plt.plot(hist)
plt.show()


# In[36]:


w


# In[37]:


b


# In[38]:


def hxFn(xd):
    xd = torch.FloatTensor(xd)
    hx = torch.matmul(xd, w) + b
    return hx.detach().numpy()


# In[44]:


# 828, 920, 1234567, 1020
# hxFn([[828, 920, 1234567, 1020]]) X
# 정규화된 값을 입력해야함

xn = scaleF.transform([[828, 920, 1234567, 1020]])
p = hxFn(xn)
p


# In[46]:


scaleL.inverse_transform(p)


# ## w,b검정지표: 결정계수(선형회귀)
# - 0 <= 결정계수 <= 1 (0.5 이상이면 예측으로 활용가능하다.)
# - 1 - (오차의 제곱합)/(편차의 제곱합)
# - 오차: 실제 값과 예측 값의 차이
# - 편차: 실제 값과 평균 값의 차이

# In[47]:


from sklearn.metrics import r2_score


# In[48]:


pred = hxFn(x)


# In[49]:


r2_score(y_dataN, pred)

