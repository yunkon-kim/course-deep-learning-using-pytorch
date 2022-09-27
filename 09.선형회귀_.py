#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')


# In[3]:


import torch
import torch.optim as o
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[6]:


df = pd.read_csv('../data/cars.csv', index_col='Unnamed: 0')
df


# In[8]:


plt.scatter(df['speed'], df['dist'])
plt.show()


# #### 딥러닝에 적합한 데이터?? 상관관계

# In[9]:


df.corr()


# - r이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,
# - r이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,
# - r이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,
# - r이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,
# - r이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,
# - r이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,
# - r이 +0.7과 +1.0 사이이면, 강한 양적 선형관계
# 
# https://ko.wikipedia.org/wiki/%EC%83%81%EA%B4%80_%EB%B6%84%EC%84%9D

# In[13]:


df['speed']


# In[12]:


df['speed'].values


# In[11]:


x = torch.FloatTensor(df['speed'].values) # 특성 데이터
y = torch.FloatTensor(df['dist'].values) # 라벨


# In[18]:


w = torch.empty(1, requires_grad=True)
b = torch.empty(1, requires_grad=True)
torch.nn.init.uniform_(w)
torch.nn.init.uniform_(b)


# In[19]:


def cost():
    hx = w*x+b
    c = torch.mean( (hx-y)**2)
    return c


# In[28]:


optimizer = o.Adam([w,b], lr=0.01)
for i in range(2000):
    c = cost()
    optimizer.zero_grad()
    c.backward()
    optimizer.step()
    print(i, 'cost=', c.item())


# In[29]:


w


# In[30]:


b


# In[35]:


def hxFn(xd):
    xd = torch.tensor(xd, dtype=torch.float32)
    hx = w*xd+b
    return hx


# In[36]:


hxFn(10)


# In[37]:


hxFn([15,20])


# In[39]:


pred = hxFn(x)


# In[40]:


pred


# In[43]:


plt.scatter(df['speed'], df['dist'])
plt.plot(df['speed'], pred.detach().numpy(), 'r--') # r-- : Red, dashed line
plt.show()


# ### 퀴즈
# - electric.csv 파일을 읽어서 전기 생산량, 전기 사용량
# - 전기생산량이 4.5인경우 전기 사용량을 예측하시요
# - 전체데이터 scatter, 예측선을 그리시요.

# In[81]:


df = pd.read_csv('../data/electric.csv', index_col='Unnamed: 0')
df


# In[82]:


plt.scatter(df['전기생산량'], df['전기사용량'])
plt.show()


# In[83]:


df.corr()


# In[84]:


x = torch.FloatTensor(df['전기생산량'].values) # 특성 데이터
y = torch.FloatTensor(df['전기사용량'].values) # 라벨


# In[85]:


w = torch.empty(1, requires_grad=True)
b = torch.empty(1, requires_grad=True)
torch.nn.init.uniform_(w)
torch.nn.init.uniform_(b)


# In[93]:


def cost():
    hx = w*x+b
    c = torch.mean( (hx-y)**2)
    return c


# In[94]:


optimizer = o.Adam([w,b], lr=0.01)
for i in range(2000):
    c = cost()
    optimizer.zero_grad()
    c.backward()
    optimizer.step() # 미분 + 추가기능
    print(i, 'cost=', c.item())


# In[95]:


w


# In[96]:


b


# In[108]:


def hxFn(xd):
    xd = torch.tensor(xd, dtype=torch.float32)
    hx = w*xd+b
    return hx.detach().numpy()


# In[109]:


hxFn(4.5)


# In[110]:


pred = hxFn(x)


# In[111]:


pred


# In[113]:


#plt.cla()
#plt.clf()
plt.scatter(df['전기생산량'], df['전기사용량'])
plt.plot(df['전기생산량'], pred, 'r--') # r-- : Red, dashed line
plt.show()

