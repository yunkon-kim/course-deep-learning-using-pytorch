#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.optim as o


# In[5]:


w = torch.tensor(2.0, requires_grad=True) # 미분을 통해 x 수정할 수 있는 Tensor 객체

y = 2*w**2
y.backward() # 미분... 이후에 왜 이 함수 이름이 backward인지 알 수 있음
print(w.grad)

# 여기서 초기화 하지 않으면, 아래 위 미분 값이 합산됨
y = 2*w**2
y.backward() # 미분... 이후에 왜 이 함수 이름이 backward인지 알 수 있음
print(w.grad)

y = 2*w**2
y.backward() # 미분... 이후에 왜 이 함수 이름이 backward인지 알 수 있음
print(w.grad)


# In[6]:


x = torch.FloatTensor([1,2,3,4,5])
y = torch.FloatTensor([3,5,7,9,11])


# In[9]:


w = torch.tensor(10.0, requires_grad=True)
b = torch.tensor(10.0, requires_grad=True)


# In[10]:


optimizer = o.Adam([w,b], lr=0.1)
for i in range(2000):
    hx = w*x+b
    cost = torch.mean((hx-y)**2)
    optimizer.zero_grad() # 초기화 (미분값이 더해지는 것 방지)
    cost.backward() # 미분 w, b
    optimizer.step() # w = w - 0.1 * 미분값, b = b - 0.1 * 미분값
    print(i, 'cost=', cost.item())


# In[12]:


w


# In[13]:


b

