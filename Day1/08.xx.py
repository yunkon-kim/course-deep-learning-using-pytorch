#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.optim as o


# In[4]:


w = torch.tensor(2.0, requires_grad=True) # 미분을 통해 수정할 수 있는 Tensor 객체

y = 2*w
y.backward() # 미분... 이후에 왜 이 함수 이름이 backward인지 알 수 있음
print(w.grad)

