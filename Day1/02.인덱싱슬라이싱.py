#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch


# In[5]:


t = torch.tensor([11,22,33,44,55])


# # 인덱싱

# In[8]:


t[0].item()


# In[10]:


t[-1].item()


# ## 복수개 인덱싱
# - [[인덱스,....]]

# In[24]:


t[[1,3,4]]


# #### 슬라이싱
# - [시작인덱스:끝인덱스:증가치]
# - 시작인덱스 <= index < 끝인덱스

# In[11]:


t[1:3:1] # 1<=idx<3 1,2


# In[14]:


t[1:3:1].numpy()


# In[15]:


t[1:3]


# In[16]:


t[2:]


# ### 2차원 인덱싱 슬라이싱
# - [행]
# - [행, 열]
# 
# ### 3차원 인덱싱 슬라이싱
# - [면]
# - [면, 행]
# - [면, 행, 열]

# In[18]:


tt = torch.tensor([[11,22],[33,44],[55,66]]) # 3 x 2
tt


# In[19]:


tt[0]


# In[21]:


tt[0,1]


# In[23]:


tt[1:]


# In[22]:


tt[1:, 1:]

