#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[3]:


m1 = torch.FloatTensor([[1,2], [3,4]])
m2 = torch.FloatTensor([[3],[4]])


# In[4]:


m1.shape


# In[5]:


m2.shape


# In[6]:


m1.size() 


# In[7]:


m2.size()


# In[8]:


m1.dim()


# In[9]:


m2.dim()


# In[11]:


m1.add(m2)


# In[12]:


torch.add(m1, m2)


# In[13]:


m1 + m2


# In[14]:


m1.sub(m2)


# In[15]:


torch.sub(m1,m2)


# In[16]:


m1-m2


# In[18]:


m1.mul(m2)


# In[19]:


torch.mul(m1,m2)


# In[20]:


m1*m2


# In[22]:


m1.square()


# In[25]:


torch.square(m1)


# In[26]:


m1**2


# ### 행렬곱
# - 선형대수의 행렬연산
# - matrix 간 연산
# - 앞의 열과 뒤의 행이 같아야한다.
# - 앞의 행과 뒤의 열로 결과

# In[27]:


m1.mul(m2)


# In[29]:


m1.matmul(m2) # torch.matmul(m1, m2)


# ### 통계함수
# - dim: None 전체 (1차원 풀어서)
# - dim: 0 컬럼별 통계
# - dim: 1 행별 통계

# In[40]:


m1


# In[41]:


m1.mean() # torch.mean(m1)


# In[42]:


m1.mean(dim=0)


# In[43]:


m1.mean(dim=1)


# In[44]:


m1.sum()


# In[45]:


m1.sum(dim=0) # torch.sum(m1, dim=0)


# In[46]:


m1.sum(dim=1)


# In[49]:


m1.max()


# In[50]:


m1.max(dim=0)


# In[52]:


## 분류
m1.argmax()


# In[53]:


m1.argmax(dim=0)


# In[54]:


m1.argmax(dim=1) # dim=1 이 앞으로 아주 많이 나올 중요한 부분!!!!!!! (행별 가장 큰 값의 인덱스)

