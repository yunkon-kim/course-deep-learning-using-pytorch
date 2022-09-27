#!/usr/bin/env python
# coding: utf-8

# In[ ]:


s = 11 # 3.14, 'abc'
v = [11, 22, 33]
m = [[11, 22, 33], [110, 220, 330]] # 2 x 3 : 2개의 vector, 각 vector에는 3개의 scalar
n3 = [[[11, 22, 33], [110, 220, 330]], [[11, 22, 33], [110, 220, 330]]] # 2 x 2 x 3 
n4 = [[[[11, 22, 33], [110, 220, 330]], [[11, 22, 33], [110, 220, 330]]],[[[11, 22, 33], [110, 220, 330]], [[11, 22, 33], [110, 220, 330]]]] # 2 x 2 x 2 x 3

# 위 모든 것이 tensor (e.g., scalar tensor, vector tensor, matrix tensor, 3-Tensor, 4-Tensor, and etc.)


# In[1]:


import torch


# In[3]:


tt = torch.IntTensor([0,1,2,3,4,5])
tt


# In[5]:


print(type(tt))


# In[6]:


tt1 = torch.IntTensor([[0,1],[2,3],[4,5]])
tt1


# In[7]:


tt2 = torch.FloatTensor([[0,1],[2,3],[4,5]])
tt2


# In[10]:


tt3 = torch.DoubleTensor([[0,1],[2,3],[4,5]])
tt3


# In[9]:


tt4 = torch.LongTensor([[0,1],[2,3],[4,5]])
tt4


# In[12]:


tt5 = torch.tensor([0,1,2,3,4,5], dtype=torch.int32)
tt5


# In[ ]:


### Python
객체: 속성 + 메소드
객체.속성, 객체.메소드()


# In[13]:


tt5.numpy() # numpy ndarray


# In[15]:


# tt6 = torch.IntTensor(20) ? scalar node x
tt6 = torch.tensor(20)
tt6


# In[ ]:


# tt5.item() #vector 이상은 사용불가...


# In[16]:


tt6.item() # scalar node 만 기능


# In[17]:


tt6.numpy()


# ### Operation

# In[18]:


a = torch.tensor(5)
b = torch.tensor([1,2,3])


# In[23]:


b[0]


# In[24]:


b[1]


# In[25]:


b[-1]


# In[20]:


c = a.add(b) # torch.add(a,b) #기본산술연산은 +, -, **, *, /, %, //
c


# In[26]:


d = a + b
d


# ## 그래프연산(산술)
# - broad casting: 연산 대상 데이터의 사이즈에 맞추어 자동확장
# - element-wise: 동인한 인덱스 끼리 연산

# In[30]:


#e = torch.tensor([[1],[2]]) # 2 x 1
#e = torch.tensor([1,2,3])
e = torch.tensor([[1],[2]]) # 2 x 1
f = torch.tensor([[1,2,3], [4,5,6]]) # 2 x 3
g = e + f
g


# In[31]:


f[0,0] # [행, 열]


# In[32]:


f[0] # [행]

