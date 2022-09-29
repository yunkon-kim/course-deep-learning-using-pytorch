#!/usr/bin/env python
# coding: utf-8

# deep learning이 등장하게된 배경은 xor 문제를 해결하기 위해서

# In[2]:


import numpy as np


# In[10]:


data=[[1,1],[1,0],[0,1],[0,0]]


# In[11]:


def common(w1,w2, theta, x1,x2):
    value = w1*x1 + w2*x2
    print('value: ', value)
    return value > theta


# In[19]:


def AND(x1, x2):
    return common (0.5, 0.5, 0.5, x1, x2)


# In[20]:


def OR(x1, x2):
    return common (0.5, 0.5, 0.2, x1, x2)


# In[21]:


def NAND(x1, x2):
    return common (-0.5, -0.5, -0.7, x1, x2)


# In[22]:


def XOR(x1, x2):
    y1 = OR(x1,x2)
    y2 = NAND(x1,x2)
    return AND(y1, y2)


# In[23]:


def show_operation(op):
    for x1, x2 in data:
        print(op(x1,x2))
    


# In[14]:


show_operation(AND)


# In[16]:


show_operation(OR)


# In[24]:


show_operation(NAND)


# In[25]:


show_operation(XOR)

