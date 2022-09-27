#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install matplotlib')


# In[16]:


import matplotlib.pyplot as plt
import numpy as np


# In[4]:


x_data = [1,2,3]
y_data = [1,2,3]


# In[11]:


plt.xlim(0,5)
plt.ylim(0,5)
plt.scatter(x_data,y_data)
plt.show()


# In[15]:


for i in range(3): # [0,1,2]
    print(i)


# In[12]:


def cost(x,y,w):
    c=0
    for i in range (len(x)):
        hx = w*x[i]
        c = c + (hx-y[i])**2
    return c/len(x)


# In[13]:


print(cost(x_data,y_data, -1))
print(cost(x_data,y_data, 0))
print(cost(x_data,y_data, 1))
print(cost(x_data,y_data, 2))


# In[18]:


np.linspace(1,10) # 1 ~ 10 사이에 동일한 간격의 50개의 값을 생성함


# In[20]:


np.linspace(-3,5,50)


# In[24]:


for w in np.linspace(-3, 5, 50):
    c = cost(x_data, y_data, w)
    print('w', w, 'cost=', c)
    plt.plot(w, c, 'ro') # r: red, o: circle marker # plt.scatter(w,c)
    
plt.xlabel('w')
plt.ylabel('cost(mse)')
plt.show()


# w값 임의의 값을 하나 부여
# 
# - y = 2 미분 0 (상수의 미분)
# - y = x 미분 1
# - y = 2x 미분 2
# - y = x^2 미분 2x
# - y = x^3 미분 3x^2
# 
# w = w - 러닝레이트 * 미분
# 반복한다.

# In[ ]:




