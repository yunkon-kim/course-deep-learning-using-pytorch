#!/usr/bin/env python
# coding: utf-8

# #### 문제 1
# 
# 1) x가 7인경우 예측값을 구하시요
# 2) 실제값 scatter와 예측값 라인차트를 그리시요

# In[50]:


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# In[51]:


x = torch.FloatTensor([1,2,3,4,5])
y = torch.FloatTensor([5,8,11,13,17])


# In[52]:


plt.xlim(0, 15)
plt.ylim(0, 30)
plt.scatter(x, y)
plt.show()


# In[53]:


w = 10 
b = 10
epochs = 3000
learning_rate = 0.01
n = 5


# In[54]:


for i in range(epochs):
    hx = w*x + b
    cost = torch.sum((hx-y)**2)/n
    gradientW = torch.sum((hx-y)*2*x)/n
    gradientB = torch.sum((hx-y)*2)/n
    w = w - learning_rate * gradientW
    b = b - learning_rate * gradientB
    print('cost: ', cost, 'w=', w, 'b=',b)
    
print('최종w: ', w, '최종b: ', b)

estimated = w * 7 + b
print('Estimated: ', estimated)


# #### Estimated:  tensor(22.3998)
# 

# In[55]:


for x in np.linspace(-1, 10, 50):
    e = w * x + b
#     print('x', w, 'cost=', c)
    plt.plot(x, e, 'ro') # r: red, o: circle marker # plt.scatter(w,c)
plt.xlabel('x')
plt.ylabel('estimated')
plt.show()


# In[56]:


x_gen = torch.arange(-1, 10)
print(x_gen)
y_est = w * x_gen + b
print(y_est)

plt.plot(x_gen, y_est) # r: red, o: circle marker # plt.scatter(w,c)
plt.xlabel('x')
plt.ylabel('estimated')
plt.show()


# #### 문제 2
# 2. y= (x+3)^3 인경우 미분과정을 적으시요

# 1. x+3을 t로 치환합니다.
# 2. t에 대해 미분합니다 --> 3*t^2와 같이 지수를 상수로 곱하고, 지수를 하나뺍니다. 
# 3. t를 본래 x+3로 치환합니다. --> 3(x+3)^2
