#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
import matplotlib.pyplot as plt


# In[6]:


print(2**2)
print(2**-2)
print(2**3)
print(2**-3)


# In[8]:


math.e


# In[17]:


3.141592e2 # e2 = 10^2


# In[18]:


3.141592e-2 # e-2 = 1/10^2


# ### sigmoid 함수는
# - 음수이면 1
# - 0: 0.5
# - 양수이면 1

# In[19]:


def sigmoid(z):
    return 1/(1+math.e**-z)


# In[20]:


print(sigmoid(-10))
print(sigmoid(-1))
print(sigmoid(0))
print(sigmoid(1))
print(sigmoid(10))


# In[24]:


for z in np.linspace(-10,10,50):
    s = sigmoid(z)
    print('z=', z, 'sigmoid=',s)
    plt.plot(z, s, 'ro')
plt.show()

