#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential, Sigmoid, Softmax, ReLU, Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sb

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# In[2]:


x_data =[[0,0],[0,1],[1,0],[1,1]]
y_data =[[0],[1],[1],[0]]


# In[3]:


x = torch.FloatTensor(x_data)
y = torch.FloatTensor(y_data)


# In[4]:


class Test:
    def __init__(self): # 객체생성시 자동호출되는 함수
        print('생성자함수')
        self.a = 100
    def setA(self, a):
        self.a = a
    def getA(self):
        return self.a


# In[5]:


obj = Test()


# In[6]:


obj.getA()


# In[7]:


obj.setA(200)
obj.getA()


# In[8]:


class XORModel(Module):
    def __init__(self):
        super().__init__() # 초기화작업
        self.nn1 = Linear(2, 20)
        self.sig1 = Sigmoid()
        self.nn2 = Linear(20, 1)
        self.sig2 = Sigmoid()
    
    def forward(self, x):
        print('forward call')
#         x = self.nn1.forward(x)
        x = self.nn1(x) # matmul(x, w1) + b1  w1:2x20
        x = self.sig1(x) # 1/1+e^-z
        x = self.nn2(x) # matmul(x, w2) + b2  w2:20x1
        x = self.sig2(x)
        return x    


# In[9]:


model = XORModel()
loss_fn = torch.nn.BCELoss()
hist=[]
optimizer = Adam(model.parameters(), lr=0.01)
for epoch in range(1000):
    optimizer.zero_grad()
    hx = model.forward(x) 
    # z = torch.matmul(x,w)+b
    # hx = Softmax(z)
    cost = loss_fn(hx,y)
    cost.backward()
    optimizer.step()
    print('------------------------------')
    pred = (hx > 0.5) + 0
    print('정확도', torch.mean((y==pred).type(torch.float32)) )
    print(epoch, cost.item())
    hist.append(cost.item())


# In[10]:


model(x) > 0.5


# In[11]:


# model = Sequential()
# model.add_module('nn1', Linear(2,20)) # w1: 2x20 b1: 20 (특성갯수, 라벨 갯수)
# model.add_module('sig1', Sigmoid()) # 활성함수
# #model.add_module('sig1', ReLU()) # 활성함수
# model.add_module('nn2', Linear(20,1)) # w1: 20x1 b1: 1 (특성갯수, 라벨 갯수)
# model.add_module('sig2', Sigmoid()) # 활성함수
# loss_fn = torch.nn.BCELoss()

