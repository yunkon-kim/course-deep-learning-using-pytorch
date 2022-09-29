#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential, Sigmoid, Softmax, ReLU
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score
import seaborn as sb
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# In[2]:


df = pd.read_csv('../data/pima-indians-diabetes.data.csv')
df


# In[3]:


x_data = df.iloc[:,:-1]
y_data = df.iloc[:,[-1]].values


# ## 정규화

# In[4]:


scaleF = MinMaxScaler()
x_dataN = scaleF.fit_transform( x_data )


# ## train test

# In[5]:


x_train, x_test, y_train, y_test = train_test_split(x_dataN,y_data, test_size=0.3, stratify=y_data)


# In[6]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[7]:


x= torch.FloatTensor( x_train )
y= torch.FloatTensor( y_train )


# ## 딥러닝학습

# In[8]:


model = Sequential()
model.add_module('nn1', Linear(8,64)) 
model.add_module('relu1', ReLU())
model.add_module('nn2', Linear(64,32)) 
model.add_module('relu2', ReLU())
model.add_module('nn3', Linear(32,1)) 
model.add_module('sig1', Sigmoid())
loss_fn = torch.nn.BCELoss()


# In[9]:


optimizer = Adam(model.parameters(),lr=0.01)


# In[10]:


for epoch in range(1000):
    optimizer.zero_grad()
    hx = model(x) #model.forward(x)
    cost = loss_fn(hx,y)
    cost.backward()
    optimizer.step()
    print('cost : ', cost.item())


# ## train f1 score

# In[11]:


trainy = (( model(x) > 0.5 ) + 0).numpy()
f1_score(y.numpy(), trainy )


# ## test f1 score

# In[12]:


testx = torch.FloatTensor(x_test)
r = y_test
soly = (( model(testx) > 0.5 ) + 0).numpy()
f1_score(r, soly, average = 'macro' )

