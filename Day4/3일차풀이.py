#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential, Sigmoid, Softmax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
import seaborn as sb
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# In[2]:


iris = load_iris()


# In[3]:


df = pd.DataFrame( iris['data'] )
df.columns = iris['feature_names']
df['species'] = iris['target']
df


# In[ ]:





# In[4]:


x_data = df.iloc[:,:-1].values
y_data = pd.get_dummies( df['species'] ).values


# In[5]:


scaleF = MinMaxScaler()
x_dataN = scaleF.fit_transform( x_data )


# In[6]:


x= torch.FloatTensor( x_dataN)
y= torch.FloatTensor( y_data)


# In[13]:


model = Sequential()
model.add_module('nn1', Linear(4,3)) # w:4x3 b:3
model.add_module( 'soft1', Softmax(dim=1) ) #활성함
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = Adam( model.parameters(), lr=0.1)


# In[14]:


hist=[]
for epoch in range(4000):
    optimizer.zero_grad()
    hx = model(x)
    cost = loss_fn( hx, y)
    cost.backward()
    optimizer.step()
    print( epoch, cost.item() )
    hist.append( cost.item() )


# In[9]:


sb.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
plt.show()


# In[15]:


plt.plot( hist)
plt.show()


# In[16]:


from sklearn.metrics import confusion_matrix,f1_score, accuracy_score


# In[20]:


r = y.argmax(dim=1).numpy()
r


# In[24]:


pred = model( x ).argmax(dim=1).detach().numpy()
pred


# In[26]:


c = confusion_matrix(r, pred)
c


# In[27]:


sb.heatmap(c, annot=True)
plt.show()


# In[28]:


accuracy_score( r, pred)


# In[30]:


f1_score(r,pred,average='macro')

