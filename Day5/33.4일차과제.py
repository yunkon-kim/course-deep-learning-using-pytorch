#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import Adam
from torch.nn import Linear,MSELoss,Sequential,Sigmoid,Softmax,ReLU,Module
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


# In[ ]:


iris데이터셋을 이용
1. 정규화
2. train, test
3. class 기반 딥러닝
4. 학습간 정확도 출력
5. test데이터 0번째 행 예측값 확인
6. 검증
- train, test f1 score
- confusion matrix, heat map


# In[2]:


iris = load_iris()


# In[3]:


df = pd.DataFrame( iris['data'] )
df.columns = iris['feature_names']
df['species'] = iris['target']
df


# In[5]:


x_data = df.iloc[:,:-1].values
y_data = pd.get_dummies( df['species'] ).values


# In[4]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[6]:


scaleF = MinMaxScaler()
x_dataN = scaleF.fit_transform( x_data )


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(x_dataN, y_data, 
                                               test_size=0.3, stratify=y_data)


# In[8]:


x_train.shape


# In[9]:


y_train.shape


# In[11]:


x = torch.FloatTensor( x_train )
y = torch.FloatTensor( y_train )


# In[15]:


class IrisModel( Module ):
    def __init__(self):
        super().__init__() #초기화작업
        self.nn1 = Linear(4,20)
        self.relu1 = ReLU()
        self.nn2 = Linear(20,3)
        self.soft1 = Softmax(dim=1)

    def forward(self, x):
        print('foward call')
        x = self.nn1( x ) # matmul( x,w1)+b1 w1:2x20
        x = self.relu1( x ) 
        x = self.nn2( x ) # matmul( x,w2)+b2 w1:20x1
        x = self.soft1( x )# 
        return x


# In[16]:


model = IrisModel()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = Adam( model.parameters(), lr=0.01)
for epoch in range(1000):
    optimizer.zero_grad()
    hx = model(x) #model.forward(x)
    cost = loss_fn( hx, y)
    cost.backward()
    optimizer.step()
    print('--------------------------')
    rdata = y.argmax(dim=1).numpy()
    pred = hx.argmax(dim=1).numpy()
    print('정확도', accuracy_score(rdata,pred) )
    print(epoch, cost.item() )


# ### 검증

# In[22]:


x.shape
105x4 4x20 = 105x20 20x3 = 105x3 


# In[26]:


r = y.argmax(dim=1).numpy()
h = model( x ).argmax( dim=1).detach().numpy()


# In[30]:


# train f1 score
f1_score(r,h, average='macro')


# In[ ]:


45x4 4x20 = 45x20 20x3  45x3


# In[43]:


r = y_test.argmax( axis=1 )
p = model( torch.FloatTensor(x_test)).argmax(dim=1).detach().numpy()


# In[44]:


# test f1 score
f1_score(r,p, average='macro')


# In[45]:


confusion_matrix(r,p)


# In[49]:


sb.heatmap(confusion_matrix(r,p),annot=True,linewidths=0.2,cmap='Reds')
plt.show()

