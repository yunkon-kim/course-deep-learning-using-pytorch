#!/usr/bin/env python
# coding: utf-8

# titanic 데이터셋
# 
# import seaborn as sb   
# titanic = sb.load_dataset('titanic')   
# titanic['age'].fillna(titanic['age'].mean(), inplace=True )   
# 특성데이터:,탑승등급(pclass),성별(sex), 나이(age), 탑승금액(fair)   
# 타겟:survived   
# 
# 1) 상관관계 와 heatmap 그리시요   
# 2) 정규화하시요   
# 3) train test 셋으로 나누시요   
# 3) 딥러닝으로 학습하시요(학습시 정확도 출력)   
# 4) cost에대한 hist를 라인차트로 그리시요   
# 5) confusion matrix 와 heat map 그리시요   
# 6) train, test set f1 score 를 구하시요   
# 7) 1등급, 여성, 32세, 65달러인경우 생존여부를 예측하시요   

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential, Sigmoid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, accuracy_score, precision_score

import seaborn as sb


# In[2]:


titanic = sb.load_dataset('titanic')
titanic


# In[3]:


titanic['age'].fillna(titanic['age'].mean(), inplace=True )
titanic


# #### sex 컬럼: text to value 변환 
# male: 1 female: 0

# In[4]:


titanic['sex'] = titanic['sex'].apply(lambda v:0 if v=='male' else 1)


# In[5]:


data = titanic[['pclass', 'sex', 'age', 'fare', 'survived']]
data


# ### 1) 상관관계 와 heatmap 그리시요   

# In[6]:


sb.heatmap( data.corr(), vmin=-1, vmax=1, annot=True,
           linewidths=0.2, cmap='seismic' )
plt.show()


# In[7]:


x_data = data.iloc[:,:-1].values
y_data = data[['survived']].values


# In[8]:


x_data.shape


# In[9]:


y_data.shape


# ### 2) 정규화하시요   

# In[10]:


scaleF = MinMaxScaler()
x_dataN = scaleF.fit_transform( x_data )


# ### 3) train test 셋으로 나누시요   

# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x_dataN, y_data, test_size=0.3, stratify=y_data)


# ### 4) 딥러닝으로 학습하시요(학습시 정확도 출력)   

# In[12]:


x = torch.FloatTensor( x_train )
y = torch.FloatTensor( y_train )
print(x.shape)
print(y.shape)
feature_len = x.shape[1]
label_len = y.shape[1]
print('feature_len:', feature_len, ', label_len:', label_len)


# In[13]:


class LogisticModel(nn.Module):
    def __init__(self, feat_len, label_len):
        super().__init__() # 초기화작업
        self.nn1 = Linear(feat_len, 20)
        self.nn2 = Linear(20, label_len)
        self.sig = Sigmoid()
        
    
    def forward(self, x):
        print('forward call')
#         x = self.nn1.forward(x)
        x = F.relu( self.nn1(x) ) # matmul(x, w1) + b1  w1:2x20
        x = self.sig( self.nn2(x) ) # matmul(x, w2) + b2  w2:20x1
        return x    


# In[14]:


model = LogisticModel(feature_len, label_len)
loss_fn = torch.nn.BCELoss()
optimizer = Adam( model.parameters(), lr=0.01)

hist = []
for step in range(1000):
    optimizer.zero_grad()
    hx = model.forward(x)
    cost = loss_fn(hx, y)
    cost.backward()
    optimizer.step()
    print('=====================')
    rdata = y.numpy()    
    pred = ((hx > 0.5) + 0).numpy()
    print('정확도',accuracy_score(rdata, pred))
    hist.append(cost.item())


# ### 5) cost에대한 hist를 라인차트로 그리시요   

# In[15]:


plt.plot(hist)
plt.show()


# ### 6) confusion matrix 와 heat map 그리시요   

# In[16]:


train_y_pred = (model( torch.FloatTensor(x_train)) > 0.5) + 0
train_y_pred


# In[17]:


train_y_true = y_train
train_y_true


# In[18]:


train_con_mat = confusion_matrix(train_y_true, train_y_pred)
sb.heatmap(train_con_mat, annot=True, fmt='d',
          xticklabels=['Pred survived', 'Pred not survived'], yticklabels=['Survived', 'Not survived'])
plt.show()


# In[19]:


test_y_pred = (model( torch.FloatTensor(x_test)) > 0.5) + 0
test_y_pred


# In[20]:


test_y_true = y_test
test_y_true


# In[21]:


test_con_mat = confusion_matrix(test_y_true, test_y_pred)
sb.heatmap(test_con_mat, annot=True, fmt='d', 
           xticklabels=['Pred survived', 'Pred not survived'], yticklabels=['Survived', 'Not survived'])
plt.show()


# ### 7) train, test set f1 score 를 구하시요

# In[22]:


f1_score(train_y_true, train_y_pred)


# In[23]:


f1_score(test_y_true, test_y_pred)


# ### 8) 1등급, 여성, 32세, 65달러인경우 생존여부를 예측하시요 

# In[24]:


p = (model( torch.FloatTensor([[1, 1, 32, 65]])) > 0.5) + 0

print("1등급, 여성, 32세, 65달러인경우 생존여부 예측")
if p == 1:
    print('Survived')
else:
    print('Not survived')

