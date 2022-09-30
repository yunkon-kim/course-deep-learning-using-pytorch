#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[4]:


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # in_channels: input 이미지의 color depth (흑백, 컬리, 등)
        # out_channels: 필터의 수 3개
        # kernel_size: 필터의 크기 5x5
        # stirde: 필터 이동 크기
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3,
                              kernel_size=5, stride=1)
#         self.relu1 = nn.ReLU() <= 이번에는 forward에서 함
        # 위 out_channels과 아래 in_channels 맞춰줌
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=10,
                              kernel_size=5, stride=1)
#         self.relu1 = nn.ReLU() <= 이번에는 forward에서 함
        # kernel_size: 필터의 크기
        # stride: 필터 이동 크기 (default는 kernel_size를 따름)
        self.max1 = nn.MaxPool2d(kernel_size=2) # 여기서는 stride=2가 됨

# deep
        self.fc1 = nn.Linear(10*6*6, 50) # 10*6*6은 2차원을 1차원으로 펴주는 효과
        # ReLU
        self.fc2 = nn.Linear(50, 10)
        # Softmax
    
    def forward(self, x):
        print('연산전', x.shape)
        # x = self.conv1(x)
        # x = F.relu(x)
        x = F.relu( self.conv1(x) )
        print('conv1 연산후', x.shape)
        
        x = F.relu( self.conv2(x) )   
        print('conv2 연산후', x.shape)
        
        x = F.relu( self.max1(x) )
        print('max1 연산후', x.shape) # c x w x h c컬러Depth w행 h열 
        
        x = x.view(-1, 10*6*6)  # x.reshape(-1, 10*6*6)도 같음        
        x = F.relu( self.fc1(x) )
        print('fc1 연산후', x.shape)
        
        x = self.fc2(x)
        print('fc2 연산후', x.shape)
        
        return x       
        


# In[7]:


rImg = torch.randn(10,1,20,20) # 이미지 갯수, color depth, width, height
rImg


# In[9]:


cnn = CNN()

# cnn.forward(rImg)
cnn(rImg)


# In[ ]:




