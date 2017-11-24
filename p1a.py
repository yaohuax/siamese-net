#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:24:00 2017

@author: yaohuaxu
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import optim
import random
import PIL.ImageOps
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
import os
import re

# Helper Functions
#def imshow(img,text,should_save=False):
#    npimg = img.numpy()
#    plt.axis("off")
#    if text:
#        plt.text(75, 8, text, style='italic',fontweight='bold',
#            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()    
#
#def show_plot(iteration,loss):
#    plt.plot(iteration,loss)
#    plt.show()

# To Generate train/test List    
def ls(root, mode):
    if mode == 'train':
        path = os.path.join(root, 'train.txt')
    else:
        path = os.path.join(root, 'test.txt')
    file = open(path)
    seq = re.compile("\s+")
    result = []
    for line in file:
        lst = seq.split(line.strip())
        item = {
            "img0": lst[0],
            "img1": lst[1],
            "label": lst[2]
        }
        result.append(item)
    file.close()
    return result

#Configuration Class
class Config():
    training_dir = "/home/yaohuaxu1/siamese-net/lfw/"
    batch_size = 64
    train_number_epochs = 3
    
#Custom Dataset
class LFWDataset(Dataset):
    
    def __init__(self, root, lst, data_augmentation, transform = None):
        self.root = root
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.lst = lst
    
    def __getitem__(self, index):
        #print index
        path0 = os.path.join(self.root, self.lst[index]['img0'])
        path1 = os.path.join(self.root, self.lst[index]['img1'])
        img0 = Image.open(path0)
        img1 = Image.open(path1) 
        label = np.array([self.lst[index]['label']])
        label = label.astype(np.float)
        if self.transform is not None:
            if self.data_augmentation:
                img0 = self.augmentation(img0)
                img1 = self.augmentation(img1)
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1 , label
    
    def augmentation(self, img0):
        rotate_range = random.uniform(-30, 30)
        translation_range = random.uniform(-10, 10)
        scale_range = random.uniform(0.7, 1.3)
        if np.random.random() < 0.7:
            img0 = img0.rotate(rotate_range)
        if np.random.random() < 0.7:
            img0 = img0.transform((img0.size[0], img0.size[1]), Image.AFFINE, (1, 0, translation_range, 0, 1, translation_range))
        if np.random.random() < 0.7:
            img0 = img0.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() < 0.7:
            img0 = img0.resize((int(128 * scale_range), int(128 * scale_range)))
            half_the_width = img0.size[0] / 2
            half_the_height = img0.size[1] / 2
            img0 = img0.crop((half_the_width - 64,
                    half_the_height - 64,
                    half_the_width + 64,
                    half_the_height + 64))
        return img0
    
    def __len__(self):
        return len(self.lst)
    
#CNN Definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(64, 128, 5, padding = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(256, 512, 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16*16*512, 1024),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )
        
        self.fc = nn.Linear(2048, 1)
        
    def forward_once(self, x):
        x = self.features(x)
        #print x.data.shape
        x = x.view(x.size(0), -1)
        #print x.data.shape
        x = self.classifier(x)
        return x
        
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        #print output1.data.shape
        output = torch.cat((output1, output2),1)
        output = self.fc(output)
        pred = torch.sigmoid(output)
        return pred


#Using ImageFolder
lst = ls(Config.training_dir, mode = 'train')
test_list = ls(Config.training_dir, mode = 'test')
#print len(test_list)
lfwDataset = LFWDataset(root = Config.training_dir, lst = lst, data_augmentation = False,
                        transform=transforms.Compose([transforms.Scale((128,128)),transforms.ToTensor()]))
testDataset = LFWDataset(root = Config.training_dir, lst = test_list, data_augmentation = False,
                         transform=transforms.Compose([transforms.Scale((128,128)),transforms.ToTensor()]))
#print len(lfwDataset)
#print len(testDataset)

#Data Loader
trainloader = DataLoader(lfwDataset, batch_size = Config.batch_size, shuffle = True)
testloader = DataLoader(testDataset, batch_size = Config.batch_size, shuffle = False)

net = Net().cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.00001)
counter = []
loss_history = []
iteration_number = 0
for epoch in range(Config.train_number_epochs):
    for i, data in enumerate(trainloader, 0):
        img0, img1, label = data
        label = label.type(torch.FloatTensor)
        #print label.size(0)
        #print type(img0), type(label)
        img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
        #print label.data.shape
        #print net.forward(img0,img1)
        output = net.forward(img0, img1)
        #print output
        optimizer.zero_grad()
        
        #print type(label)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss.data[0]))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss.data[0])
#show_plot(counter,loss_history)

#Save Model
torch.save(net.state_dict(), f='p1a model')

#Test
net.load_state_dict(torch.load(f='p1a model'))

total = 0
correct = 0
for _, data in enumerate(trainloader,0):
    img0, img1, label = data
    label = label.type(torch.FloatTensor)
    img0, img1, label = Variable(img0, volatile = True).cuda(), Variable(img1, volatile = True).cuda(), Variable(label).cuda()
    output = net.forward(img0, img1)
    output = torch.round(output)
    total += label.size(0)
    correct += (output == label).sum()
print('Accuracy of the network on the train images: %d %%' % (
      100 * correct / total))    

total = 0
correct = 0
for _, data in enumerate(testloader,0):
    img0, img1, label = data
    label = label.type(torch.FloatTensor)
    img0, img1, label = Variable(img0, volatile = True).cuda(), Variable(img1, volatile = True).cuda(), Variable(label).cuda()
    output = net.forward(img0, img1)
    output = torch.round(output)
    total += label.size(0)
    correct += (output == label).sum()
print('Accuracy of the network on the test images: %d %%' % ((
      100 * correct / total)).data.numpy())         