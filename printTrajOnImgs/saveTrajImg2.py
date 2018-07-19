#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:32:41 2017

@author: nigno
"""

# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

from DataLoading import JaeseokDataset, Rescale, ToTensor, Normalize, show_trajectories, Shift, ChangeLight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from skimage import transform
import matplotlib.pyplot as plt
import time
import os
import cv2

from GMRLoss import GMRLoss
from GMR import GMR
from JaeseokNet import JaeseokNetPretrained, JaeseokNet
from PrintGaussians import plot_results, plot_results_time

use_gpu = torch.cuda.is_available()

new_h = 240
new_w = 320
    
seed = 3;

checkpoint_dir = '/media/nigno/Data/checkpoints/'
img_dir = 'realtest_imgs/'
save_dir = img_dir + 'results/'

file_list = os.listdir(img_dir)
nframes = len(file_list)

model = JaeseokNet()
        
print (model)

if use_gpu:
    model = model.cuda()
    
criterion = nn.MSELoss()
exp_lr_scheduler = None
optimizer_ft = optim.Adam(model.parameters())

cpName = checkpoint_dir + 'checkpointAllEpochs.tar'
if os.path.isfile(cpName):
    print("=> loading checkpoint '{}'".format(cpName))
    checkpoint = torch.load(cpName)
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    loss_list = checkpoint['loss_list']
    loss_list_val = checkpoint['loss_list_val']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer_ft.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
              
model.train(False)

traj_len = 200
if use_gpu:
    torchType = torch.cuda.FloatTensor
else:
    torchType = torch.FloatTensor
    
img_count = 0   

for i in range(nframes):
    
    
    img_name = img_dir + 'test' + str(i + 1) + '.jpg'
    #img_name = img_dir + str(i) + '.png'
    cv_image = cv2.imread(img_name)
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    new_h = 240
    new_w = 320
    image = transform.resize(image, (new_h, new_w))
    image = image.transpose((2, 0, 1))
    image_torch = torch.from_numpy(image).float()
    
    
    
    if use_gpu:
        inputs = Variable(image_torch.cuda())
    else:
        inputs = Variable(image_torch)
    inputs = inputs.unsqueeze(0)

    outputs = model(inputs)

    fig = plt.figure()
    
    nameFile = save_dir + str(i) + '.txt'
    np.savetxt(nameFile, outputs[0].cpu().data.numpy())        
    
    outData = GMR(outputs[0])
    
    inp = inputs[0].cpu().data.numpy().transpose((1, 2, 0))
    plt.imshow(inp, origin='upper')
    plt.axis('off')
    
    img_out_data = outData.cpu().data.numpy()
    
    img_out_data[:, 0] = (img_out_data[:, 0] + 1.0) * 240
    img_out_data[:, 1] = (img_out_data[:, 1] + (2 / 3)) * 240
    
    plt.scatter(img_out_data[:, 1], img_out_data[:, 0], s=10, marker='.', color='blue')
    
    fig_name = save_dir + str(i) + '.png'         
    fig.savefig(fig_name)
     
    plt.pause(0.001)
    
    plt.close(fig)