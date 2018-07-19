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
import matplotlib.pyplot as plt
import time
import os
import cv2

from GMRLoss import GMRLoss
from GMR import GMR
from JaeseokNet import JaeseokNetPretrained, JaeseokNet
from PrintGaussians import plot_results, plot_results_time

use_gpu = torch.cuda.is_available()

data_transforms_custom = transforms.Compose([Rescale((240, 320)),
                                             #Shift(-0.6, 0.2, -0.65, 0.4),
                                             #ChangeLight(0.3, percentage = 0.8),
                                             #Normalize(mean, std),
                                             ToTensor()])
    
seed = 3;

checkpoint_dir = '/media/nigno/Data/checkpoints/'

dataset = JaeseokDataset(root_dir = 'Corrected_dataset5/',
                             transform = data_transforms_custom,
                             dset_type='train', seed=seed, 
                             training_per = 0.01,
                             permuted= True)
                             
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=8)
                                         
print(len(dataset))

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

for i, data in enumerate(dataloader):
    samples = data
    if use_gpu:
        inputs = Variable(samples['image'].cuda())
        trajectories = Variable(samples['trajectories'].cuda())
    else:
        inputs, trajectories = Variable(samples['image']), Variable(samples['trajectories'])

    outputs = model(inputs)

    for j in range(outputs.size()[0]):
        fig = plt.figure()
        
        nameFile = 'outputnet/outputRede' + str(img_count) + '.txt'
        np.savetxt(nameFile, outputs[j].cpu().data.numpy())        
        
        outData = GMR(outputs[j])
        loss = (trajectories[j] - outData).pow(2).sum().sqrt() / 200
        print(loss)
        
        inp = inputs[j].cpu().data.numpy().transpose((1, 2, 0))
        plt.imshow(inp, origin='upper')
        plt.axis('off')
        
        img_traj = trajectories[j].cpu().data.numpy()
        img_out_data = outData.cpu().data.numpy()
        
        img_traj[:, 0] += 0.00    
        img_out_data[:, 0] += 0.00
        
        img_traj[:, 0] = (img_traj[:, 0] + 1.0) * 240
        img_traj[:, 1] = (img_traj[:, 1] + (2 / 3)) * 240
        img_out_data[:, 0] = (img_out_data[:, 0] + 1.0) * 240
        img_out_data[:, 1] = (img_out_data[:, 1] + (2 / 3)) * 240
        
#        img_traj[:, 0] = (img_traj[:, 0] + 1.0) * 240
#        img_traj[:, 1] = (img_traj[:, 1] + 0.75) * 240
#        img_out_data[:, 0] = (img_out_data[:, 0] + 1.0) * 240
#        img_out_data[:, 1] = (img_out_data[:, 1] + 0.75) * 240
        
        # Four corners of the book in source image
        pts_src = np.array([[103, 84], [99, 222], [162, 91], [157, 223]])
 
        # Four corners of the book in destination image.
        pts_dst = np.array([[120, 100], [120, 220], [168, 100], [168, 220]])
 
        # Calculate Homography
        h, status = cv2.findHomography(pts_src, pts_dst)
        
        new_traj = np.concatenate((img_traj.T, np.ones((1, 200))), axis = 0)
        
        new_traj = h.dot(new_traj)
        new_traj = new_traj / new_traj[2, :]
     
        
#        img_traj[:, 0] = (img_traj[:, 0] + 1.0) * 240
#        img_traj[:, 1] = (img_traj[:, 1] + 0.75) * 240
#        img_out_data[:, 0] = (img_out_data[:, 0] + 1.0) * 240
#        img_out_data[:, 1] = (img_out_data[:, 1] + 0.75) * 240
        
        #imshow(inputs.cpu().data[j])
        #plt.scatter(new_traj.T[:, 1], new_traj.T[:, 0], s=10, marker='.', color='green')
        plt.scatter(img_traj[:, 1], img_traj[:, 0], s=10, marker='.', color='red')
        #plt.scatter(img_out_data[:, 1], img_out_data[:, 0], s=10, marker='.', color='blue')
        
        fig_name = '/home/nigno/Robots/pytorch_tests/JaeseokNet/icubTest/Corrected_dataset4/imageTraj_test_prova/' + \
                   str(img_count) + '.png'         
        fig.savefig(fig_name)
        img_count += 1
         
        plt.pause(0.001)
        
        plt.close(fig)