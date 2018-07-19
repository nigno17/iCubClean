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

from GMRLoss import GMRLoss
from GMR import GMR
from JaeseokNet import JaeseokNetPretrained, JaeseokNet
from PrintGaussians import plot_results, plot_results_time


plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

fixed = False
train = False
restore = False

checkpoint_dir = '/media/nigno/checkpoints/'

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    #inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
                 
data_transforms_custom_train = transforms.Compose([Rescale((240, 320)),
                                                   #Shift(-0.6, 0.2, -0.65, 0.4),
                                                   #ChangeLight(0.3, percentage = 0.8),
                                                   #Normalize(mean, std),
                                                   ToTensor()])
data_transforms_custom_val = transforms.Compose([Rescale((240, 320)),
                                                 #Normalize(mean, std),
                                                 ToTensor()])
    
seed = 3;

train_dataset = JaeseokDataset(root_dir = 'Corrected_dataset4/',
                               transform = data_transforms_custom_train,
                               dset_type='train', seed=seed, 
                               training_per = 0.7)

val_dataset = JaeseokDataset(root_dir = 'Corrected_dataset4/',
                             transform = data_transforms_custom_val,
                             dset_type='val', seed=seed, 
                             training_per = 0.7)

print(len(train_dataset))
print(len(val_dataset))

dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=200,
                                              shuffle=True, num_workers=6)
dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=2,
                                              shuffle=True, num_workers=6)
train_size = len(train_dataset)
val_size = len(val_dataset)

use_gpu = torch.cuda.is_available()

samples = next(iter(dataloader_train))

## Make a grid from batch
out = torchvision.utils.make_grid(samples['image'])

imshow(out, title=['batch'])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, start_epoch=0, loss_list=[], loss_list_val=[]):
    since = time.time()

    best_model_wts = model.state_dict()
#    best_acc = 0.0
    best_loss = 100.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + start_epoch, num_epochs - 1 + start_epoch))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if (exp_lr_scheduler != None):
                    scheduler.step()
                    print ('scheduler on')
                model.train(True)  # Set model to training mode
                dataloader = dataloader_train
                dataset_size = train_size
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = dataloader_val
                dataset_size = val_size

            running_loss = 0.0
#            running_corrects = 0

            # Iterate over data.
            samples_count = 0
            print('reading data')
            for data in dataloader:
                samples_count += 1
                printProgressBar(samples_count, len(dataloader), prefix = phase, suffix = 'Complete', length = 50)
                # get the inputs
                samples = data
                
                samples['trajectories'].resize_((samples['trajectories'].size()[0], 
                                                 samples['trajectories'].size()[1] * samples['trajectories'].size()[2]))

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(samples['image'].cuda())
                    trajectories = Variable(samples['trajectories'].cuda())
                else:
                    inputs, trajectories = Variable(samples['image']), Variable(samples['trajectories'])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
#                _, preds = torch.max(outputs.data, 1)
                #loss = criterion(outputs, trajectories)
                
                loss = 0
                for i in range(outputs.size()[0]):
                    loss = loss + GMRLoss(outputs[i], trajectories[i])
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                #print('sample {}/{}. Loss: {}. Dataset size: {}'.format(samples_count, len(dataloader), loss.data[0], dataset_size))
#                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
#            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.7f}'.format(phase, epoch_loss))
            
            if phase == 'val':
                loss_list_val += [epoch_loss]
            else:
                loss_list += [epoch_loss]

            # deep copy the model
            #if phase == 'val' and epoch_loss < best_loss:
            torch.save({
                        'epoch': epoch + start_epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_loss': best_loss,
                        'loss_list': loss_list,
                        'loss_list_val': loss_list_val,
                        'optimizer': optimizer.state_dict(),
                        }, checkpoint_dir + 'checkpointAllEpochs.tar' )
            #if phase == 'val' and epoch_loss < best_loss:
            if ((epoch + start_epoch + 1) % 10) == 0:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save({
                            'epoch': epoch + start_epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_loss': best_loss,
                            'loss_list': loss_list,
                            'loss_list_val': loss_list_val,
                            'optimizer': optimizer.state_dict(),
                            }, checkpoint_dir + 'checkpoint' + str(epoch + start_epoch + 1) + '.tar')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:7f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=4, tort = 'Val'):
    images_so_far = 0
    fig = plt.figure()
    model.train(False)
    
    traj_len = 200
    if use_gpu:
        torchType = torch.cuda.FloatTensor
    else:
        torchType = torch.FloatTensor
    time = Variable(torch.arange(0, (traj_len - 1) / traj_len, 1 / traj_len).type(torchType)  , requires_grad=False)
    
    if tort == 'Train':
        dataloader = dataloader_train
    else:
        dataloader = dataloader_val

    for i, data in enumerate(dataloader):
        samples = data
        if use_gpu:
            inputs = Variable(samples['image'].cuda())
            trajectories = Variable(samples['trajectories'].cuda())
        else:
            inputs, trajectories = Variable(samples['image']), Variable(samples['trajectories'])

        outputs = model(inputs)

        for j in range(outputs.size()[0]):
            outData = GMR(outputs[j])
            loss = (trajectories[j] - outData).pow(2).sum().sqrt() / 200
            print(loss)
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            
            inp = inputs[j].cpu().data.numpy().transpose((1, 2, 0))
            #mean = np.array([0.485, 0.456, 0.406])
            #std = np.array([0.229, 0.224, 0.225])
            #inp = std * inp + mean
            #inp = np.clip(inp, 0, 1)
            ax.imshow(inp, origin='upper')
            
            img_traj = trajectories[j].cpu().data.numpy()
            img_out_data = outData.cpu().data.numpy()
            
#            img_traj[:, 1] = (img_traj[:, 1] + 0.75) * 240
#            img_traj[:, 0] = (img_traj[:, 0] + 1.0) * 240
#            img_out_data[:, 1] = (img_out_data[:, 1] + 0.75) * 240
#            img_out_data[:, 0] = (img_out_data[:, 0] + 1.0) * 240
            
            img_traj[:, 0] = (img_traj[:, 0] + 1.0) * 240
            img_traj[:, 1] = (img_traj[:, 1] + (2.0 / 3.0)) * 240
            img_out_data[:, 0] = (img_out_data[:, 0] + 1.0) * 240
            img_out_data[:, 1] = (img_out_data[:, 1] + (2.0 / 3.0)) * 240
            
            #imshow(inputs.cpu().data[j])
            plt.scatter(img_traj[:, 1], img_traj[:, 0], s=10, marker='.', color='red')
            plt.scatter(img_out_data[:, 1], img_out_data[:, 0], s=10, marker='.', color='blue')
#            plt.scatter(time.cpu().data.numpy(), trajectories[j][:, 0].cpu().data.numpy(), s=10, marker='.', color='red')
#            plt.scatter(time.cpu().data.numpy(), outData[:, 0].cpu().data.numpy(), s=10, marker='.', color='blue')
            
#            plot_results(outputs[j], ax)
#            plot_results_time(outputs[j], ax)
             
            plt.pause(0.001)

            if images_so_far == num_images:
                return
            
def save_model(model, tort = 'Val'):
    images_so_far = 0

    if tort == 'Train':
        dataloader = dataloader_train
    else:
        dataloader = dataloader_val

    for i, data in enumerate(dataloader):
        samples = data
        if use_gpu:
            inputs = Variable(samples['image'].cuda())
            trajectories = Variable(samples['trajectories'].cuda())
        else:
            inputs, trajectories = Variable(samples['image']), Variable(samples['trajectories'])

        outputs = model(inputs)

        for j in range(inputs.size()[0]):
            outData = GMR(outputs[j], trajectories[j])
            images_so_far += 1
            file_path = 'predictedTraj' + tort + '/' + str(images_so_far) + '.txt'
            np.savetxt(file_path, outData.cpu().data.numpy())
            
#model_ft = models.resnet18(pretrained=False)
model_ft = JaeseokNet()
if (fixed == True):
    for param in model_ft.parameters():
        param.requires_grad = False
#num_ftrs = model_ft.fc.in_features
#num_output = (3 + 3 * 3) * 5
#model_ft.fc = nn.Linear(num_ftrs, num_output)
        
print (model_ft)

if use_gpu:
    model_ft = model_ft.cuda()
    
if (train == True):
    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    if (fixed == True):
        #optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
        optimizer_ft = optim.Adam(model_ft.fc.parameters())
    else:
        #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        optimizer_ft = optim.Adam(model_ft.parameters())

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    exp_lr_scheduler = None

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=1000)
else:
     # Observe that all parameters are being optimized
    if (fixed == True):
        #optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
        optimizer_ft = optim.Adam(model_ft.fc.parameters())
    else:
        #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        optimizer_ft = optim.Adam(model_ft.parameters())
        
    cpName = 'checkpoint.tar'
    if os.path.isfile(cpName):
        print("=> loading checkpoint '{}'".format(cpName))
        checkpoint = torch.load(cpName)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        loss_list = checkpoint['loss_list']
        loss_list_val = checkpoint['loss_list_val']
        model_ft.load_state_dict(checkpoint['state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        
if (restore == True):
    criterion = nn.MSELoss()
    
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    exp_lr_scheduler = None
    
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=1000, start_epoch=start_epoch,
                           loss_list=loss_list, loss_list_val=loss_list_val)
    
if train != True:
    visualize_model(model_ft, tort = 'Train')
    visualize_model(model_ft, tort = 'Val')
    #save_model(model_ft, tort = 'Train')
    #save_model(model_ft, tort = 'Val')

    fig = plt.figure()
    plt.plot(loss_list[2:], color='red')
    plt.plot(loss_list_val[2:], color='blue')

raw_input('Press enter to continue: ')

#
#visualize_model(model_conv)
#
#plt.ioff()
#plt.show()
#
#plt.pause(2)
#
#
