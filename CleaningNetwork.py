#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:32:41 2017

@author: nigno
"""

# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

from DataLoading import JaeseokDataset, JaeseokDatasetRam, Rescale, ToTensor, Normalize, show_trajectories, Shift, ChangeLight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import visdom

from GMRLoss import GMRLoss
from GMR import GMR
from JaeseokNet import JaeseokNetPretrained, JaeseokNet


#plt.ion()   # interactive mode

vis = visdom.Visdom()

vis.close(None)

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

# Modalities
train = True
restore = False

# check if the checkpoints dir exist otherwise create it
checkpoint_dir = 'checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
# check if the checkpoints dir exist otherwise create it
save_dir = 'errors/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# mean and std of the pretrained Alexnet with imagenet
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
    
                 
data_transforms_custom_train = transforms.Compose([Rescale((240, 320)),
                                                   #Shift(-0.6, 0.2, -0.65, 0.4),
                                                   #ChangeLight(0.3, percentage = 0.8),
                                                   #Normalize(mean, std),
                                                   ToTensor()])
data_transforms_custom_val = transforms.Compose([Rescale((240, 320)),
                                                 #Normalize(mean, std),
                                                 ToTensor()])
    
seed = 3;

train_dataset = JaeseokDataset(root_dir = 'datasetICDL-JINT/JINT/',
                                  transform = data_transforms_custom_train,
                                  dset_type='train', seed=seed, 
                                  training_per = 0.8)

val_dataset = JaeseokDataset(root_dir = 'datasetICDL-JINT/JINT/',
                                transform = data_transforms_custom_val,
                                dset_type='val', seed=seed, 
                                training_per = 0.8)

# Load all the dataset in RAM

#train_dataset = JaeseokDatasetRam(root_dir = 'datasetICDL-JINT/JINT/',
#                                  transform = data_transforms_custom_train,
#                                  dset_type='train', seed=seed, 
#                                  training_per = 0.8)
#
#val_dataset = JaeseokDatasetRam(root_dir = 'datasetICDL-JINT/JINT/',
#                                transform = data_transforms_custom_val,
#                                dset_type='val', seed=seed, 
#                                training_per = 0.8)

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
vis.images(samples['image'], 
           opts=dict(title='Batch',),)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, start_epoch=0, 
                loss_list=[], loss_list_val=[], 
                loss_1_list=[], loss_1_list_val=[],
                loss_2_list=[], loss_2_list_val=[],
                loss_mse_list=[], loss_mse_list_val=[]):   

    vis = visdom.Visdom()   

    lossWin = vis.line(Y=np.linspace(0, 4, 4),  
               X=np.linspace(0, 4, 4),
               opts=dict(
               markersize=10,
               ytickmax=50,),
               #markercolor=np.ndarray([255, 0, 0]),),
               name='Train',)  
    vis.line(Y=np.linspace(0, 9, 4),  
               X=np.linspace(0, 4, 4),
               opts=dict(
               markersize=10,
               ytickmax=50,),
               #markercolor=np.ndarray([255, 0, 0]),),
               win=lossWin,
               name='Val',
               update='new') 
                    
    since = time.time()

    best_model_wts = model.state_dict()
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
            running_loss_1 = 0.0
            running_loss_2 = 0.0
            running_loss_mse = 0.0

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
                #loss = criterion(outputs, trajectories)
                
                # Ad hoc Loss Function
                loss = 0
                loss_1 = 0
                loss_2 = 0
                loss_mse = 0
                for i in range(outputs.size()[0]):
                    [l, l_1, l_2, l_mse] = GMRLoss(outputs[i], trajectories[i])
                    loss = loss + l                    
                    loss_1 = loss_1 + l_1
                    loss_2 = loss_2 + l_2
                    loss_mse = loss_mse + l_mse
                    
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_loss_1 += loss_1.data[0]
                running_loss_2 += loss_2.data[0]
                running_loss_mse += loss_mse.data[0]

            epoch_loss = running_loss / dataset_size
            epoch_loss_1 = running_loss_1 / dataset_size
            epoch_loss_2 = running_loss_2 / dataset_size
            epoch_loss_mse = running_loss_mse / dataset_size

            print('{} Loss: {:.7f}'.format(phase, epoch_loss))
            
            if phase == 'val':
                loss_list_val += [epoch_loss]
                loss_1_list_val += [epoch_loss_1]
                loss_2_list_val += [epoch_loss_2]
                loss_mse_list_val += [epoch_loss_mse]
            else:
                loss_list += [epoch_loss]
                loss_1_list += [epoch_loss_1]
                loss_2_list += [epoch_loss_2]
                loss_mse_list += [epoch_loss_mse]

            # deep copy the model
            torch.save({
                        'epoch': epoch + start_epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_loss': best_loss,
                        'loss_list': loss_list,
                        'loss_list_val': loss_list_val,
                        'loss_1_list': loss_1_list,
                        'loss_1_list_val': loss_1_list_val,
                        'loss_2_list': loss_2_list,
                        'loss_2_list_val': loss_2_list_val,
                        'loss_mse_list': loss_mse_list,
                        'loss_mse_list_val': loss_mse_list_val,
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
                            'loss_1_list': loss_1_list,
                            'loss_1_list_val': loss_1_list_val,
                            'loss_2_list': loss_2_list,
                            'loss_2_list_val': loss_2_list_val,
                            'loss_mse_list': loss_mse_list,
                            'loss_mse_list_val': loss_mse_list_val,
                            'optimizer': optimizer.state_dict(),
                            }, checkpoint_dir + 'checkpoint' + str(epoch + start_epoch + 1) + '.tar')

        print()
        # VISDOM START
        #    fig = plt.figure()
        
        vis.line(Y=np.array(loss_list),  
                   X=np.linspace(0, len(loss_list), len(loss_list)),
                   opts=dict(
                   markersize=10,),
                   #markercolor=np.ndarray([255, 0, 0]),),
                   win=lossWin,
                   name='Train',
                   update='replace')
        vis.line(Y=np.array(loss_list_val),  
                   X=np.linspace(0, len(loss_list_val), len(loss_list_val)),       
             opts=dict(
             markersize=10,),
             #markercolor=np.ndarray([0, 0, 255]),),
             win=lossWin,
             name='Val',
             update='replace')
        
        #    fig = plt.figure()
        #    plt.plot(loss_1_list[2:], color='red')
        #    plt.plot(loss_1_list_val[2:], color='blue')
        #    plt.ylabel('Loss first component')
        #    vis.matplot(plt)
        #    
        #    fig = plt.figure()
        #    plt.plot(loss_2_list[2:], color='red')
        #    plt.plot(loss_2_list_val[2:], color='blue')
        #    plt.ylabel('Loss second component')
        #    vis.matplot(plt)
        #    
        #    fig = plt.figure()
        #    plt.plot(loss_mse_list[2:], color='red')
        #    plt.plot(loss_mse_list_val[2:], color='blue')
        #    plt.ylabel('Mean Square Error')
        #    vis.matplot(plt)
        # VISDOM END

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
            
            if use_gpu:
                inp = inputs[j].cpu().data.numpy().transpose((1, 2, 0))
                img_traj = trajectories[j].cpu().data.numpy()
                img_out_data = outData.cpu().data.numpy()
            else:
                inp = inputs[j].data.numpy().transpose((1, 2, 0))
                img_traj = trajectories[j].data.numpy()
                img_out_data = outData.data.numpy()
            
            ax.imshow(inp, origin='upper')            
            
            img_traj[:, 0] = (img_traj[:, 0] + 1.0) * 240
            img_traj[:, 1] = (img_traj[:, 1] + (2.0 / 3.0)) * 240
            img_out_data[:, 0] = (img_out_data[:, 0] + 1.0) * 240
            img_out_data[:, 1] = (img_out_data[:, 1] + (2.0 / 3.0)) * 240
            
            plt.scatter(img_traj[:, 1], img_traj[:, 0], s=10, marker='.', color='red')
            plt.scatter(img_out_data[:, 1], img_out_data[:, 0], s=10, marker='.', color='blue')
             
            plt.pause(0.001)

            if images_so_far == num_images:
                return
            
model_ft = JaeseokNet()
        
print (model_ft)

if use_gpu:
    model_ft = model_ft.cuda()
    
if (train == True):
    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters())

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    exp_lr_scheduler = None

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=1000)
else:
    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters())
        
    cpName = checkpoint_dir + 'checkpointAllEpochs.tar'
    if os.path.isfile(cpName):
        print("=> loading checkpoint '{}'".format(cpName))
        checkpoint = torch.load(cpName)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        loss_list = checkpoint['loss_list']
        loss_list_val = checkpoint['loss_list_val']
        loss_1_list = checkpoint['loss_1_list']
        loss_1_list_val = checkpoint['loss_1_list_val']
        loss_2_list = checkpoint['loss_2_list']
        loss_2_list_val = checkpoint['loss_2_list_val']
        loss_mse_list = checkpoint['loss_mse_list']
        loss_mse_list_val = checkpoint['loss_mse_list_val']
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
                           loss_list=loss_list, loss_list_val=loss_list_val,
                           loss_1_list=loss_1_list, loss_1_list_val=loss_1_list_val,
                           loss_2_list=loss_2_list, loss_2_list_val=loss_2_list_val,
                           loss_mse_list=loss_mse_list, loss_mse_list_val=loss_mse_list_val)
    
if train != True:
    visualize_model(model_ft, tort = 'Train')
    visualize_model(model_ft, tort = 'Val')

    fig = plt.figure()
    plt.plot(loss_list[2:], color='red')
    plt.plot(loss_list_val[2:], color='blue')
    
    fig = plt.figure()
    plt.plot(loss_1_list[2:], color='red')
    plt.plot(loss_1_list_val[2:], color='blue')
    
    fig = plt.figure()
    plt.plot(loss_2_list[2:], color='red')
    plt.plot(loss_2_list_val[2:], color='blue')
    
    fig = plt.figure()
    plt.plot(loss_mse_list[2:], color='red')
    plt.plot(loss_mse_list_val[2:], color='blue')
    
    np.save(save_dir + 'loss_list.npy', loss_list)
    np.save(save_dir + 'loss_list_val.npy', loss_list_val)
    np.save(save_dir + 'loss_1_list.npy', loss_1_list)
    np.save(save_dir + 'loss_1_list_val.npy', loss_1_list_val)
    np.save(save_dir + 'loss_2_list.npy', loss_2_list)
    np.save(save_dir + 'loss_2_list_val.npy', loss_2_list_val)
    np.save(save_dir + 'loss_mse_list.npy', loss_mse_list)
    np.save(save_dir + 'loss_mse_list_val.npy', loss_mse_list_val)
    

raw_input('Press enter to continue: ')
