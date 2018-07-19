#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:51:24 2017

@author: nigno
"""

from __future__ import print_function, division

import numpy as np
import torch
import torch.optim as optim
import os
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from skimage import io, transform
from JaeseokNet import JaeseokNet

from GMR import GMR

use_gpu = torch.cuda.is_available()

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        new_h, new_w = self.output_size

        img = transform.resize(image, (new_h, new_w))

        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()
        
class Normalize(object):
    """Convert ndarrays in sample to Tensors.
    
    Args:
        mean: Mean of rgb channels
        std: Standard deviation of rgb channels
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        

    def __call__(self, image):

        image = (image - self.mean) / self.std
        
        return image
        
def loadModel():

    
    model_ft = JaeseokNet()
    optimizer_ft = optim.Adam(model_ft.parameters())
    
    print (model_ft)
    
    if use_gpu:
        model_ft = model_ft.cuda()
        
            
    cpName = 'checkpoint250.tar'
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
                  
    model_ft.train(False)
    
    return model_ft
    

def trajPredictor(model, image):  
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    transform_data = transforms.Compose([Rescale((240, 320)),
                                     #Normalize(mean, std),
                                     ToTensor()])
    if use_gpu:
        inputs = Variable(transform_data(image).cuda())
    else:
        inputs = Variable(transform_data(image))
    inputs = inputs.unsqueeze(0)
    
    
    outputs = model(inputs)
    traj = GMR(outputs[0])
    
    if use_gpu:
        nptraj = traj.data.cpu().numpy()
    else:
        nptraj = traj.data.numpy()
    
    return nptraj