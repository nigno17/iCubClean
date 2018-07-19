from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import noise
import random
import xml.etree.ElementTree as ET


plt.ion()   # interactive mode

res = 5

channels = 3

root_dir = 'Corrected_dataset4/'
root_dir_img = root_dir + 'new_dataset_jaeseok_cut/'
root_dir_labels = root_dir + 'lables_cut/'

file_list = os.listdir(root_dir_img)

dataset_dim = len(file_list)
augmentation_num = 20
plain_back_num = augmentation_num / 2


def normalizeImg(image):
    minVal = np.min(image)
    image = image - minVal
    maxVal = np.max(image) 
    image = image / maxVal
    
    return image

threshold = 20.0

for rep in range(augmentation_num + 1):
    for image_it in range(dataset_dim):
        persistence = random.random()
        lacunarity = random.random() * 5
        octaves = int(random.random() * 5 + 1)
        
        img_count = rep * dataset_dim + image_it
        image = io.imread(root_dir_img + str(image_it) + '.png')
        tree = ET.parse(root_dir_labels + str(image_it) + '.xml')
        root = tree.getroot()
        
        img_size = image.shape
        
        if img_count < (augmentation_num * dataset_dim):         
            segmentation_image = np.zeros((image.shape[0], image.shape[1]))
            for bndbox in root.iter('bndbox'):
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                for i in range(segmentation_image.shape[0]):
                    for j in range(segmentation_image.shape[1]):
                        if j >= xmin and j < xmax and i >= ymin and i < ymax:
                            segmentation_image[i, j] = 1
            
            
            print ('octaves: ' + str(octaves) + \
                   ' persistence: ' + str(persistence) + \
                   ' lacunarity: ' + str(lacunarity) + \
                   ' iteration: ' + str(img_count))
            
            noiseImg = np.zeros((img_size[0], img_size[1], channels))
            if img_count < (plain_back_num * dataset_dim):
                noiseImg = np.ones(img_size)
                noiseImg[:, :, 0] = noiseImg[:, :, 0] * random.random()
                noiseImg[:, :, 1] = noiseImg[:, :, 1] * random.random()
                noiseImg[:, :, 2] = noiseImg[:, :, 2] * random.random()
            else:    
                for channel in range(channels):
                    basex = random.random() * img_size[1]
                    basey = random.random() * img_size[0]
                    for i in range(img_size[1]):
                        for j in range(img_size[0]):
                            x = (i / img_size[1] * res) + basex
                            y = (j / img_size[0] * res) + basey
                            noiseImg[j, i, channel] = noise.pnoise2(x, y, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=img_size[1], repeaty=img_size[0], base=0)
                noiseImg = normalizeImg(noiseImg)
            
            
            
            final_image = np.zeros((img_size[0], img_size[1], channels))
            for channel in range(channels):
                for i in range(img_size[1]):
                    for j in range(img_size[0]):
                        if segmentation_image[j, i] == 1:
                            final_image[j, i, channel] = image[j, i, channel] / 255
                        else:
                            final_image[j, i, channel] = noiseImg[j, i, channel]
        else:
            final_image = image / 255
                
        
        image_name = str(img_count) + '.png' 
        image_path = root_dir + 'aug_dataset_cut/'
        
        io.imsave(image_path + image_name, final_image)
        
        label_name = str(img_count) + '.xml' 
        label_path = root_dir + 'aug_labels_cut/'
        
        for path in root.iter('path'):
            path.text = label_path + image_name
        for filename in root.iter('filename'):
            filename.text = image_name
        
        tree.write(label_path + label_name)
        
#        plt.figure()
#        plt.imshow(image)
#        plt.figure()
#        gray_to_print = cropped_img
#        for i in range(3):
#            gray_to_print[:, :, i] = gray_img
#        plt.imshow(gray_to_print)
#        plt.figure()
#        plt.imshow(normalizeImg(gray_to_print))
#        plt.figure()
#        plt.imshow(noiseImg)
#        raw_input('Press enter to continue: ')


raw_input('Press enter to continue: ')

