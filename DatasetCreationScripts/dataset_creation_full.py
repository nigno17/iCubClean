from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw
import noise
import random
import xml.etree.ElementTree as ET
from DataLoading import Shift, ChangeLight


plt.ion()   # interactive mode

res = 5

channels = 3

root_dir = 'Corrected_dataset8/'
root_dir_img = root_dir + 'new_dataset_jaeseok_cut/'
root_dir_labels = root_dir + 'lables_cut/'
root_old_traj_dir = root_dir + 'rightarm_modified_trajectory_cut/'

file_list = os.listdir(root_dir_img)

dataset_dim = len(file_list)
augmentation_num = 20
originals_num = 10


def normalizeImg(image):
    minVal = np.min(image)
    image = image - minVal
    maxVal = np.max(image) 
    image = image / maxVal
    
    return image

threshold = 20.0

for rep in range(augmentation_num + 1):
    for image_it in range(dataset_dim):
        
        img_count = rep * dataset_dim + image_it

        image = io.imread(root_dir_img + str(image_it) + '.png')
        tree = ET.parse(root_dir_labels + str(image_it) + '.xml')
        root = tree.getroot()
        traj = np.loadtxt(root_old_traj_dir + 'data' + str(image_it + 1) + '.txt', dtype='float')
        
        img_size = image.shape
        
        if img_count < (augmentation_num * dataset_dim):
            temp_image = np.zeros((img_size[0], img_size[1], channels))
            temp_image = image / 255
            
            ### SHIFT PART

            trans = transforms.Compose([Shift(-1.0, 1.0, -1.0, 1.0),
                                        ChangeLight(0.15, percentage = 0.8)])
                                        
            trajTemp =  np.copy(traj)
                                                        
            repFlag = True
            while repFlag:
        
                sample = {'image': np.copy(temp_image), 'trajectories': np.copy(traj)}
                sample = trans(sample)
                
                trajTemp = sample['trajectories']
                
                diff = trajTemp[0,:] - traj[0,:]            
                diff *= img_size[0]
                
                repFlag = False
                x_mean = 0
                y_mean = 0
                x_min_bound = 1000
                y_min_bound = 1000
                x_max_bound = 0
                y_max_bound = 0
                count_BB = 0
                for bndbox in root.iter('bndbox'):
                    count_BB += 1                    
                    
                    xmin = int(bndbox.find('xmin').text) + diff[1]
                    ymin = int(bndbox.find('ymin').text) + diff[0]
                    xmax = int(bndbox.find('xmax').text) + diff[1]
                    ymax = int(bndbox.find('ymax').text) + diff[0]
                    
                    if x_min_bound > xmin:
                        x_min_bound = xmin
                    if y_min_bound > ymin:
                        y_min_bound = ymin
                    if x_max_bound < xmax:
                        x_max_bound = xmax
                    if y_max_bound < ymax:
                        y_max_bound = ymax
                    
                    x_mean += (xmin + xmax) / 2
                    y_mean += (ymin + ymax) / 2
                    
                    if (xmin < 0 or xmax >= img_size[1]) or \
                       (ymin < 0 or ymax >= img_size[0]):
                           repFlag = True
                x_mean /= count_BB
                y_mean /= count_BB
            
            temp_image = sample['image']            
            traj = sample['trajectories']

            ## END SHIFT PART
            if img_count < (originals_num * dataset_dim):
                final_image = temp_image
            else:

                segmentation_image = np.zeros((image.shape[0], image.shape[1]))
                for bndbox in root.iter('bndbox'):
                    xmin = int(bndbox.find('xmin').text) + diff[1]
                    ymin = int(bndbox.find('ymin').text) + diff[0]
                    xmax = int(bndbox.find('xmax').text) + diff[1]
                    ymax = int(bndbox.find('ymax').text) + diff[0]
                    for i in range(segmentation_image.shape[0]):
                        for j in range(segmentation_image.shape[1]):
                            if j >= xmin and j < xmax and i >= ymin and i < ymax:
                                segmentation_image[i, j] = 1  
                
                ## TABLE SEGMENTATION
#                new_x_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
#                new_y_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
#                p1 = (int((img_size[1] / 2) - (img_size[0] / 4)) + diff[1] + new_x_shift, (img_size[0]) + diff[0] + new_y_shift)
#                new_x_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
#                new_y_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
#                p2 = (int((img_size[1] / 2) - (img_size[0] / 4)) + diff[1] + new_x_shift, (img_size[0] / 2) + diff[0] + new_y_shift)
#                new_x_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
#                new_y_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
#                p3 = (int((img_size[1] / 2) + (img_size[0] / 4)) + diff[1] + new_x_shift, (img_size[0] / 2) + diff[0] + new_y_shift)
#                new_x_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
#                new_y_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
#                p4 = (int((img_size[1] / 2) + (img_size[0] / 4)) + diff[1] + new_x_shift, (img_size[0]) + diff[0] + new_y_shift)
                
                table_size = 120
                pixels_drift_x = table_size - (x_max_bound - x_min_bound)
                pixels_drift_y = table_size - (y_max_bound - y_min_bound)
                pixels_drift_small = 5
                
                new_x_shift_small = (random.random() * (pixels_drift_small * 2)) - pixels_drift_small
                new_y_shift_small = (random.random() * (pixels_drift_small * 2)) - pixels_drift_small
                new_x_shift = random.random() * pixels_drift_x
                new_y_shift = random.random() * pixels_drift_y
                p1 = ((x_min_bound - new_x_shift) + new_x_shift_small, (y_min_bound - new_y_shift) + new_y_shift_small)
                new_x_shift_small = (random.random() * (pixels_drift_small * 2)) - pixels_drift_small
                new_y_shift_small = (random.random() * (pixels_drift_small * 2)) - pixels_drift_small
                p2 = ((x_min_bound - new_x_shift) + new_x_shift_small, (y_min_bound - new_y_shift) + table_size + new_y_shift_small)
                new_x_shift_small = (random.random() * (pixels_drift_small * 2)) - pixels_drift_small
                new_y_shift_small = (random.random() * (pixels_drift_small * 2)) - pixels_drift_small
                p3 = ((x_min_bound - new_x_shift) + table_size + new_x_shift_small, (y_min_bound - new_y_shift) + table_size + new_y_shift_small)
                new_x_shift_small = (random.random() * (pixels_drift_small * 2)) - pixels_drift_small
                new_y_shift_small = (random.random() * (pixels_drift_small * 2)) - pixels_drift_small
                p4 = ((x_min_bound - new_x_shift) + table_size + new_x_shift_small, (y_min_bound - new_y_shift) + new_y_shift_small)
                
                polygon = [p1, p2, p3, p4]
                
                table_image_pil = Image.new('L', (img_size[1], img_size[0]), 0)
                ImageDraw.Draw(table_image_pil).polygon(polygon, outline=1, fill=1)
                table_image = np.array(table_image_pil)
                
                ## END TABLE SEGMENTATION
                
                ## FOV SEGMENTATION                
                
                pixels_drift = 30
                
                new_x_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
                new_y_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
                p1 = (x_mean - (img_size[1] / 3) + new_x_shift, y_mean - (img_size[0]/ 3) + new_y_shift)
                new_x_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
                new_y_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
                p2 = (x_mean - (img_size[1] / 3) + new_x_shift, y_mean + (img_size[0]/ 3) + new_y_shift)
                new_x_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
                new_y_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
                p3 = (x_mean + (img_size[1] / 3) + new_x_shift, y_mean + (img_size[0]/ 3) + new_y_shift)
                new_x_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
                new_y_shift = (random.random() * (pixels_drift * 2)) - pixels_drift
                p4 = (x_mean + (img_size[1] / 3) + new_x_shift, y_mean - (img_size[0]/ 3) + new_y_shift)
                
                polygon = [p1, p2, p3, p4]
                
                fov_image_pil = Image.new('L', (img_size[1], img_size[0]), 0)
                ImageDraw.Draw(fov_image_pil).polygon(polygon, outline=1, fill=1)
                fov_image = np.array(fov_image_pil)
                
                ## END FOV SEGMENTATION
                
                persistence = random.random()
                lacunarity = random.random() * 5
                octaves = int(random.random() * 5 + 1)
                
                noiseTableImg = np.zeros((img_size[0], img_size[1], channels))
                for channel in range(channels):
                    basex = random.random() * img_size[1]
                    basey = random.random() * img_size[0]
                    for i in range(img_size[1]):
                        for j in range(img_size[0]):
                            x = (i / img_size[1] * res) + basex
                            y = (j / img_size[0] * res) + basey
                            noiseTableImg[j, i, channel] = noise.pnoise2(x, y, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=img_size[1], repeaty=img_size[0], base=0)
                noiseTableImg = normalizeImg(noiseTableImg)
                
                white_ratio = 0.6
                
                noiseTableImg = (noiseTableImg * (1 - white_ratio)) + (np.ones((img_size[0], img_size[1], channels)) * white_ratio)
                
                persistence = random.random()
                lacunarity = random.random() * 5
                octaves = int(random.random() * 5 + 1)
                
                noiseImg = np.zeros((img_size[0], img_size[1], channels))
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
                                final_image[j, i, channel] = temp_image[j, i, channel]
                            elif fov_image[j, i] == 1:
                                final_image[j, i, channel] = noiseImg[j, i, channel]
                                if table_image[j, i] == 1:
                                    final_image[j, i, channel] = noiseTableImg[j, i, channel]
        else:
            final_image = image / 255
        
        print (' iteration: ' + str(img_count))
            
        
        traj_name = str(img_count) + '.txt'
        root_new_traj_dir = root_dir + 'aug_trajectories_cut/'
        np.savetxt(root_new_traj_dir + traj_name, traj)
                
        
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

