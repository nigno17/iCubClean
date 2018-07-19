#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:51:24 2017

@author: nigno
"""

from __future__ import print_function, division

import numpy as np

import yarp
import time
from skimage import io, transform
from skimage.measure import compare_ssim
import matplotlib.pylab as plt
from PIL import Image
import cv2
import os.path

yarp.Network_init();

def from_yarp_rgb_Image_to_numpy_array(yarp_image,numpy_array):
    for l1 in range(0,yarp_image.height()):
        for l2 in range(0,yarp_image.width()):
            currPixel = yarp.PixelRgb()
            currPixel= yarp_image.pixel(l2,l1)
            numpy_array[l1][l2][0] = currPixel.r
            numpy_array[l1][l2][1] = currPixel.g
            numpy_array[l1][l2][2] = currPixel.b

# define the height
height = -0.01

# prepare a property object
props = yarp.Property()
props.put("device","remote_controlboard")
props.put("local","/client/head")
props.put("remote","/icubSim/head")

props_torso = yarp.Property()
props_torso.put("device","remote_controlboard")
props_torso.put("local","/client/torso")
props_torso.put("remote","/icubSim/torso")

output_port = yarp.Port()
output_port.open("/dc-image")

input_port = yarp.BufferedPortImageRgb()
input_port.open("/homo_image_read")

lambda_port = yarp.BufferedPortBottle()
lambda_port.open("/lambda_read")

# create remote driver
headDriver = yarp.PolyDriver(props)
torsoDriver = yarp.PolyDriver(props_torso)

#query motor control interfaces
iPos = headDriver.viewIPositionControl()
iEnc = headDriver.viewIEncoders()
iPos_torso = torsoDriver.viewIPositionControl()
iEnc_torso = torsoDriver.viewIEncoders()

#retrieve number of joints
jnts = iPos.getAxes()
print ('Controlling ' + str(jnts) + ' joints')
jnts_torso = iPos_torso.getAxes()
print ('Controlling ' + str(jnts_torso) + ' joints')

# read encoders
encs = yarp.Vector(jnts)
iEnc.getEncoders(encs.data())
encs_torso = yarp.Vector(jnts_torso)
iEnc_torso.getEncoders(encs_torso.data())

# store as home position
home = yarp.Vector(jnts, encs.data())
home_torso = yarp.Vector(jnts_torso, encs_torso.data())

raw_input('Press enter to continue: ')

yarp.Network.connect("/perspectiveChanger/img:io", "/homo_image_read")
yarp.Network.connect("/perspectiveChanger/matrix:o", "/lambda_read")
#yarp.Network.connect("/icubSim/cam/right", "/homo_image_read")

end_loop = False
loop = 0
loop_encoders = 0
dir_count = 1
total_elements = 0
headEncoders = []
torsoEncoders = []
while end_loop == False:
    # move to new position

    print(total_elements)

    main_root = '/media/nigno/Data/for_dataset/data' + str(dir_count) + '/'
    #'data-prova-jaeseok/'
    
    if loop == 0:
        name_counter = ''
    elif loop < 10:
        name_counter = '_0000' + str(loop)
    elif loop < 100:
        name_counter = '_000' + str(loop)
    elif loop < 1000:
        name_counter = '_00' + str(loop)
    elif loop < 10000:
        name_counter = '_0' + str(loop)
        
    if loop_encoders == 0:
        name_counter_encoders = ''
    elif loop_encoders < 10:
        name_counter_encoders = '_0000' + str(loop_encoders)
    elif loop_encoders < 100:
        name_counter_encoders = '_000' + str(loop_encoders)
    elif loop_encoders < 1000:
        name_counter_encoders = '_00' + str(loop_encoders)
    elif loop_encoders < 10000:
        name_counter_encoders = '_0' + str(loop_encoders)
        
    head_root = 'head/head'
    torso_root = 'torso/torso'
    head_name = main_root + head_root + name_counter_encoders + '/data.log'
    torso_name = main_root + torso_root + name_counter_encoders + '/data.log'
    image_root = 'images/right'
    image_name = main_root + image_root + name_counter + '/00000000.ppm'
    
    if (os.path.isfile(torso_name) and os.path.isfile(image_name) and os.path.isfile(head_name)): 
        head_file = open(head_name, 'r')
        head_line = head_file.readline()
        head = map(float, head_line.split())
        for j_it in range(jnts):
            home.set(j_it, head[j_it + 2])
        
        headEncoders.append(head[2:])
         
        torso_file = open(torso_name, 'r')
        torso_line = torso_file.readline()
        torso = map(float, torso_line.split())
        for j_it in range(jnts_torso):
            home_torso.set(j_it, torso[j_it + 2])
        
        torsoEncoders.append(torso[2:])
        
        iPos_torso.positionMove(home_torso.data())    
        iPos.positionMove(home.data())
        
        while (iPos.checkMotionDone() == False):
            time.sleep(0.01)
        while (iPos_torso.checkMotionDone() == False):
            time.sleep(0.01)
        
        #time.sleep(0.1)
        
        # network part
        #image = io.imread(image_name)
        
        image_name2 = '/media/nigno/Data/right_arm_data/right_original_image/' + str(total_elements + 1) + '.ppm'
        if os.path.isfile(image_name2): 
            cv_image2 = cv2.imread(image_name2)
            image2 = cv2.cvtColor(cv_image2, cv2.COLOR_BGR2RGB)
            
            cv_image = cv2.imread(image_name)
            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            diff = image - image2
            score = np.mean(diff) 
        
            if score < 5:      
                #plt.imshow(image)
                #plt.show()      
                #if  total_elements == 440:
                
                # Create the yarp image and wrap it around the array  
                yarp_image = yarp.ImageRgb()
                yarp_image.setExternal(image, image.shape[1], image.shape[0])   
                output_port.write(yarp_image)
                
                time.sleep(0.1)
                
                yarp_image2 = yarp.ImageRgb()
                yarp_image2.resize(image.shape[1], image.shape[0])
                            
                #input_port.read(yarp_image2)
                yarp_image2 = input_port.read(False)
                while yarp_image2 == None:
                    print ('None')
                    output_port.write(yarp_image)        
                    time.sleep(0.1)
                    yarp_image2 = input_port.read(False)       
                
                from_yarp_rgb_Image_to_numpy_array(yarp_image2, image)
                
                new_name = '/media/nigno/Data/new_dataset_jaeseok/' + str(total_elements) + '.png'
                
                im = Image.fromarray(image)
                im.save(new_name)
                
                lambdaMat = lambda_port.read(False)
                while lambdaMat == None:
                    lambdaMat = lambda_port.read(False)
                    
                lambdaMatStr = lambdaMat.toString()
                print(lambdaMat.toString())
                lambda_name = '/media/nigno/Data/Corrected_dataset/lambdas/' + str(total_elements) + '.txt'
                text_file = open(lambda_name, 'w')
                text_file.write(lambdaMatStr)
                text_file.close()
                
                #io.imsave(new_name, image)
                total_elements += 1        
        
        loop += 1
        loop_encoders +=1
    elif os.path.isfile(torso_name) and os.path.isfile(head_name):
        loop += 1
        if loop > (loop_encoders + 5):
            loop_encoders += 1
    elif dir_count < 3:
        loop = 0
        loop_encoders = 0
        dir_count += 1
    else:
        end_loop = True

print ('head:')
print (np.amax(np.asanyarray(headEncoders), axis=0))
print (np.amin(np.asanyarray(headEncoders), axis=0))
print ('torso:')
print (np.amax(np.asanyarray(torsoEncoders), axis=0))
print (np.amin(np.asanyarray(torsoEncoders), axis=0))
        
output_port.close()
input_port.close()
lambda_port.close()
        

