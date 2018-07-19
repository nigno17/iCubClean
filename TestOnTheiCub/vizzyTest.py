#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:51:24 2017

@author: nigno
"""

from __future__ import print_function, division

import numpy as np
import math
import yarp
import time
import python_simworld_control as psc
from trajPredictor import trajPredictor, loadModel
from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.pylab


def axis2dcm(v):
    R = np.eye(4)
    
    theta = v[3]
    print(theta)
    if theta == 0.0:
        return R
    
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c
    
    xs = v[0] * s
    ys = v[1] * s
    zs = v[2] * s
    xC = v[0] * C
    yC = v[1] * C
    zC = v[2] * C
    xyC = v[0] * yC
    yzC = v[1] * zC
    zxC = v[2] * xC
    
    R[0,0] = v[0] * xC + c
    R[0,1] = xyC - zs
    R[0,2] = zxC + ys
    R[1,0] = xyC + zs
    R[1,1] = v[1] * yC + c
    R[1,2] = yzC - xs
    R[2,0] = zxC - ys
    R[2,1] = yzC + xs
    R[2,2] = v[2] * zC + c
    
    return R
 
def dcm2axis(R):
    v = np.zeros(4)
    v[0] = R[2,1] - R[1,2]
    v[1] = R[0,2] - R[2,0]
    v[2] = R[1,0] - R[0,1]
    v[3] = 0.0
    r = np.linalg.norm(v)
    theta = math.atan2(0.5 * r, 
                       0.5 * (R[0,0] + R[1,1] + R[2,2] - 1))
    
    print (R)
    print (R[:3, :3])
    
    if r < 1e-9:
       # if we enter here, then
       # R is symmetric; this can
       # happen only if the rotation
       # angle is 0 (R=I) or 180 degrees
       A = R[:3, :3]

       # A=I+sin(theta)*S+(1-cos(theta))*S^2
       # where S is the skew matrix.
       # Given a point x, A*x is the rotated one,
       # hence if Ax=x then x belongs to the rotation
       # axis. We have therefore to find the kernel of
       # the linear application (A-I).
       U, S, V = np.linalg.svd(A - np.eye(3))

       v[0] = V[0,2]
       v[1] = V[1,2]
       v[2] = V[2,2]
       r = np.linalg.norm(v)
     
    v = (1.0 / r ) * v
    v[3] = theta;
     
    return v

model = loadModel()

yarp.Network_init();

def from_yarp_rgb_Image_to_numpy_array(yarp_image,numpy_array):
    for l1 in range(0,yarp_image.height()):
        for l2 in range(0,yarp_image.width()):
            currPixel = yarp.PixelRgb()
            currPixel= yarp_image.pixel(l2,l1)
            numpy_array[l1][l2][0] = currPixel.r
            numpy_array[l1][l2][1] = currPixel.g
            numpy_array[l1][l2][2] = currPixel.b

# Create a port and connect it to the iCub simulator virtual camera
input_port = yarp.BufferedPortImageRgb()
input_port.open("/homo_image_read")
yarp.Network.connect("/perspectiveChanger/img:io", "/homo_image_read")


# define the height
height = 0.0

# initialize the world controller
wc = psc.WorldController()
wc.del_all()
# spawn table
thick = 0.05
pos_rob = np.array([-0.25, 0, height - (thick / 2), 1])
T = np.array([[0, -1, 0, 0], 
              [0, 0, 1, 0.5976],
              [-1, 0, 0, -0.026],
              [0, 0, 0, 1]])
pos = np.dot(T, pos_rob)[:3]
dim = [0.5, thick, 0.5]
print(pos)

color = [0.5, 1, 0]

thick_dirt = thick / 2.0
dim_dirt = [0.05, (thick / 2) + (thick_dirt / 2.0), 0.05]
pos_dirt = pos + [0.05, thick_dirt, 0.1]
color_dirt = [0.6, 0.15 , 0.2]

white_table = wc.create_object('sbox', dim, pos, color)
red_square = wc.create_object('sbox', dim_dirt, pos_dirt, color_dirt)

raw_input('Press enter to continue: ')

## prepare a property object
#props = yarp.Property()
#props.put("device","remote_controlboard")
#props.put("local","/client/right_arm")
#props.put("remote","/icubSim/right_arm")
#
#props_left = yarp.Property()
#props_left.put("device","remote_controlboard")
#props_left.put("local","/client/left_arm")
#props_left.put("remote","/icubSim/left_arm")
#
#props_torso = yarp.Property()
#props_torso.put("device","remote_controlboard")
#props_torso.put("local","/client/torso")
#props_torso.put("remote","/icubSim/torso")
#
#propsKC = yarp.Property()
#propsKC.put("device","cartesiancontrollerclient")
#propsKC.put("local","/clientKC/right_arm")
#propsKC.put("remote","/icubSim/cartesianController/right_arm")
#
#propsKC_left = yarp.Property()
#propsKC_left.put("device","cartesiancontrollerclient")
#propsKC_left.put("local","/clientKC/left_arm")
#propsKC_left.put("remote","/icubSim/cartesianController/left_arm")
#
## create remote driver
#armDriver = yarp.PolyDriver(props)
#armDriver_left = yarp.PolyDriver(props_left)
#torsoDriver = yarp.PolyDriver(props_torso)
#clientCartCtrl = yarp.PolyDriver(propsKC)
#clientCartCtrl_left = yarp.PolyDriver(propsKC_left)
#
#if (clientCartCtrl.isValid()):
#   icart = clientCartCtrl.viewICartesianControl()
#   icart_left = clientCartCtrl_left.viewICartesianControl()
#
#
##query motor control interfaces
#iPos = armDriver.viewIPositionControl()
#iEnc = armDriver.viewIEncoders()
#iPos_left = armDriver_left.viewIPositionControl()
#iEnc_left = armDriver_left.viewIEncoders()
#iPos_torso = torsoDriver.viewIPositionControl()
#iEnc_torso = torsoDriver.viewIEncoders()
#
##retrieve number of joints
#jnts = iPos.getAxes()
#print ('Controlling ' + str(jnts) + ' joints')
#jnts_torso = iPos_torso.getAxes()
#print ('Controlling ' + str(jnts_torso) + ' joints')
#
## read encoders
#encs = yarp.Vector(jnts)
#iEnc.getEncoders(encs.data())
#encs_left = yarp.Vector(jnts)
#iEnc_left.getEncoders(encs_left.data())
#encs_torso = yarp.Vector(jnts_torso)
#iEnc_torso.getEncoders(encs_torso.data())
#
## store as home position
#home = yarp.Vector(jnts, encs.data())
#home_left = yarp.Vector(jnts, encs_left.data())
#home_torso = yarp.Vector(jnts_torso, encs_torso.data())
#
#home.set(0, -80)
#home.set(1, 70)
#home.set(2, 0)
#home.set(3, 80)
#
#home_left.set(0, -80)
#home_left.set(1, 70)
#home_left.set(2, 0)
#home_left.set(3, 80)

for loop in range(20):
    # move to new position

#    iMode = armDriver.viewIControlMode()
#    for j in range(jnts):
#        iMode.setPositionMode(j)
#    
#    home_torso.set(0, 0)
#    home_torso.set(1, 0)
#    home_torso.set(2, -10)
#    
#    iPos_torso.positionMove(home_torso.data())
#    
#    time.sleep(2)
#    
#    #iPos.positionMove(home.data())
#    #iPos_left.positionMove(home_left.data())
#    
#    time.sleep(2)
#    
#    home_torso.set(0, 0)
#    home_torso.set(1, 0)
#    home_torso.set(2, 0)
#    
#    iPos_torso.positionMove(home_torso.data())
#    
#    time.sleep(2)

    raw_input('Press enter to continue: ')
    
    # network part
    image = io.imread('test_set/31.ppm')
    yarp_image2 = yarp.ImageRgb()
    yarp_image2.resize(320, 240)
                
    #input_port.read(yarp_image2)
    yarp_image2 = input_port.read(False)
    while yarp_image2 == None:
        print ('None')        
        time.sleep(0.1)
        yarp_image2 = input_port.read(False)       
    
    from_yarp_rgb_Image_to_numpy_array(yarp_image2, image)
    
    traj = trajPredictor(model, image)
    
    print(traj)
    
    plt.imshow(image, origin='upper')
    plt.axis('off')
    
    img_out_data = traj.copy()
    
    img_out_data[:, 0] = (img_out_data[:, 0] + 1.0) * 240
    img_out_data[:, 1] = (img_out_data[:, 1] + 0.75) * 240
    
    img_out_data[:, 0] -= 0
    
    #imshow(inputs.cpu().data[j])
    plt.scatter(img_out_data[:, 1], img_out_data[:, 0], s=10, marker='.', color='blue')
    plt.show()
    
    
    
    
    
    # perform the trajectory
    
#    # set up the orientation
#    o_init = yarp.Vector(4)
#    o_x = np.zeros(4)
#    o_y = np.zeros(4)
#    o_z = np.zeros(4)
#    
#    o_x[0] = 1.0
#    o_x[3] = math.pi
#    o_y[1] = 1.0
#    o_y[3] = 0.0
#    o_z[2] = 1.0
#    o_z[3] = 0.0
#    
#    Rx = axis2dcm(o_x)
#    Ry = axis2dcm(o_y)
#    Rz = axis2dcm(o_z)
#    
#    R = np.dot(Rz, np.dot(Ry, Rx))
#    
#    print (R)
#    
#    o = dcm2axis(R)
#    print('Old O: ' + str(o_x) + '. New O: ' + str(o))
#    
#    o_init.set(0, o[0])
#    o_init.set(1, o[1])
#    o_init.set(2, o[2])
#    o_init.set(3, o[3])
#    
#    # first move the arm to the desired position
#    x_init = yarp.Vector(3)
#    
#    x_init.set(0, float(traj[0, 0]))
#    x_init.set(1, float(traj[0, 1]))
#    x_init.set(2, height + 0.05)
#    
#    icart.goToPoseSync(x_init, o_init)
#    
#    done=False
#    while done == False:
#       done = icart.checkMotionDone()
#       time.sleep(0.01)
#    
#    
#    # then let's performe the trajectory at 100 hertz
#    for traj_it in traj[1:]:
#        x_init.set(0, float(traj_it[0]))
#        x_init.set(1, float(traj_it[1]))
#        x_init.set(2, height +0.05)
#        
#        icart.goToPose(x_init, o_init)
#        time.sleep(0.01)
#    
#    icart.goToPoseSync(x_init, o_init)
#    done=False
#    while done == False:
#       done = icart.checkMotionDone()
#       time.sleep(0.01)
    

clientCartCtrl.close()
clientCartCtrl_left.close()
input_port.close()

