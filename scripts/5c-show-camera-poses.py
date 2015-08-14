#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import numpy as np
import os.path
import random
from visual import *

import navpy

sys.path.append('../lib')
import Pose
import ProjectMgr
import transformations

# for all the images in the project image_dir, compute the camera poses from
# the aircraft pose (and camera mounting transform)

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--ground', type=float, help='ground elevation in meters')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()            # for image dimensions

# place objects at alt - ground level (so z=0 corresponds to ground
# elevation) to help the world fit better in the window if specified
g = 0.0
if args.ground:
    g = args.ground
    
draw_ref = True
axis_len = 20
if draw_ref:
    # draw (0,0,0) reference location
    p = arrow(pos=(0,0,0), axis=(axis_len,0,0), shaftwidth=1, up=(0,0,1),
              color=color.red)
    p = arrow(pos=(0,0,0), axis=(0,axis_len,0), shaftwidth=1, up=(0,0,1),
              color=color.green)
    p = arrow(pos=(0,0,0), axis=(0,0,axis_len), shaftwidth=1, up=(0,0,1),
              color=color.cyan)

# draw aircraft locations and orientations
ref = proj.ned_reference_lla
for image in proj.image_list:
    lla = image.aircraft_pose['lla']
    ned = navpy.lla2ned( lla[0], lla[1], lla[2], ref[0], ref[1], ref[2] )
    quat = image.aircraft_pose['quat']
    # forward vector in ned
    f = transformations.quaternion_backTransform(quat, [7.0, 0.0, 0.0])
    # up vector in ned
    up = transformations.quaternion_backTransform(quat, [0.0, 0.0, -5.0])
    p = arrow(pos=(ned[1],ned[0],-ned[2]-g), axis=(f[1], f[0], -f[2]),
              up=(up[1], up[0], -up[2]), shaftwidth=1, color=color.yellow)
    p = arrow(pos=(ned[1],ned[0],-ned[2]-g), axis=(up[1], up[0], -up[2]),
              up=(f[1], f[0], -f[2]),shaftwidth=1, color=color.yellow)

# draw camera locations and orientations, notice pos=() is the 'base'
# of the pyramid and not the point so we have to project out a target
# base location based on camera orientation and invert the axis.
for image in proj.image_list:
    ned = image.camera_pose['ned']
    quat = image.camera_pose['quat']
    lens = proj.cam.get_lens_params()
    # position vector in ned
    pos = transformations.quaternion_backTransform(quat, [lens[2], 0.0, 0.0])
    # forward vector in ned
    f = transformations.quaternion_backTransform(quat, [-1.0, 0.0, 0.0])
    # up vector in ned
    up = transformations.quaternion_backTransform(quat, [0.0, 0.0, -1.0])
    p = pyramid(pos=(ned[1]+pos[1],ned[0]+pos[0],-(ned[2]+pos[2])-g),
                size=(lens[2], lens[1], lens[0]),
                axis=(f[1], f[0], -f[2]), up=(up[1], up[0], -up[2]),
                color=color.orange, opacity=1.0)

camw, camh = proj.cam.get_image_params()
fx, fy, cu, cv, dist_coeffs, skew = proj.cam.get_calibration_params()
for image in proj.image_list:
    print image.name
    # scale the K matrix if we have scaled the images
    scale = float(image.width) / float(camw)
    #print image.width, camw, scale
    K = np.array([ [fx*scale, skew*scale, cu*scale],
                   [ 0,       fy  *scale, cv*scale],
                   [ 0,       0,          1       ] ], dtype=np.float32)
    IK = np.linalg.inv(K)

    corner_list = []
    corner_list.append( [0, 0] )
    corner_list.append( [image.width, 0] )
    corner_list.append( [image.width, image.height] )
    corner_list.append( [0, image.height] )
    
    quat = image.camera_pose['quat']
    proj_list = proj.projectVectors( IK, quat, corner_list )
    #print "proj_list:\n", proj_list
    pts = proj.intersectVectorsWithGround(image.camera_pose['ned'],
                                          g, proj_list)
    #print "pts (ned):\n", pts
    cart = []
    for ned in pts:
        cart.append( [ned[1], ned[0], -ned[2]-g] )
    #print "cart:\n", cart
    
    # two faces makes a quad
    r = random.random()*0.5
    b = random.random()*0.5
    mycolor=(r, 1.0, b)
    #print mycolor
    f = faces()
    f.append( pos=cart[0], normal=(0,0,1), color=mycolor )
    f.append( pos=cart[1], normal=(0,0,1), color=mycolor )
    f.append( pos=cart[2], normal=(0,0,1), color=mycolor )
    f.append( pos=cart[0], normal=(0,0,1), color=mycolor )
    f.append( pos=cart[2], normal=(0,0,1), color=mycolor )
    f.append( pos=cart[3], normal=(0,0,1), color=mycolor )
    #f.make_twosided()
