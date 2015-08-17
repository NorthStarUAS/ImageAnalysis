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
from PIL import Image
import navpy

sys.path.append('../lib')
import Pose
import ProjectMgr
import SRTM
import transformations

# for all the images in the project image_dir, compute the camera poses from
# the aircraft pose (and camera mounting transform)

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')
#parser.add_argument('--ground', type=float, help='ground elevation in meters')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()            # for image dimensions

ref = proj.ned_reference_lla

# setup SRTM ground interpolator
sss = SRTM.NEDGround( ref, 2000, 2000, 30 )

# lookup reference ground altitude (lla reference is [0,0,0] in ned frame)
g = sss.interp([0.0, 0.0])[0]
print "Reference ground elevation is:", g

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

# draw approximate image areas 'direct georectified'
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
    #pts = proj.intersectVectorsWithGroundPlane(image.camera_pose,
    #                                           g, proj_list)
    pts = sss.interpolate_vectors(image.camera_pose, proj_list)
    #print "pts (ned):\n", pts
    
    cart = []
    for ned in pts:
        cart.append( [ned[1], ned[0], -ned[2]-g] )
    #print "cart:\n", cart

    # I haven't figure out how to control the texture coordinates, it
    # doesn't appear that there is any support for this at all. :-(
    #im = Image.open(image.image_file)
    #im = im.resize((128,128), Image.ANTIALIAS)
    
    # two faces makes a quad
    mycolor=(random.random()*0.5,
             random.random()*0.25+0.75,
             random.random()*0.5)
    vertices = [ cart[0], cart[1], cart[2] ]
    #uvmap = [ [0.0, 0.0], [1.0, 0.0], [1.0, 1.0] ]
    #tex = materials.texture(data=im)
    f = faces( pos=vertices, color=mycolor )
    f.make_normals()

    vertices = [ cart[0], cart[2], cart[3] ]
    #uvmap = [ [1.0, 0.0], [1.0, 1.0], [0.0, 1.0] ]
    #tex = materials.texture(data=im)
    f = faces( pos=vertices, color=mycolor)
    f.make_normals()

    #im = Image.open(image.image_file)
    #im = im.resize((128,128), Image.ANTIALIAS)
    #tex = materials.texture(data=im, mapping="sign")
    #f.material = tex
