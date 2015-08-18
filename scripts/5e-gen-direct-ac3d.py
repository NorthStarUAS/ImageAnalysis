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
import navpy

import simplekml

sys.path.append('../lib')
import AC3D
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

ac3d_steps = 1

# compute the uv grid for each image and project each point out into
# ned space, then intersect each vector with the srtm ground.

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

    quat = image.camera_pose['quat']

    grid_list = []
    u_list = np.linspace(0, image.width, ac3d_steps + 1)
    v_list = np.linspace(0, image.height, ac3d_steps + 1)
    #print "u_list:", u_list
    #print "v_list:", v_list
    for v in v_list:
        for u in u_list:
            grid_list.append( [u, v] )
    
    proj_list = proj.projectVectors( IK, quat, grid_list )
    #print "proj_list:\n", proj_list
    pts_ned = sss.interpolate_vectors(image.camera_pose, proj_list)
    #print "pts_3d (ned):\n", pts_ned

    # convert ned to xyz and stash the result for each image
    image.grid_list = []
    for p in pts_ned:
        image.grid_list.append( [p[1], p[0], -p[2]] )
    
# call the ac3d generator
AC3D.generate(proj.image_list, image_dir=args.project, base_name='direct', version=1.0, trans=0.0)
