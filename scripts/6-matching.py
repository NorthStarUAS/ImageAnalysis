#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import math
import numpy as np
import os.path
import scipy.spatial

sys.path.append('../lib')
import Matcher
import Pose
import ProjectMgr
import SRTM

# working on matching features ...

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

# setup SRTM ground interpolator
ref = proj.ned_reference_lla
sss = SRTM.NEDGround( ref, 2000, 2000, 30 )

# project undistorted keypoints into NED space
camw, camh = proj.cam.get_image_params()
fx, fy, cu, cv, dist_coeffs, skew = proj.cam.get_calibration_params()
for image in proj.image_list:
    print "Projecting keypoints to vectors:", image.name
    scale = float(image.width) / float(camw)
    K = np.array([ [fx*scale, skew*scale, cu*scale],
                   [ 0,       fy  *scale, cv*scale],
                   [ 0,       0,          1       ] ], dtype=np.float32)
    IK = np.linalg.inv(K)
    quat = image.camera_pose['quat']
    image.vec_list = proj.projectVectors(IK, quat, image.uv_list)

# intersect keypoint vectors with srtm terrain
for image in proj.image_list:
    print "Intersecting keypoint vectors with terrain:", image.name
    image.coord_list = sss.interpolate_vectors(image.camera_pose,
                                               image.vec_list)
    
# build kdtree() of 3d point locations for fast spacial nearest
# neighbor lookups.
print "Notice: constructing KDTree's"
for image in proj.image_list:
    image.kdtree = scipy.spatial.KDTree(image.coord_list)

    #result = image.kdtree.query_ball_point(image.coord_list[0], 5.0)
    #p1 = image.coord_list[0]
    #print "ref =", p1
    #for i in result:
    #    p2 = image.coord_list[i]
    #    d1 = p1[0] - p2[0]
    #    d2 = p1[1] - p2[1]
    #    dist = math.sqrt(d1**2 + d2**2)
    #    print "dist=%.2f  coord=%s" % (dist, p2)

# fire up the matcher
m = Matcher.Matcher()
