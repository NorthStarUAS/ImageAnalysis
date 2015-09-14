#!/usr/bin/python

# 1. Iterate through all the image pairs and triangulate the match points.
# 2. Set the 3d location of features to triangulated position (possibly
#    averaged if the feature is included in multiple matches
# 3. Compute new camera poses with solvePnP() using triangulated point locations
# 4. Repeat

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import json
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os.path
from progress.bar import Bar
import scipy.spatial

sys.path.append('../lib')
import Matcher
import Pose
import ProjectMgr
import SRTM
import transformations

# working on matching features ...

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.load_matches()
proj.undistort_keypoints()

m = Matcher.Matcher()

f = open(args.project + "/Matches.json", 'r')
matches_dict = json.load(f)
f.close()

# iterate through the image list and build the camera pose dictionary
# (and a simple list of camera locations for plotting)
f = open( args.project + '/sba-cams.txt', 'w' )
for image in proj.image_list:
    body2cam = image.get_body2cam()
    ned2body = image.get_ned2body()
    Rtotal = body2cam.dot( ned2body )
    q = transformations.quaternion_from_matrix(Rtotal)
    ned = np.array(image.camera_pose['ned']) / 10000.0
    s = "%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n" % (q[0], q[1], q[2], q[3],
                                                  ned[1], ned[0], -ned[2])
    f.write(s)
f.close()

# iterate through the matches dictionary to produce a list of matches
f = open( args.project + '/sba-points.txt', 'w' )
for key in matches_dict:
    feat = matches_dict[key]
    ned = np.array(feat['ned']) / 10000.0
    s = "%.8f %.8f %.8f " % (ned[1], ned[0], -ned[2])
    f.write(s)
    pts = feat['pts']
    s = "%d " % (len(pts))
    f.write(s)
    for p in pts:
        image_num = p[0]
        kp = proj.image_list[image_num].kp_list[p[1]]
        s = "%d %.2f %.2f " % (image_num, kp.pt[0], kp.pt[1])
        f.write(s)
    f.write('\n')
f.close()

# print the calibration matrix "K"
f = open( args.project + '/sba-calib.txt', 'w' )
K = proj.cam.get_K()
s = "%.4f %.4f %.4f\n" % (K[0,0], K[0,1], K[0,2])
f.write(s)
s = "%.4f %.4f %.4f\n" % (K[1,0], K[1,1], K[1,2])
f.write(s)
s = "%.4f %.4f %.4f\n" % (K[2,0], K[2,1], K[2,2])
f.write(s)
