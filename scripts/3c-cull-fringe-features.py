#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import json
import math
import numpy as np
import os.path
from progress.bar import Bar
import scipy.spatial

sys.path.append('../lib')
import Matcher
import Pose
import ProjectMgr
import SRTM

# undistort and project keypoints and cull any the blow up in the fringes

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

# setup SRTM ground interpolator
ref = proj.ned_reference_lla
sss = SRTM.NEDGround( ref, 5000, 5000, 30 )

# compute keypoint usage map
proj.compute_kp_usage(all=True)
    
# fast way:
# 1. make a grid (i.e. 8x8) of uv coordinates covering the whole image
# 2. undistort these uv coordinates
# 3. project them into vectors
# 4. intersect them with the srtm terrain to get ned coordinates
# 5. use linearndinterpolator ... g = scipy.interpolate.LinearNDInterpolator([[0,0],[1,0],[0,1],[1,1]], [[0,4,8],[1,3,2],[2,2,-4],[4,1,0]])
#    with origin uv vs. 3d location to build a table
# 6. interpolate original uv coordinates to 3d locations
proj.fastProjectKeypointsTo3d(sss)

# at this point image.coord_list will contain nans for any troublesome
# fringe features, lets dump them
for image in proj.image_list:
    for i in reversed(range(len(image.coord_list))):
        if np.isnan( image.coord_list[i][0]):
            image.kp_list.pop(i)
            np.des_list = np.delete(image.des_list, i, 0)
            image.coord_list.pop(i)
    image.save_features()
    image.save_descriptors()
    # and wipe any existing matches since the index may have all changed
    image.match_list = []
    image.save_matches()
