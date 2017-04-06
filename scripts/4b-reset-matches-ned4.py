#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import argparse
import commands
import cPickle as pickle
import cv2
import fnmatch
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

# Rest all match point locations to their original direct
# georeferenced locations based on estimated camera pose and
# projection onto DEM earth surface

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()
proj.load_match_pairs()

# setup SRTM ground interpolator
ref = proj.ned_reference_lla
sss = SRTM.NEDGround( ref, 2000, 2000, 30 )

# compute keypoint usage map
proj.compute_kp_usage()
                      
# fast way:
# 1. make a grid (i.e. 8x8) of uv coordinates covering the whole image
# 2. undistort these uv coordinates
# 3. project them into vectors
# 4. intersect them with the srtm terrain to get ned coordinates
# 5. use linearndinterpolator ... g = scipy.interpolate.LinearNDInterpolator([[0,0],[1,0],[0,1],[1,1]], [[0,4,8],[1,3,2],[2,2,-4],[4,1,0]])
#    with origin uv vs. 3d location to build a table
# 6. interpolate original uv coordinates to 3d locations
proj.fastProjectKeypointsTo3d(sss)

# build a list of all keypoints, but only consider pairwise
# matches and don't try to find single matches that span 3 or more
# images.
print "Constructing unified match structure..."
matches_direct = []
for i, i1 in enumerate(proj.image_list):
    # print i1.name
    for j, matches in enumerate(i1.match_list):
        # print proj.image_list[j].name
        if j > i:
            for pair in matches:
                ned1 = proj.image_list[i].coord_list[pair[0]]
                ned2 = proj.image_list[j].coord_list[pair[1]]
                ned = (ned1 + ned2) / 2
                #print ned1, ned2, ned
                match = [ ned, [i, pair[0]], [j, pair[1]] ]
                matches_direct.append( match )

print "total features in image set = %d" % len(matches_direct)
print "2 images per feature, no redundancy removal."

print "Writing match file ..."
pickle.dump(matches_direct, open(args.project + "/matches_direct", "wb"))

print "temp: writing matches_direct ascii version..."
f = open(args.project + "/matches_direct.ascii", "wb")
for match in matches_direct:
    f.write( str(match) + '\n' )
