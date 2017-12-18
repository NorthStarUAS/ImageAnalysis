#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

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

print "Constructing unified match structure..."
easy_dumb = False
if not easy_dumb:
    # build a list of all 'unique' keypoints.  Include an index to
    # each containing image and feature.  This seems smarter, but
    # complicates outlier detection and removal and subsequent cycle
    # and connection computations.
    matches_dict = {}
    for i, i1 in enumerate(proj.image_list):
        # print i1.name
        for j, matches in enumerate(i1.match_list):
            # print proj.image_list[j].name
            if j > i:
                for pair in matches:
                    key = "%d-%d" % (i, pair[0])
                    #if key == '1-8450':
                    #    print key
                    #    print "  ", i, "vs", j
                    m1 = [i, pair[0]]
                    m2 = [j, pair[1]]
                    #print "  ", m1, "; ", m2
                    if key in matches_dict:
                        feature_dict = matches_dict[key]
                        feature_dict['pts'].append(m2)
                    else:
                        feature_dict = {}
                        feature_dict['pts'] = [m1, m2]
                        matches_dict[key] = feature_dict
else:
    # build a list of all keypoints, but only consider pairwise
    # matches and don't try to find single matches that span 3 or more
    # images.
    matches_dict = {}
    for i, i1 in enumerate(proj.image_list):
        # print i1.name
        for j, matches in enumerate(i1.match_list):
            # print proj.image_list[j].name
            if j > i:
                for pair in matches:
                    key = "%d-%d-%d" % (i, j, pair[0])
                    m1 = [i, pair[0]]
                    m2 = [j, pair[1]]
                    #print "  ", m1, "; ", m2
                    feature_dict = {}
                    feature_dict['pts'] = [m1, m2]
                    matches_dict[key] = feature_dict

# matches_dict is used with the key so we can find existing keys
# quickly for triple + match points.  But now lets convert the result
# to a list
matches_direct = []
for key in matches_dict:
    match = []
    # ned place holder
    match.append([0.0, 0.0, 0.0])
    feature_dict = matches_dict[key]
    for p in feature_dict['pts']:
        match.append( [ p[0], p[1] ] )
    matches_direct.append( match )
    
#print match_dict
count = 0.0
sum = 0.0
for match in matches_direct:
    n = len(match)
    if n >= 2:
        # len should be 2, 3, 4, etc..
        sum += (n-1)
        count += 1
if count > 0.1:
    print "total unique features in image set = %d" % count
    print "keypoint average instances = %.4f" % (sum / count)

# compute an initial guess at the 3d location of each unique feature
# by averaging the locations of each projection
print "Estimating world coordinates of each keypoint..."
for match in matches_direct:
    sum = np.array( [0.0, 0.0, 0.0] )
    for p in match[1:]:
        sum += proj.image_list[ p[0] ].coord_list[ p[1] ]
    ned = sum / len(match[1:])
    match[0] = ned.tolist()

print "Writing match file ..."
pickle.dump(matches_direct, open(args.project + "/matches_direct", "wb"))

print "temp: writing ascii version..."
for match in matches_direct:
    print match
