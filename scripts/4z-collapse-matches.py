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

# Reset all match point locations to their original direct
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

print "Loading original (direct) matches ..."
matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )

print "Loading fitted (sba) matches..."
matches_sba = pickle.load( open( args.project + "/matches_sba", "rb" ) )

# collect/group match chains that refer to the same keypoint (warning,
# if there are bad matches this can over-constrain the problem or tie
# the pieces together too tightly/incorrectly and lead to nans.)
count = 0
done = False
while not done:
    print "Iteration:", count
    count += 1
    matches_dir_new = []
    matches_sba_new = []
    matches_lookup = {}
    for i, (match_dir, match_sba) in enumerate(zip(matches_direct, matches_sba)):
        # scan if any of these match points have been previously seen
        # and record the match index
        index = -1
        for p in match_dir[1:]:
            key = "%d-%d" % (p[0], p[1])
            if key in matches_lookup:
                index = matches_lookup[key]
                break
        if index < 0:
            # not found, append to the new list
            for p in match_dir[1:]:
                key = "%d-%d" % (p[0], p[1])
                matches_lookup[key] = len(matches_dir_new)
            matches_dir_new.append(match_dir)
            matches_sba_new.append(match_sba)
        else:
            # found a previous reference, append these match items
            existing_dir = matches_dir_new[index]
            existing_sba = matches_sba_new[index]
            # print existing, "+", match
            # only append items that don't already exist in the early
            # match
            for p in match_dir[1:]:
                key = "%d-%d" % (p[0], p[1])
                found = False
                for e in existing_dir[1:]:
                    if p[0] == e[0] and p[1] == e[1]:
                        found = True
                        break
                if not found:
                    # add
                    existing_dir.append(p)
                    existing_sba.append(p)
                    matches_lookup[key] = index
            # print "new:", existing
    if len(matches_dir_new) == len(matches_direct):
        done = True
    else:
        matches_direct = matches_dir_new
        matches_sba = matches_sba_new
 
#print match_dict
count = 0.0
sum = 0.0
for match in matches_direct:
    n = len(match)
    if n >= 3:
        # len should be 3, 4, etc..
        sum += (n-1)
        count += 1
    else:
        print "Oops, match with < 2 image references!"
        
if count > 0.1:
    print "total unique features in image set = %d" % count
    print "keypoint average instances = %.4f" % (sum / count)


print "Writing direct match file ..."
pickle.dump(matches_direct, open(args.project + "/matches_direct", "wb"))

print "Writing sba match file ..."
pickle.dump(matches_sba, open(args.project + "/matches_sba", "wb"))

for match in matches_direct:
    print match
