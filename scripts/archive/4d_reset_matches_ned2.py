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

# superseded by reset-matches-ned3.py

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--no-grouping', action='store_true', help='don\'t group match cyles')

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

print "Indexing features by unique uv coordinates..."
# we typically expect many duplicated feature uv coordinates, but they
# may have different scaling or other attributes important during
# feature matching
for image in proj.image_list:
    print image.name
    # pass one, build a tmp structure of unique keypoints (by uv) and
    # the index of the first instance.
    image.kp_remap = {}
    for i, kp in enumerate(image.kp_list):
        if image.kp_used[i]:
            key = "%.2f-%.2f" % (kp.pt[0], kp.pt[1])
            if not key in image.kp_remap:
                image.kp_remap[key] = i
            # else:
            #     print "%d -> %d" % (i, image.kp_remap[key])
            #     print " ", image.coord_list[i], image.coord_list[image.kp_remap[key]]
    print " features:", len(image.kp_list)
    print " unique by uv and used:", len(image.kp_remap)

print "Collapsing keypoints with duplicate uv coordinates..."
# after feature matching we don't care about other attributes, just
# the uv coordinate.
for i1 in proj.image_list:
    for j, matches in enumerate(i1.match_list):
        i2 = proj.image_list[j]
        for k, pair in enumerate(matches):
            # print pair
            idx1 = pair[0]
            idx2 = pair[1]
            kp1 = i1.kp_list[idx1]
            kp2 = i2.kp_list[idx2]
            key1 = "%.2f-%.2f" % (kp1.pt[0], kp1.pt[1])
            key2 = "%.2f-%.2f" % (kp2.pt[0], kp2.pt[1])
            if j == 1 and idx2 == 1360:
                print key1, key2
            if j == 1 and idx2 == 3212:
                print key1, key2
            # print key1, key2
            new_idx1 = i1.kp_remap[key1]
            new_idx2 = i2.kp_remap[key2]
            if False and idx1 != new_idx1:
                print "1: %d -> %d" % (idx1, new_idx1)
                new_uv1 = list(i1.kp_list[new_idx1].pt)
                print "  [%.2f, %.2f] -> [%.2f, %.2f]" % (uv1[0], uv1[1],
                                                          new_uv1[0],
                                                          new_uv1[1])
                if not np.allclose(uv1, new_uv1):
                    print "OOPS!"
            if False and idx2 != new_idx2:
                print "2: %d -> %d" % (idx2, new_idx2)
                new_uv2 = list(i2.kp_list[new_idx2].pt)
                print "  [%.2f, %.2f] -> [%.2f, %.2f]" % (uv2[0], uv2[1],
                                                          new_uv2[0],
                                                          new_uv2[1])
                if not np.allclose(uv2, new_uv2):
                    print "OOPS!"
            matches[k] = [new_idx1, new_idx2]                
                
print "Eliminating duplicates..."
# after collapsing by uv coordinate, we could be left with duplicate
# matches (mached at different scales, but same exact point.)
for i1 in proj.image_list:
    for j, matches in enumerate(i1.match_list):
        i2 = proj.image_list[j]
        kp_dict = {}
        pair_dict = {}
        new_matches = []
        for k, pair in enumerate(matches):
            key = "%d-%d" % (pair[0], pair[1])
            if not pair[0] in kp_dict:
                kp_dict[pair[0]] = True
                if not key in pair_dict:
                    pair_dict[key] = True
                    new_matches.append(pair)
            else:
                print "warning skipping keypoint idx", pair[0], "already used in another match."
                
        i1.match_list[j] = new_matches

print "Constructing unified match structure..."
# create an initial pair-wise match list
matches_direct = []
for i, i1 in enumerate(proj.image_list):
    # print i1.name
    for j, matches in enumerate(i1.match_list):
        # print proj.image_list[j].name
        if j > i:
            for pair in matches:
                match = []
                # ned place holder
                match.append([0.0, 0.0, 0.0])
                match.append([i, pair[0]])
                match.append([j, pair[1]])
                matches_direct.append(match)

# collect/group match chains that refer to the same keypoint (warning,
# if there are bad matches this can over-constrain the problem or tie
# the pieces together too tightly/incorrectly and lead to nans.)
count = 0
if args.no_grouping:
    done = True
else:
    done = False
while not done:
    print "Iteration:", count
    count += 1
    matches_new = []
    matches_lookup = {}
    for i, match in enumerate(matches_direct):
        # scan if any of these match points have been previously seen
        # and record the match index
        index = -1
        for p in match[1:]:
            key = "%d-%d" % (p[0], p[1])
            if key in matches_lookup:
                index = matches_lookup[key]
                break
        if index < 0:
            # not found, append to the new list
            for p in match[1:]:
                key = "%d-%d" % (p[0], p[1])
                matches_lookup[key] = len(matches_new)
            matches_new.append(match)
        else:
            # found a previous reference, append these match items
            existing = matches_new[index]
            print existing, "+", match
            # only append items that don't already exist in the early
            # match, and only one match per image (!)
            for p in match[1:]:
                key = "%d-%d" % (p[0], p[1])
                found = False
                for e in existing[1:]:
                    if p[0] == e[0]:
                        found = True
                        break
                if not found:
                    # add
                    existing.append(p)
                    matches_lookup[key] = index
            print "new:", existing
            print 
    if len(matches_new) == len(matches_direct):
        done = True
    else:
        matches_direct = matches_new

# matches_direct format is a 3d_coord, img-feat, img-feat, ...
# len of 3 means features shows up on 2 images.  We would like
# to only use features that show up in 3 or more images.
print "discarding matches that appear in less than 3 images"
matches_new = []
for m in matches_direct:
    if len(m) >= 4:
        matches_new.append(m)
matches_direct = matches_new

for m in matches_direct:
    print m
    
count = 0.0
sum = 0.0
for match in matches_direct:
    n = len(match)
    sum += (n-1)
    count += 1
        
if count >= 1:
    print "total unique features in image set = %d" % count
    print "keypoint average instances = %.4f" % (sum / count)

# compute an initial guess at the 3d location of each unique feature
# by averaging the locations of each projection
print "Estimating world coordinates of each keypoint..."
for match in matches_direct:
    sum = np.array( [0.0, 0.0, 0.0] )
    for p in match[1:]:
        # print proj.image_list[ p[0] ].coord_list[ p[1] ]
        sum += proj.image_list[ p[0] ].coord_list[ p[1] ]
    ned = sum / len(match[1:])
    # print "avg =", ned
    match[0] = ned.tolist()

print "Writing match file ..."
pickle.dump(matches_direct, open(args.project + "/matches_direct", "wb"))

#print "temp: writing ascii version..."
#for match in matches_direct:
#    print match
