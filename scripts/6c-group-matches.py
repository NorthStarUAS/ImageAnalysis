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

# working on matching features ...

parser = argparse.ArgumentParser(description='Group matches.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.load_matches()

m = Matcher.Matcher()

# remove non-reciprocal matches (this is another filtering technique
# to remove potential false positives)
bar = Bar('Removing non-reciprocal matches:', max=len(proj.image_list))
for i, i1 in enumerate(proj.image_list):
    for j, i2 in enumerate(proj.image_list):
        #print "testing %i vs %i" % (i, j)
        matches = i1.match_list[j]
        rmatches = i2.match_list[i]
        before = len(matches)
        for k, pair in enumerate(matches):
            rpair = [pair[1], pair[0]]
            found = False
            for r in rmatches:
                if rpair == r:
                    found = True
                    break
            if not found:
                #print "not found =", rpair
                matches[k] = [-1, -1]
        for pair in reversed(matches):
            if pair == [-1, -1]:
                matches.remove(pair)
        after = len(matches)
        #if before != after:
        #    print "  (%d vs. %d) matches %d -> %d" % (i, j, before, after)
    bar.next()
#    i1.save_matches()
bar.finish()

# build a list of all 'unique' keypoints and which containting image
# (image, index)
match_dict = {}
for i, i1 in enumerate(proj.image_list):
    for j, matches in enumerate(i1.match_list):
        if j > i:
            for pair in matches:
                key1 = "%d-%d" % (i, pair[0])
                key2 = "%d-%d" % (j, pair[1])
                if key1 in match_dict:
                    match_dict[key1].append(key2)
                else:
                    match_dict[key1] = [key1, key2]
#print match_dict
count = 0.0
sum = 0.0
for keys in match_dict:
    sum += len(match_dict[keys])
    count += 1
if count > 0.1:
    print "total unique features in image set = %d" % count
    print "kp average instances = %.4f" % (sum / count)

f = open("matches.json", 'w')
json.dump(match_dict, f, indent=2, sort_keys=True)
f.close()
