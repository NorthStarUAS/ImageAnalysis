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

# build a list of all 'unique' keypoints and which containting image
# (image, index)
match_dict = {}
for i, i1 in enumerate(proj.image_list):
    for j, matches in enumerate(i1.match_list):
        if j > i:
            for pair in matches:
                key = "%d-%d" % (i, pair[0])
                m1 = [i, pair[0]]
                m2 = [j, pair[1]]
                if key in match_dict:
                    match_dict[key].append(m2)
                else:
                    match_dict[key] = [m1, m2]
#print match_dict
count = 0.0
sum = 0.0
for keys in match_dict:
    sum += len(match_dict[keys])
    count += 1
if count > 0.1:
    print "total unique features in image set = %d" % count
    print "kp average instances = %.4f" % (sum / count)

f = open(args.project + "/Matches.json", 'w')
json.dump(match_dict, f, sort_keys=True)
f.close()
