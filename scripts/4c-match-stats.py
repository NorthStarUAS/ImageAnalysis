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

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.load_matches()

f = open(args.project + "/Matches.json", 'r')
matches_dict = json.load(f)
f.close()

print "Total unique features =", len(matches_dict)

image_dict = {}
for key in matches_dict:
    feature_dict = matches_dict[key]
    points = feature_dict['pts']
    ned = matches_dict[key]['ned']
    for p in points:
        image_name = proj.image_list[ p[0] ].name
        if image_name in image_dict:
            image_dict[image_name] += 1
        else:
            image_dict[image_name] = 1

print "Total images connected =", len(image_dict)
sum = 0
for key in sorted(image_dict):
    sum += image_dict[key]
    print "%s keypoints = %d" % (key, image_dict[key])
print "Average matches per connected image = %.1f" % (sum / len(image_dict))

Matcher.groupByConnections(proj.image_list)

# save the results
for image in proj.image_list:
    image.save_meta()

Matcher.buildConnectionDetail(proj.image_list, matches_dict)
