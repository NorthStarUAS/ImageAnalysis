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

# setup SRTM ground interpolator
ref = proj.ned_reference_lla

print "loading matches lists..."
for image in proj.image_list:
    image.load_matches()
    #image.save_matches_json()

m = Matcher.Matcher()

for i, i1 in enumerate(proj.image_list):
    for j, i2 in enumerate(proj.image_list):
        if i >= j:
            # don't repeat reciprocal matches
            continue
        if len(i1.match_list[j]):
            print "Showing %s vs %s" % (i1.name, i2.name)
            status = m.showMatch(i1, i2, i1.match_list[j])
