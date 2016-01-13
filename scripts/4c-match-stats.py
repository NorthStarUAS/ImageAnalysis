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

# working on matching features ...

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.load_matches()

print "Loading original (direct) matches ..."
matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )

print "Total unique features =", len(matches_direct)

Matcher.groupByConnections(proj.image_list, matches_direct)

# save the results
for image in proj.image_list:
    image.save_meta()
