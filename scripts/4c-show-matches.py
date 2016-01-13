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
parser.add_argument('--order', default='sequential',
                    choices=['sequential', 'fewest-matches'],
                    help='project directory')
parser.add_argument('--image', default="", help='show specific image matches')
parser.add_argument('--index', type=int, help='show specific image by index')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.load_matches()

# setup SRTM ground interpolator
ref = proj.ned_reference_lla

m = Matcher.Matcher()

order = 'fewest-matches'

if args.image:
    i1 = proj.findImageByName(args.image)
    if i1 != None:
        for j, i2 in enumerate(proj.image_list):
            if len(i1.match_list[j]):
                print "Showing %s vs %s" % (i1.name, i2.name)
                status = m.showMatch(i1, i2, i1.match_list[j])
    else:
        print "Cannot locate:", args.image
elif args.index:
    i1 = proj.image_list[args.index]
    if i1 != None:
        for j, i2 in enumerate(proj.image_list):
            if len(i1.match_list[j]):
                print "Showing %s vs %s" % (i1.name, i2.name)
                status = m.showMatch(i1, i2, i1.match_list[j])
    else:
        print "Cannot locate:", args.index
elif args.order == 'sequential':
    for i, i1 in enumerate(proj.image_list):
        for j, i2 in enumerate(proj.image_list):
            if i >= j:
                # don't repeat reciprocal matches
                continue
            if len(i1.match_list[j]):
                print "Showing %s vs %s" % (i1.name, i2.name)
                status = m.showMatch(i1, i2, i1.match_list[j])
elif args.order == 'fewest-matches':
    match_list = []
    for i, i1 in enumerate(proj.image_list):
        for j, i2 in enumerate(proj.image_list):
            if i >= j:
                # don't repeat reciprocal matches
                continue
            if len(i1.match_list[j]):
                match_list.append( ( len(i1.match_list[j]), i, j ) )
    match_list = sorted(match_list,
                        key=lambda fields: fields[0],
                        reverse=False)
    for match in match_list:
        count = match[0]
        i = match[1]
        j = match[2]
        i1 = proj.image_list[i]
        i2 = proj.image_list[j]
        print "Showing %s vs %s (matches=%d)" % (i1.name, i2.name, count)
        status = m.showMatch(i1, i2, i1.match_list[j])
              
