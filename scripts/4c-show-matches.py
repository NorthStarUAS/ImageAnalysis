#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

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
parser.add_argument('--order', default='sequential',
                    choices=['sequential', 'fewest-matches'],
                    help='sort order')
parser.add_argument('--orient', default='aircraft',
                    choices=['aircraft', 'camera', 'sba'],
                    help='yaw orientation reference')
parser.add_argument('--image', default="", help='show specific image matches')
parser.add_argument('--index', type=int, help='show specific image by index')
parser.add_argument('--direct', action='store_true', help='show matches_direct')
parser.add_argument('--sba', action='store_true', help='show matches_sba')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.load_match_pairs()

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
                status = m.showMatchOrient(i1, i2, i1.match_list[j],
                                           orient=args.orient)
    else:
        print "Cannot locate:", args.image
elif args.index:
    i1 = proj.image_list[args.index]
    if i1 != None:
        for j, i2 in enumerate(proj.image_list):
            if len(i1.match_list[j]):
                print "Showing %s vs %s" % (i1.name, i2.name)
                status = m.showMatchOrient(i1, i2, i1.match_list[j],
                                           orient=args.orient)
    else:
        print "Cannot locate:", args.index
elif args.direct or args.sba:
    if args.direct:
        matches_list = pickle.load( open( args.project + "/matches_direct", "rb" ) )
    else:
        matches_list = pickle.load( open( args.project + "/matches_sba", "rb" ) )
    for i, i1 in enumerate(proj.image_list):
        for j, i2 in enumerate(proj.image_list):
            if j <= i:
                continue
            # a little inefficient, but just used for debugging ...
            matches = []
            for match in matches_list:
                p1 = None
                p2 = None
                for p in match[1:]:
                    if p[0] == i:
                        p1 = p[1]
                    if p[0] == j:
                        p2 = p[1]
                if p1 != None and p2 != None:
                    matches.append( [p1, p2] )
            if len(matches):
                print "Showing (direct) %s vs %s" % (i1.name, i2.name)
                status = m.showMatchOrient(i1, i2, matches,
                                           orient=args.orient)
elif args.order == 'sequential':
    for i, i1 in enumerate(proj.image_list):
        for j, i2 in enumerate(proj.image_list):
            if i >= j:
                # don't repeat reciprocal matches
                continue
            if len(i1.match_list[j]):
                print "Showing %s vs %s" % (i1.name, i2.name)
                status = m.showMatchOrient(i1, i2, i1.match_list[j],
                                           orient=args.orient)
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
        status = m.showMatchOrient(i1, i2, i1.match_list[j],
                                   orient=args.orient)
