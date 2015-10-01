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

def hasPlacedNeighbor(image, image_list):
    for i, pairs in enumerate(image.match_list):
        if len(pairs):
            i2 = image_list[i]
            if i2.placed:
                return True
    return False

def groupByConnections(image_list):
    # reset the placed flag
    for image in image_list:
        image.placed = False
        
    # compute number of connections per image
    for image in image_list:
        image.connections = 0
        for pairs in image.match_list:
            if len(pairs) >= 8:
                image.connections += 1
        if image.connections > 1:
            print "%s connections: %d" % (image.name, image.connections)
        
    group_list = []
    group = []
    done = False
    while not done:
        done = True
        maxcon = None
        maxidx = None
        # find an unplaced image with a placed neighbor that has
        # the most connections to other images
        for i, image in enumerate(image_list):
            if not image.placed and hasPlacedNeighbor(image, image_list) and (maxcon == None or image.connections > maxcon):
                maxcon = image.connections
                maxidx = i
                done = False
        if maxidx == None:
            if len(group):
                # commit the previous group (if it exists)
                group_list.append(group)
                # and start a new group
                group = []
            # now find an unplaced image that has the most connections
            # to other images
            for i, image in enumerate(image_list):
                if not image.placed and (maxcon == None or image.connections > maxcon):
                    maxcon = image.connections
                    maxidx = i
                    done = False
        if maxidx != None:
            image = image_list[maxidx]
            #print "Adding %s (connections = %d)" % (image.name, maxcon)
            image.placed = True
            group.append(image)

    print "Group (cycles) report:"
    for group in group_list:
        if len(group) < 2:
            continue
        print "group (size=%d):" % (len(group)),
        for image in group:
            print image.name,
        print ""

    return group_list

groupByConnections(proj.image_list)
