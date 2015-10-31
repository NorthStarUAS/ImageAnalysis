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

def bestNeighbor(image, image_list):
    best_cycle_dist = len(image_list) + 1
    best_index = None
    for i, pairs in enumerate(image.match_list):
        if len(pairs):
            i2 = image_list[i]
            dist = i2.cycle_dist
            #print "  neighbor check %d = %d" % ( i, dist )
            if dist >= 0 and dist < best_cycle_dist:
                best_cycle_dist = dist
                best_index = i
    return best_index, best_cycle_dist

def groupByConnections(image_list):
    # reset the cycle distance for all images
    for image in image_list:
        image.cycle_dist = -1
        
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
        best_index = None
        best_cycle_dist = len(image_list) + 1
        # find an unplaced image with a placed neighbor that has
        # the most connections to other images
        for i, image in enumerate(image_list):
            if image.cycle_dist < 0:
                index, cycle_dist = bestNeighbor(image, image_list)
                if cycle_dist >= 0 and (cycle_dist+1 < best_cycle_dist):
                    best_index = i
                    best_cycle_dist = cycle_dist+1
                    done = False
        if best_index == None:
            #print "Cannot find an unplaced image with a connected neighbor"
            if len(group):
                # commit the previous group (if it exists)
                group_list.append(group)
                # and start a new group
                group = []
                cycle_dist = 0
            # now find an unplaced image that has the most connections
            # to other images (new cycle start)
            max_connections = None
            best_cycle_dist = 0
            for i, image in enumerate(image_list):
                if image.cycle_dist < 0:
                    if (max_connections == None or image.connections > max_connections):
                        max_connections = image.connections
                        best_index = i
                        done = False
                        #print " found image %d connections = %d" % (i, max_connections)
        if best_index != None:
            image = image_list[best_index]
            image.cycle_dist = best_cycle_dist
            #print "Adding %s (cycles = %d)" % (image.name, best_cycle_dist)
            group.append(image)

    print "Group (cycles) report:"
    for group in group_list:
        if len(group) < 2:
            continue
        print "group (size=%d):" % (len(group)),
        for image in group:
            print "%s(%d)" % (image.name, image.cycle_dist),
        print ""

    return group_list

groupByConnections(proj.image_list)
