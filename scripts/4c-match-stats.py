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

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.load_match_pairs()

print "Loading original (direct) matches ..."
matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )
print "Total unique features =", len(matches_direct)

print "Generating match pairs"
match_pairs = proj.generate_match_pairs(matches_direct)

group_list = Matcher.groupByConnections(proj.image_list, matches_direct, match_pairs)

# group_list[0] should be the primary collection of connected images.
# The rest need to get removed and ignored from the solution
remove_dict = {}
for group in group_list[1:]:
    for image in group:
        i = proj.findIndexByName(image.name)
        remove_dict[i] = True
print "Removing all features and matches from the following images."
print "These are not part of the primary group."
print remove_dict

# mark any features in the weak images list
mark_sum = 0
for i, match in enumerate(matches_direct):
    #print 'before:', match
    for j, p in enumerate(match[1:]):
        if p[0] in remove_dict:
             match[j+1] = [-1, -1]
             mark_sum += 1
    #print 'after:', match

if mark_sum > 0:
    result=raw_input('Remove ' + str(mark_sum) + ' non-group features from the original matches? (y/n):')
    if result == 'y' or result == 'Y':
        print " deleting marked items..."
        for i in reversed(range(len(matches_direct))):
            match_direct = matches_direct[i]
            has_bad_elem = False
            for j in reversed(range(1, len(match_direct))):
                p = match_direct[j]
                if p == [-1, -1]:
                    has_bad_elem = True
                    match_direct.pop(j)
            if len(match_direct) < 3:
                print "deleting match that is now in less than 2 images:", match_direct
                matches_direct.pop(i)
        # write out the updated match dictionaries
        print "Writing direct matches..."
        pickle.dump(matches_direct, open(args.project+"/matches_direct", "wb"))

