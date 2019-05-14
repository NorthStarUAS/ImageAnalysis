#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import json
import math
import numpy as np
import os.path
import scipy.spatial

sys.path.append('../lib')
import Matcher
import Pose
import ProjectMgr
import SRTM
import transformations

# plot sba results

parser = argparse.ArgumentParser(description='Plot SBA solution.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

f = open(args.project + "/Matches.json", 'r')
matches_direct = json.load(f)
f.close()
f = open(args.project + "/Matches-sba.json", 'r')
matches_sba = json.load(f)
f.close()

min = None
max = None

# iterate through the direct match dictionary and build a per image
# list of obj_pts and img_pts
f = open('plot-direct.txt', 'w')
for key in matches_direct:
    feature_dict = matches_direct[key]
    ned = feature_dict['ned']
    # track min/max for plot ranges
    if min == None:
        min = list(ned) # copy
    if max == None:
        max = list(ned) # copy
    for i in range(3):
        #print "min[%d] = %.2f" % (i, min[i])
        #print "ned[%d] = %s" % (i, ned[i])
        if ned[i] < min[i]:
            #print "Updating min[%d] = %.2f" % (i, ned[i])
            min[i] = ned[i]
        if ned[i] > max[i]:
            max[i] = ned[i]
    f.write( "%.2f %.2f %.2f\n" % (ned[1], ned[0], -ned[2]) )
f.close()

# iterate through the sba match dictionary and build a per image
# list of obj_pts and img_pts
f = open('plot-sba.txt', 'w')
for key in matches_sba:
    feature_dict = matches_sba[key]
    ned = feature_dict['ned']
    # track min/max for plot ranges
    if min == None:
        min = list(ned) # copy
    if max == None:
        max = list(ned) # copy
    for i in range(3):
        #print "min[%d] = %.2f" % (i, min[i])
        #print "ned[%d] = %s" % (i, ned[i])
        if ned[i] < min[i]:
            #print "Updating min[%d] = %.2f" % (i, ned[i])
            min[i] = ned[i]
        if ned[i] > max[i]:
            max[i] = ned[i]
    f.write( "%.2f %.2f %.2f\n" % (ned[1], ned[0], -ned[2]) )
f.close()

print "min = %s max = %s" % (min, max)
diff = [ 0.0, 0.0, 0.0 ]
center = [ 0.0, 0.0, 0.0 ]
for i in range(3):
    diff[i] = max[i] - min[i]
    center[i] = (max[i] + min[i]) / 2.0
print "diff = %s" % (diff)
print "center = %s" % (center)

maxdiff = 0.0
for i in range(3):
    if diff[i] > maxdiff:
        maxdiff = diff[i]
print "max diff = %.2f" % (maxdiff)
half = maxdiff / 2.0

print "splot",
print "[%.2f:%.2f]" % (center[1] - half, center[1] + half),
print "[%.2f:%.2f]" % (center[0] - half, center[0] + half),
print "[%.2f:%.2f]" % (-center[2] - half, -center[2] + half),
print "\"plot-direct.txt\" with dots",

print ",",
print "\"plot-sba.txt\" with dots"

#g = Gnuplot.Gnuplot(debug=1)
#g.title('A simple example') # (optional)
#g('set data style linespoints') # give gnuplot an arbitrary command
# Plot a list of (x, y) pairs (tuples or a numpy array would
# also be OK):
#g.splot(gdata)
#raw_input('Please press return to continue...\n')
