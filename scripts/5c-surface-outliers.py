#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import itertools
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
from progress.bar import Bar
import scipy.spatial

sys.path.append('../lib')
import Matcher
import Pose
import ProjectMgr
import SRTM
import transformations


def meta_stats(report):
    sum = 0.0
    count = len(report)
    for line in report:
        diff = line[0]
        sum += diff
    average = sum / len(report)
    print "mean error = %.2f" % (average)

    sum = 0.0
    for line in report:
        diff = line[0]
        off = average - diff
        sum += diff**2
    stddev = math.sqrt(sum / count)
    print "standard deviation = %.2f" % (stddev)
    return average, stddev

parser = argparse.ArgumentParser(description='Compute Delauney triangulation of matches.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--stddev', default=5, type=int, help='standard dev threshold')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
        
print "Loading original matches ..."
f = open(args.project + "/Matches.json", 'r')
matches_direct = json.load(f)
f.close()

print "Loading match points..."
f = open(args.project + "/Matches-sba.json", 'r')
matches_sba = json.load(f)
f.close()

# iterate through the sba match dictionary and build a list of feature
# points and heights (in x=east,y=north,z=up coordinates)
print "Building raw mesh interpolator"
raw_points = []
raw_values = []
sum_values = 0.0
reverse_lookup = [None] * len(matches_sba)
for i, key in enumerate(matches_sba):
    reverse_lookup[i] = key
    feature_dict = matches_sba[key]
    ned = feature_dict['ned']
    raw_points.append( [ned[1], ned[0]] )
    raw_values.append( -ned[2] )
    sum_values += -ned[2]
avg_height = sum_values / len(matches_sba)
print "Average elevation = %.1f" % ( avg_height )
tri = scipy.spatial.Delaunay(np.array(raw_points))

# custom slope routine
def my_slope(p1, p2, z1, z2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = z2 - z1
    hdist = math.sqrt(dx**2 + dy**2)
    if hdist > 0.00001:
        slope = dz / hdist
    else:
        slope = 0
    return slope

# look for outliers by comparing depth of a point with the average
# depth of it's neighbors.  Outliers will tend to stick out this way
# (although you could be looking at the top of a flag pole so it's not
# a guarantee of a bad value.)

print "raw points =", len(raw_points)
print "tri points =", len(tri.points)

print "neighbors:", len(tri.vertex_neighbor_vertices[0]), len(tri.vertex_neighbor_vertices[1])
#print "neighbor[0]:\n", tri.vertex_neighbor_vertices[0][0], tri.vertex_neighbor_vertices[0][1]
indices, indptr = tri.vertex_neighbor_vertices
report = []
x = []; y = []; slope = []
for i in range(len(tri.points)):
    pi = raw_points[i]
    zi = raw_values[i]
    sum_slope = 0.0
    neighbors = indptr[indices[i]:indices[i+1]]
    if len(neighbors) == 0:
        continue
    # print neighbors
    for j in neighbors:
        pj = raw_points[j]
        zj = raw_values[j]
        sum_slope += my_slope(pi, pj, zi, zj)
    avg_slope = sum_slope / len(neighbors)
    # print i, avg_slope
    report.append( (avg_slope, i) )
    x.append(raw_points[i][0])
    y.append(raw_points[i][1])
    slope.append(avg_slope)

# plot results
x = np.array(x)
y = np.array(y)
slope_diff = np.array(slope)
plt.scatter(x, y, c=slope)
plt.show()

avg, stddev = meta_stats(report)

report = sorted(report, key=lambda fields: fields[0], reverse=True)

delete_list = []
for line in report:
    slope = line[0]
    index = line[1]
    if slope >= args.stddev * stddev:
        print "index=", index, "slope=", slope
        delete_list.append( reverse_lookup[index] )

result = raw_input('Remove these outliers from the original matches? (y/n):')
if result == 'y' or result == 'Y':
    for key in delete_list:
        print "deleting", key
        del matches_direct[key]
        del matches_sba[key]
        
    # write out the updated match dictionaries
    print "Writing original matches..."
    f = open(args.project + "/Matches.json", 'w')
    json.dump(matches_direct, f, sort_keys=True)
    f.close()
    print "Writing sba matches..."
    f = open(args.project + "/Matches-sba.json", 'w')
    json.dump(matches_sba, f, sort_keys=True)
    f.close()
