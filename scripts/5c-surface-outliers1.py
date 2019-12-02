#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cPickle as pickle
import cv2
import fnmatch
import itertools
#import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
import scipy.spatial

sys.path.append('../lib')
import Pose
import ProjectMgr
import SRTM
import transformations


def meta_stats(report):
    sum = 0.0
    count = len(report)
    for line in report:
        value = line[0]
        sum += value
    average = sum / len(report)
    print "average value = %.2f" % (average)

    sum = 0.0
    for line in report:
        value = line[0]
        diff = average - value
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
        
print "Loading original (direct) matches ..."
matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )

print "Loading fitted (sba) matches..."
matches_sba = pickle.load( open( args.project + "/matches_sba", "rb" ) )

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

def compute_surface_outliers():
    # iterate through the sba match dictionary and build a list of feature
    # points and heights (in x=east,y=north,z=up coordinates)
    print "Building Delaunay triangulation..."
    raw_points = []
    raw_values = []
    sum_values = 0.0
    for match in matches_sba:
        ned = match[0]
        raw_points.append( [ned[1], ned[0]] )
        raw_values.append( -ned[2] )
        sum_values += -ned[2]
    avg_height = sum_values / len(matches_sba)
    print "Average elevation = %.1f" % ( avg_height )
    tri = scipy.spatial.Delaunay(np.array(raw_points))

    # look for outliers by comparing depth of a point with the average
    # depth of it's neighbors.  Outliers will tend to stick out this way
    # (although you could be looking at the top of a flag pole so it's not
    # a guarantee of a bad value.)

    print "raw points =", len(raw_points)
    print "tri points =", len(tri.points)

    print "neighbors:", len(tri.vertex_neighbor_vertices[0]), len(tri.vertex_neighbor_vertices[1])
    #print "neighbor[0]:\n", tri.vertex_neighbor_vertices[0][0], tri.vertex_neighbor_vertices[0][1]
    print "Computing average slope to neighbors..."
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
    do_plot = False
    if do_plot:
        x = np.array(x)
        y = np.array(y)
        slope_diff = np.array(slope)
        plt.scatter(x, y, c=slope)
        plt.show()

    average, stddev = meta_stats(report)

    report = sorted(report, key=lambda fields: abs(fields[0]), reverse=True)

    delete_list = []
    for line in report:
        slope = line[0]
        index = line[1]
        if abs(average - slope) >= args.stddev * stddev:
            print "index=", index, "slope=", slope
            delete_list.append( index )

    delete_list = sorted(delete_list, reverse=True)
    for index in delete_list:
        #print "deleting", index
        matches_direct.pop(index)
        matches_sba.pop(index)

    return len(delete_list)

deleted_sum = 0
result = compute_surface_outliers()
while result > 0:
    deleted_sum += result
    result = compute_surface_outliers()

if deleted_sum > 0:
    result=raw_input('Remove ' + str(deleted_sum) + ' outliers from the original matches? (y/n):')
    if result == 'y' or result == 'Y':

        # write out the updated match dictionaries
        print "Writing original matches..."
        pickle.dump(matches_direct, open(args.project+"/matches_direct", "wb"))

        print "Writing sba matches..."
        pickle.dump(matches_sba, open(args.project+"/matches_sba", "wb"))
