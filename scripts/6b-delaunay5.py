#!/usr/bin/python3

import argparse
import pickle
import cv2
import fnmatch
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
from progress.bar import Bar
import scipy.spatial

from lib import Groups
from lib import Matcher
from lib import Pose
from lib import ProjectMgr
from lib import SRTM
from lib import transformations

parser = argparse.ArgumentParser(description='Compute Delauney triangulation of matches.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--group', type=int, default=0, help='group index')
args = parser.parse_args()

def gen_ac3d_surface(name, points_group, values_group, tris_group):
    kids = len(tris_group)
    # write out the ac3d file
    f = open( name, "w" )
    f.write("AC3Db\n")
    trans = 0.0
    f.write("MATERIAL \"\" rgb 1 1 1  amb 0.6 0.6 0.6  emis 0 0 0  spec 0.5 0.5 0.5  shi 10  trans %.2f\n" % (trans))
    f.write("OBJECT world\n")
    f.write("kids " + str(kids) + "\n")

    for i in range(kids):
        points = points_group[i]
        values = values_group[i]
        tris = tris_group[i]
        f.write("OBJECT poly\n")
        f.write("loc 0 0 0\n")
        f.write("numvert %d\n" % len(points))
        for j in range(len(points)):
            f.write("%.3f %.3f %.3f\n" % (points[j][0], points[j][1],
                                          values[j]))
        f.write("numsurf %d\n" % len(tris.simplices))
        for tri in tris.simplices:
            f.write("SURF 0x30\n")
            f.write("mat 0\n")
            f.write("refs 3\n")
            for t in tri:
                f.write("%d 0 0\n" % (t))
        f.write("kids 0\n")
                
proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()

print("Loading optimized points ...")
matches = pickle.load( open( os.path.join(proj.analysis_dir, "matches_grouped"), "rb" ) )

# load the group connections within the image set
groups = Groups.load(proj.analysis_dir)

points_group = []
values_group = []
tris_group = []

# initialize temporary structures for vanity stats
for image in proj.image_list:
    image.raw_points = []
    image.raw_values = []
    image.sum_values = 0.0
    image.sum_count = 0.0
    image.max_z = -9999.0
    image.min_z = 9999.0

# elevation stats
print("Computing stats...")
ned_list = []
for match in matches:
    if match[1] == args.group:  # used by current group
        ned_list.append(match[0])
avg = -np.mean(np.array(ned_list)[:,2])
std = np.std(np.array(ned_list)[:,2])
print("Average elevation: %.2f" % avg)
print("Standard deviation: %.2f" % std)

# sort through points
print('Reading feature locations from optimized match points ...')
global_raw_points = []
global_raw_values = []
for match in matches:
    if match[1] == args.group:  # used by current group
        ned = match[0]
        diff = abs(-ned[2] - avg)
        if diff < 5*std:
            global_raw_points.append( [ned[1], ned[0]] )
            global_raw_values.append( -ned[2] )
        else:
            print("Discarding match with excessive altitude:", match)

print('Generating Delaunay meshes ...')
global_tri_list = scipy.spatial.Delaunay(np.array(global_raw_points))

print('Generating ac3d surface model ...')
name = os.path.join(proj.analysis_dir, "surface-global.ac")
gen_ac3d_surface(name, [global_raw_points], [global_raw_values], [global_tri_list])
