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
import sys

sys.path.append('../lib')
import Groups
import Matcher
import Pose
import ProjectMgr
import SRTM
import transformations

parser = argparse.ArgumentParser(description='Compute Delauney triangulation of matches.')
parser.add_argument('--project', required=True, help='project directory')
args = parser.parse_args()

# project the estimated uv coordinates for the specified image and ned
# point
def compute_feature_uv(K, image, ned):
    if image.PROJ == None:
        rvec, tvec = image.get_proj_sba()
        R, jac = cv2.Rodrigues(rvec)
        image.PROJ = np.concatenate((R, tvec), axis=1)

    PROJ = image.PROJ
    uvh = K.dot( PROJ.dot( np.hstack((ned, 1.0)) ).T )
    uvh /= uvh[2]
    uv = np.array( [ np.squeeze(uvh[0,0]), np.squeeze(uvh[1,0]) ] )
    return uv

def redistort(u, v, dist_coeffs, K):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    #print [x, y]
    k1, k2, p1, p2, k3 = dist_coeffs
    
    # Compute radius^2
    r2 = x**2 + y**2
    r4, r6 = r2**2, r2**3
  
    # Compute tangential distortion
    dx = 2*p1*x*y + p2*(r2 + 2*x*x)
    dy = p1*(r2 + 2*y*y) + 2*p2*x*y
    
    # Compute radial factor
    Lr = 1.0 + k1*r2 + k2*r4 + k3*r6
  
    ud = Lr*x + dx
    vd = Lr*y + dy
    
    return ud * fx + cx, vd * fy + cy

# adds the value to the list and returns the index
def unique_add( mylist, value ):
    key = "%.5f,%.5f,%.5f" % (value[0], value[1], value[2])
    if key in mylist:
        return mylist[key]['index']
    else:
        mylist[key] = {}
        mylist[key]['index'] = mylist['counter']
        mylist[key]['vertex'] = value
        mylist['counter'] += 1
    return mylist['counter'] - 1

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
proj.load_features()
        
print("Loading optimized points ...")
matches_opt = pickle.load( open( os.path.join(args.project, "matches_opt"), "rb" ) )

# load the group connections within the image set
groups = Groups.load(args.project)

if False:
    # collect/group match chains that refer to the same keypoint
    print("Grouping matches ...")
    matches_group = list(matches_opt) # shallow copy
    count = 0
    done = False
    while not done:
        print("Iteration:", count)
        count += 1
        matches_new = []
        matches_lookup = {}
        for i, match in enumerate(matches_group):
            # scan if any of these match points have been previously seen
            # and record the match index
            index = -1
            for p in match[1:]:
                key = "%d-%d" % (p[0], p[1])
                if key in matches_lookup:
                    index = matches_lookup[key]
                    break
            if index < 0:
                # not found, append to the new list
                for p in match[1:]:
                    key = "%d-%d" % (p[0], p[1])
                    matches_lookup[key] = len(matches_new)
                matches_new.append(list(match)) # shallow copy
            else:
                # found a previous reference, append these match items
                existing = matches_new[index]
                # only append items that don't already exist in the early
                # match, and only one match per image (!)
                for p in match[1:]:
                    key = "%d-%d" % (p[0], p[1])
                    found = False
                    for e in existing[1:]:
                        if p[0] == e[0]:
                            found = True
                            break
                    if not found:
                        # add
                        existing.append(list(p)) # shallow copy
                        matches_lookup[key] = index
                # print "new:", existing
                # print 
        if len(matches_new) == len(matches_group):
            done = True
        else:
            matches_group = list(matches_new) # shallow copy
    print("unique features (after grouping):", len(matches_group))

points_group = []
values_group = []
tris_group = []

for i, image in enumerate(proj.image_list):
    if not i in groups[0]:
        print('Skipping image not in primary group:', image.name)
        continue
    
    # iterate through the optimized match dictionary and build a list of feature
    # points and heights (in x=east,y=north,z=up coordinates)
    print("Building raw mesh:", image.name)
    raw_points = []
    raw_values = []
    sum_values = 0.0
    sum_count = 0
    max_z = -9999.0
    min_z = 9999.0
    for match in matches_opt:
        count = 0
        found = False
        for m in match[1:]:
            if m[0] in groups[0]:
                count += 1
            if m[0] == i:
                found = True
        if found and count >= 2:
            ned = match[0]
            raw_points.append( [ned[1], ned[0]] )
            z = -ned[2]
            raw_values.append( z )
            sum_values += z
            sum_count += 1
            if z < min_z:
                min_z = z
                #print(min_z, match)
            if z > max_z:
                max_z = z
                #print(max_z, match)
    if sum_count == 0:
        # no suitable features found for this image ... skip
        continue
    avg_height = sum_values / sum_count
    spread = max_z - min_z
    print("  Average elevation = %.1f Spread = %.1f" % ( avg_height, spread ))
    try:
        tri_list = scipy.spatial.Delaunay(np.array(raw_points))
    except:
        print('problem with delaunay triangulation, skipping')
        continue

    # compute min/max range of horizontal surface
    p0 = raw_points[0]
    x_min = p0[0]
    x_max = p0[0]
    y_min = p0[1]
    y_max = p0[1]
    for p in raw_points:
        if p[0] < x_min: x_min = p[0]
        if p[0] > x_max: x_max = p[0]
        if p[1] < y_min: y_min = p[1]
        if p[1] > y_max: y_max = p[1]
    print("  Area coverage = %.1f,%.1f to %.1f,%.1f (%.1f x %.1f meters)" % \
        (x_min, y_min, x_max, y_max, x_max-x_min, y_max-y_min))

    print("  Points:", len(raw_points))
    print("  Triangles:", len(tri_list.simplices))

    points_group.append(raw_points)
    values_group.append(raw_values)
    tris_group.append(tri_list)
    
name = args.project + "/surface.ac"
gen_ac3d_surface(name, points_group, values_group, tris_group)