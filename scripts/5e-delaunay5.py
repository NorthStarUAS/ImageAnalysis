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
from progress.bar import Bar
import scipy.spatial

import Polygon
import Polygon.Shapes
import Polygon.Utils

sys.path.append('../lib')
import Matcher
import Pose
import ProjectMgr
import SRTM
import transformations

parser = argparse.ArgumentParser(description='Compute Delauney triangulation of matches.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--depth', action='store_const', const=True,
                    help='generate 3d surface')
args = parser.parse_args()


# project the estimated uv coordinates for the specified image and ned
# point
def compute_feature_uv(K, image, ned):
    uvh = K.dot( image.PROJ.dot( np.hstack((ned, 1.0)) ).T )
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

def gen_ac3d_object(f, name, raw_tris):
    vertices = {}
    vertices['counter'] = 0
    tris = []
    for tri in raw_tris:
        for v in tri:
            ned = [ v[0], v[1], v[2] ]
            index = unique_add( vertices, ned )
            tris.append( [ index, v[3], v[4] ] )
    print "raw vertices =", len(raw_tris)*3
    print "indexed vertices =", len(vertices)
    # sort the dictionary into an array so we can output it in the
    # correct order
    vertex_list = [None] * (len(vertices) - 1) # skip counter record
    for key in vertices:
        if key != 'counter':
            index = vertices[key]['index']
            v = vertices[key]['vertex']
            vertex_list[index] = v
    f.write("OBJECT poly\n")
    f.write("texture \"./Textures/" + name + "\"\n")
    f.write("loc 0 0 0\n")
    f.write("numvert %d\n" % len(vertex_list))
    for i, v in enumerate(vertex_list):
        f.write("%.3f %.3f %.3f\n" % (v[1], v[0], -v[2]))
    f.write("numsurf %d\n" % (len(tris) / 3))
    for i in range(len(tris) / 3):
        f.write("SURF 0x30\n")
        f.write("mat 0\n")
        f.write("refs 3\n")
        t = tris[3*i]
        f.write("%d %.4f %.4f\n" % (t[0], t[1], t[2]))
        t = tris[3*i + 1]
        f.write("%d %.4f %.4f\n" % (t[0], t[1], t[2]))
        t = tris[3*i + 2]
        f.write("%d %.4f %.4f\n" % (t[0], t[1], t[2]))
    f.write("kids 0\n")
                
proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.load_matches()
        
print "Loading match points (sba)..."
matches_sba = pickle.load( open( args.project + "/matches_sba", "rb" ) )
#f = open(args.project + "/Matches-sba.json", 'r')
#matches_sba = json.load(f)
#f.close()

# iterate through the sba match dictionary and build a list of feature
# points and heights (in x=east,y=north,z=up coordinates)
print "Building raw mesh interpolator"
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
i = scipy.interpolate.LinearNDInterpolator(tri, raw_values)

print "Building individual image meshes from matched features ..."
for image in proj.image_list:
    image.raw_points = []
    image.raw_values = []
    image.raw_indices = []
for match in matches_sba:
    ned = match[0]
    #raw_points.append( [ned[1], ned[0]] )
    #raw_values.append( -ned[2] )
    for p in match[1:]:
        image = proj.image_list[ p[0] ]
        image.raw_indices.append( p[1] )
        image.raw_points.append( [ned[1], ned[0]] )
        image.raw_values.append( -ned[2] )
for image in proj.image_list:
    if len(image.raw_points) >= 3:
        image.delaunay = scipy.spatial.Delaunay(np.array(image.raw_points))
        #image.interp = scipy.interpolate.LinearNDInterpolator(image.tri, image.raw_values)

# compute min/max range of horizontal surface
print "Determining coverage area"
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
print "Area coverage = %.1f,%.1f to %.1f,%.1f (%.1f x %.1f meters)" % \
    (x_min, y_min, x_max, y_max, x_max-x_min, y_max-y_min)

# compute number of connections and cycle depth per image
Matcher.groupByConnections(proj.image_list)

# start with empty triangle lists
# format: [ [v[0], v[1], v[2], u, v], .... ]
for image in proj.image_list:
    image.tris = []
    
good_tris = 0
failed_tris = 0

# compute image.PROJ for each image
for image in proj.image_list:
    rvec, tvec = image.get_proj_sba()
    R, jac = cv2.Rodrigues(rvec)
    image.PROJ = np.concatenate((R, tvec), axis=1)

# create an image list sorted by cycle depth
by_cycles = []
for i, image in enumerate(proj.image_list):
    by_cycles.append( (image.cycle_depth, i) )
by_cycles = sorted(by_cycles, key=lambda fields: fields[0])

mask = Polygon.Polygon()

# iterate image list in cycle depth order
for line in by_cycles:
    (cycle_depth, index) = line
    image = proj.image_list[index]
    print image.name
    if image.connections == 0:
        continue
    if len(image.raw_points) < 3:
        continue
    image.tris = []
    for tris in image.delaunay.simplices:
        # print tri
        t = []
        shape = []
        for vert in tris:
            shape.append( (image.raw_points[vert][1], image.raw_points[vert][0]) )
            v = []
            v.append( image.raw_points[vert][1] )
            v.append( image.raw_points[vert][0] )
            if args.depth:
                v.append( -image.raw_values[vert] + 2 * image.cycle_depth )
            else:
                v.append( 0 * image.cycle_depth )
            i_orig = image.raw_indices[vert] 
            v.append( image.kp_list[i_orig].pt[0] / image.width )
            v.append( (1.0 - image.kp_list[i_orig].pt[1]) / image.height )
            # print " ", v
            t.append(v)

        new = Polygon.Polygon(shape)
        if mask.covers(new):
            # do nothing
            a = 1
        elif False and mask.overlaps(new):
            # more complicated case
            rem = new - mask
            mask += new
            print "tri:", rem.triStrip()
            tristrip = rem.triStrip()
            for contour in tristrip:
                for i in range(2, len(contour)):
                    print contour[i-2]
                    print contour[i-1]
                    print contour[i]
            #image.tris.append(t)
        else:
            # easiest case, just add the tri
            mask += new
            image.tris.append(t)

print "Finished assigning tris..."

# write out an ac3d file
name = args.project + "/sba3d.ac"
f = open( name, "w" )
f.write("AC3Db\n")
trans = 0.0
f.write("MATERIAL \"\" rgb 1 1 1  amb 0.6 0.6 0.6  emis 0 0 0  spec 0.5 0.5 0.5  shi 10  trans %.2f\n" % (trans))
f.write("OBJECT world\n")
f.write("kids " + str(len(proj.image_list)) + "\n")

for image in proj.image_list:
    print image.name, len(image.tris)
    gen_ac3d_object(f, image.name, image.tris)
