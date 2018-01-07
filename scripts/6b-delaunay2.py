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
from progress.bar import Bar
import scipy.spatial

sys.path.append('../lib')
import Matcher
import Pose
import ProjectMgr
import SRTM
import transformations


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
    for v in vertex_list:
        #f.write("%.3f %.3f %.3f\n" % (v[1], v[0], -v[2]))
        f.write("%.3f %.3f %.3f\n" % (v[1], v[0], 0.0))
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
                
parser = argparse.ArgumentParser(description='Compute Delauney triangulation of matches.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.load_match_pairs()
        
print "Loading match points..."
f = open(args.project + "/Matches-sba.json", 'r')
matches_sba = json.load(f)
f.close()

# iterate through the sba match dictionary and build a list of feature
# points
points = []
rkeys = []
for key in matches_sba:
    feature_dict = matches_sba[key]
    ned = feature_dict['ned']
    points.append( [ned[1], ned[0]] )
    rkeys.append( key )

# compute number of connections per image
for image in proj.image_list:
    image.connections = 0
    for pairs in image.match_list:
        if len(pairs) >= 8:
            image.connections += 1
    #if image.connections > 1:
    #    print "%s connections: %d" % (image.name, image.connections)

# start with empty triangle lists
# format: [ [v[0], v[1], v[2], u, v], .... ]
for image in proj.image_list:
    image.tris = []
    
print "Building triangulation..."
tri = scipy.spatial.Delaunay(np.array(points))

print "Points:", len(points)
print "Triangles:", len(tri.simplices)

easy_tris = 0
hard_tris = 0
failed_tris = 0

# make sure we start with an empty projection matrix for each image
for image in proj.image_list:
    image.PROJ = None
        
# iterate through the triangle list
bar = Bar('Computing 3d model:', max=len(tri.simplices),
          suffix='%(percent).1f%% - %(eta)ds')
bar.sma_window = 100
camw, camh = proj.cam.get_image_params()
fuzz = 20.0
count = 0
update_steps = 50
for tri in tri.simplices:
    # print "Triangle:", tri

    # compute triangle center
    sum = np.array( [0.0, 0.0, 0.0] )
    for v in tri:
        feat = matches_sba[rkeys[v]]
        ned = feat['ned']
        sum += np.array( ned )
    tri_center = sum / len(tri)

    # look for complete coverage, possibly estimating uv by
    # reprojection if a feature wasn't found originally
    done = False 
    best_image = None
    best_connections = -1
    best_distance = 1000000.0
    best_triangle = []
    for image in proj.image_list:
        ok = True
        # quick 3d bounding radius rejection
        dist = np.linalg.norm(image.center - tri_center)
        if dist > image.radius + fuzz:
            ok = False
            continue
        # we passed the initial proximity test
        triangle = []
        for vert in tri:
            feat = matches_sba[rkeys[vert]]
            ned = feat['ned']
            scale = float(image.width) / float(camw)
            uv = compute_feature_uv(proj.cam.get_K(scale), image, ned)
            dist_coeffs = proj.cam.get_dist_coeffs()
            uv[0], uv[1] = redistort(uv[0], uv[1], dist_coeffs, proj.cam.get_K(scale))
            uv[0] /= image.width
            uv[1] /= image.height
            v = list(ned)
            v.append(uv[0])
            v.append(1.0 - uv[1])
            triangle.append(v)
            if uv[0] < 0.0 or uv[0] > 1.0:
                #print "  fail"
                ok = False
            if uv[1] < 0.0 or uv[1] > 1.0:
                #print "  fail"
                ok = False
        if ok:
            # print " pass!"
            # compute center of triangle
            dist = np.linalg.norm( image.camera_pose_sba['ned'] - tri_center )
            if dist < best_distance:
                best_distance = dist
                best_image = image
                best_triangle = list(triangle)
    if not best_image == None:
        # print "Best image (hard): %d (%d)" % (best_image, best_connections)
        # print "  ", best_triangle
        best_image.tris.append(best_triangle)
        hard_tris += 1
        done = True
    if not done:
        # print "failed triangle"
        failed_tris += 1
    count += 1
    if count % update_steps == 0:
        bar.next(update_steps)
bar.finish()

print "easy tris =", easy_tris
print "hard tris =", hard_tris
print "failed tris =", failed_tris

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
