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



parser = argparse.ArgumentParser(description='Compute Delauney triangulation of matches.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.load_matches()

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
for tri in tri.simplices:
    print "Triangle:", tri
    union = {}
    done = False 
    # iterate through each vertex index of the tri
    for vert in tri:
        feat = matches_sba[rkeys[vert]]
        pts = feat['pts']
        # iterate through each point/image match of the vertex/feature
        for p in pts:
            if p[0] in union:
                union[p[0]] += 1
            else:
                union[p[0]] = 1
            # print " ", vert, matches_sba[rkeys[vert]], union
    if not done:
        # snag the easy cases first
        best_image = -1
        best_connections = -1
        for key in union:
            if union[key] >= 3:
                image = proj.image_list[key]
                # print "covered by image: %d (%d)" % (key, image.connections)
                if image.connections > best_connections:
                    best_connections = image.connections
                    best_image = key
        if best_image >= 0:
            print "Best image (easy): %d (%d)" % (best_image, best_connections)
            image = proj.image_list[best_image]
            triangle = []
            for vert in tri:
                feat = matches_sba[rkeys[vert]]
                ned = feat['ned']
                pts = feat['pts']
                for p in pts:
                    #print p
                    if p[0] == best_image:
                        v = list(ned)
                        v.append( image.kp_list[p[1]].pt[0] / image.width )
                        v.append( image.kp_list[p[1]].pt[1] / image.height )
                        triangle.append(v)
            print " ", triangle
            image.tris.append(triangle)
            easy_tris += 1
            done = True
    if not done:
        # look for complete coverage, possibly estimating uv by
        # reprojection if a feature wasn't found originally
        best_image = -1
        best_connections = -1
        best_triangle = []
        for key in union:
            image = proj.image_list[key]
            # print image.name
            ok = True
            triangle = []
            for vert in tri:
                feat = matches_sba[rkeys[vert]]
                ned = feat['ned']
                uv = compute_feature_uv(proj.cam.get_K(), image, ned)
                uv[0] /= image.width
                uv[1] /= image.height
                #print " ", uv
                v = list(ned)
                v.append(uv[0])
                v.append(uv[1])
                triangle.append(v)
                if uv[0] < 0.0 or uv[0] > 1.0:
                    #print "  fail"
                    ok = False
                if uv[1] < 0.0 or uv[1] > 1.0:
                    #print "  fail"
                    ok = False
            if ok:
                # print " pass!"
                if image.connections > best_connections:
                    best_connections = image.connections
                    best_image = key
                    best_triangle = list(triangle)
        if best_image >= 0:
            print "Best image (hard): %d (%d)" % (best_image, best_connections)
            print "  ", best_triangle
            image.tris.append(best_triangle)
            hard_tris += 1
            done = True
    if not done:
        print "failed triangle"
        failed_tris += 1

print "easy tris =", easy_tris
print "hard tris =", hard_tris
print "failed tris =", failed_tris

for image in proj.image_list:
    print image.name, len(image.tris)
