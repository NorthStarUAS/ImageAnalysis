#!/usr/bin/python

# for all the images in the project image_dir, compute the camera
# poses from the aircraft pose (and camera mounting transform).
# Project the image plane onto an SRTM (DEM) surface for our best
# layout guess (at this point before we do any matching/bundle
# adjustment work.)

import sys
sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import argparse
import cPickle as pickle
import cv2
import numpy as np
import os.path

sys.path.append('../lib')
import Groups
import ProjectMgr

import match_culling as cull

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

print "Loading match points (sba)..."
matches_direct = pickle.load( open( args.project + "/matches_sba", "rb" ) )

# load the group connections within the image set
groups = Groups.load(args.project)

# traverse the matches_direct structure and create a pair-wise match
# structure.  (Start with an empty n x n list of empty pair lists,
# then fill in the structures.)
pairs = []
for i in range(len(proj.image_list)):
    p = [ [] for j in range(len(proj.image_list)) ]
    pairs.append(p)
for match in matches_direct:
    for p1 in match[1:]:
        for p2 in match[2:]:
            if p1[0] == p2[0]:
                continue
            i1 = p1[0]
            i2 = p2[0]
            pairs[ p1[0] ][ p2[0] ].append( [ p1[1], p2[1] ] )
# print pairs

# sanity check (display pairs)
# import Matcher
# m = Matcher.Matcher()
# for i in range(len(proj.image_list)):
#     for j in range(len(proj.image_list)):
#         if len(pairs[i][j]):
#             print i, j, len(pairs[i][j])
#             i1 = proj.image_list[i]
#             i2 = proj.image_list[j]
#             status = m.showMatchOrient(i1, i2, pairs[i][j] )
            
# for each image, find all the placed features, and compute an average
# elevation
for image in proj.image_list:
    image.z_list = []
    image.grid_list = []
for match in matches_sba:
    ned = match[0]
    for p in match[1:]:
        index = p[0]
        proj.image_list[index].z_list.append(-ned[2])
for image in proj.image_list:
    if len(image.z_list):
        avg = np.mean(np.array(image.z_list))
        std = np.std(np.array(image.z_list))
    else:
        avg = None
        std = None
    image.z_avg = avg
    image.z_std = std
    print image.name, 'features:', len(image.z_list), 'avg:', avg, 'std:', std

# for fun rerun through the matches and find elevation outliers
outliers = []
for i, match in enumerate(matches_sba):
    ned = match[0]
    error_sum = 0
    for p in match[1:]:
        image = proj.image_list[p[0]]
        dist = abs(-ned[2] - image.z_avg)
        error_sum += dist
    if error_sum > 3 * (image.z_std * len(match[1:])):
        print 'possible outlier match index:', i, error_sum, 'z:', ned[2]
        outliers.append( [error_sum, i] )

result = sorted(outliers, key=lambda fields: fields[0], reverse=True)
for line in result:
    print 'index:', line[1], 'error:', line[0]
    #cull.draw_match(line[1], 1, matches_sba, proj.image_list)
    
depth = 0.0
camw, camh = proj.cam.get_image_params()
#for group in groups:
if True:
    group = groups[0]
    #if len(group) < 3:
    #    continue
    for g in group:
        image = proj.image_list[g]
        print image.name, image.z_avg
        # scale the K matrix if we have scaled the images
        scale = float(image.width) / float(camw)
        K = proj.cam.get_K(scale)
        IK = np.linalg.inv(K)

        grid_list = []
        u_list = np.linspace(0, image.width, ac3d_steps + 1)
        v_list = np.linspace(0, image.height, ac3d_steps + 1)
        #print "u_list:", u_list
        #print "v_list:", v_list
        for v in v_list:
            for u in u_list:
                grid_list.append( [u, v] )
        #print 'grid_list:', grid_list

        if args.direct:
            proj_list = proj.projectVectors( IK, image.get_body2ned(),
                                             image.get_cam2body(), grid_list )
        else:
            print image.get_body2ned_sba()
            proj_list = proj.projectVectors( IK, image.get_body2ned_sba(),
                                             image.get_cam2body(), grid_list )
        #print 'proj_list:', proj_list

        if args.direct:
            ned = image.camera_pose['ned']
        else:
            ned = image.camera_pose_sba['ned']
        print 'ned', image.camera_pose['ned'], ned
        if args.ground:
            pts_ned = proj.intersectVectorsWithGroundPlane(ned,
                                                           args.ground, proj_list)
        # convert ned to xyz and stash the result for each image
        image.grid_list = []
        ground_sum = 0
        for p in pts_ned:
            image.grid_list.append( [p[1], p[0], -(p[2]+depth)] )
            #image.grid_list.append( [p[1], p[0], -(depth)] )
            ground_sum += -p[2]
        depth -= 0.01                # favor last pictures above earlier ones
