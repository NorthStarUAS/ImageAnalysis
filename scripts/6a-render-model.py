#!/usr/bin/python

# for all the images in the project image_dir, compute the camera
# poses from the aircraft pose (and camera mounting transform).
# Project the image plane onto an SRTM (DEM) surface for our best
# layout guess (at this point before we do any matching/bundle
# adjustment work.)

import sys
sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import argparse
import commands
import cPickle as pickle
import cv2
import fnmatch
import numpy as np
import os.path
import random

sys.path.append('../lib')
import AC3D
import Groups
import Pose
import ProjectMgr
import SRTM
import transformations

ac3d_steps = 8

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--texture-resolution', type=int, default=512, help='texture resolution (should be 2**n, so numbers like 256, 512, 1024, etc.')
parser.add_argument('--srtm', action='store_true', help='use srtm elevation')
parser.add_argument('--ground', type=float, help='force ground elevation in meters')
parser.add_argument('--direct', action='store_true', help='use direct pose')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()

ref = proj.ned_reference_lla

# setup SRTM ground interpolator
sss = SRTM.NEDGround( ref, 6000, 6000, 30 )

print "Loading match points (sba)..."
matches_sba = pickle.load( open( args.project + "/matches_sba", "rb" ) )

# load the group connections within the image set
groups = Groups.load(args.project)

# compute the uv grid for each image and project each point out into
# ned space, then intersect each vector with the srtm ground.

# for each image, find all the placed features, and compute an average
# elevation
for image in proj.image_list:
    image.temp_sum = 0.0
    image.temp_count = 0
    image.grid_list = []
for match in matches_sba:
    ned = match[0]
    for p in match[1:]:
        index = p[0]
        proj.image_list[index].temp_sum += -ned[2]
        proj.image_list[index].temp_count += 1
for image in proj.image_list:
    if image.temp_count > 0:
        average = image.temp_sum / image.temp_count
    else:
        average = None
    image.average_elevation = average
    print image.name, 'features:', image.temp_count, 'average:', average

depth = 0.0
camw, camh = proj.cam.get_image_params()
#for group in groups:
if True:
    group = groups[0]
    #if len(group) < 3:
    #    continue
    for g in group:
        image = proj.image_list[g]
        print image.name, image.average_elevation
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
        elif args.srtm:
            pts_ned = sss.interpolate_vectors(ned, proj_list)
        else:
            pts_ned = proj.intersectVectorsWithGroundPlane(ned,
                                                           image.average_elevation, proj_list)
        #print "pts_3d (ned):\n", pts_ned

        # convert ned to xyz and stash the result for each image
        image.grid_list = []
        ground_sum = 0
        for p in pts_ned:
            #image.grid_list.append( [p[1], p[0], -(p[2]+depth)] )
            image.grid_list.append( [p[1], p[0], -(depth)] )
            ground_sum += -p[2]
        depth -= 0.01                # favor last pictures above earlier ones
    
# call the ac3d generator
AC3D.generate(proj.image_list, src_dir=proj.source_dir,
              project_dir=args.project, base_name='direct',
              version=1.0, trans=0.1, resolution=args.texture_resolution)

if not args.ground:
    print 'Avg ground elevation (SRTM):', ground_sum / len(pts_ned)
