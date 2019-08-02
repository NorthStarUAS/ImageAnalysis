#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import numpy as np
import os.path
import random
import navpy

import simplekml

sys.path.append('../lib')
import Pose
import ProjectMgr
import SRTM
import transformations

# for all the images in the project image_dir, compute the camera
# poses from the aircraft pose (and camera mounting transform).
# Project the image plane onto an SRTM (DEM) surface for our best
# layout guess (at this point before we do any matching/bundle
# adjustment work.)

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()

ref = proj.ned_reference_lla

# setup SRTM ground interpolator
sss = SRTM.NEDGround( ref, 2000, 2000, 30 )

# start a new kml file
kml = simplekml.Kml()

camw, camh = proj.cam.get_image_params()
for image in proj.image_list:
    print image.name
    scale = float(image.width) / float(camw)
    K = proj.cam.get_K(scale)
    IK = np.linalg.inv(K)
    corner_list = []
    corner_list.append( [0, image.height] )
    corner_list.append( [image.width, image.height] )
    corner_list.append( [image.width, 0] )
    corner_list.append( [0, 0] )
    
    proj_list = proj.projectVectors( IK, image, corner_list )
    print "proj_list:\n", proj_list
    #pts = proj.intersectVectorsWithGroundPlane(image.camera_pose['ned'],
    #                                           g, proj_list)
    pts = sss.interpolate_vectors(image.camera_pose, proj_list)
    #print "pts (ned):\n", pts
    
    corners_lonlat = []
    for ned in pts:
        print ned
        lla = navpy.ned2lla([ned], ref[0], ref[1], ref[2])
        corners_lonlat.append([lla[1], lla[0]])
    ground = kml.newgroundoverlay(name=image.name)
    ground.icon.href = "Images/" + image.name
    ground.gxlatlonquad.coords.addcoordinates(corners_lonlat)

filename = args.project + "/GroundOverlay.kml"
kml.save(filename)
