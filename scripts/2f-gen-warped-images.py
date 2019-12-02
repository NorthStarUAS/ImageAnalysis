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

sys.path.append('../lib')
import ProjectMgr
import Render
import SRTM
import transformations

# for all the images in the project image_dir, compute the camera
# poses from the aircraft pose (and camera mounting transform).
# Project the image plane onto an SRTM (DEM) surface for our best
# layout guess (at this point before we do any matching/bundle
# adjustment work.)

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--pose', required=True, default='direct',
                    choices=(['direct', 'sba']), help='select pose')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()

ref = proj.ned_reference_lla

# setup SRTM ground interpolator
sss = SRTM.NEDGround( ref, 2000, 2000, 30 )

camw, camh = proj.cam.get_image_params()
dist_coeffs = proj.cam.get_dist_coeffs()
for image in proj.image_list:
    print image.name
    scale = float(image.width) / float(camw)
    K = proj.cam.get_K(scale)
    IK = np.linalg.inv(K)
    corner_list = []
    corner_list.append( [0, 0] )
    corner_list.append( [image.width, 0] )
    corner_list.append( [0, image.height] )
    corner_list.append( [image.width, image.height] )
    
    proj_list = proj.projectVectors( IK, image, corner_list, pose=args.pose )
    #print "proj_list:\n", proj_list
    if args.pose == 'direct':
        pts_ned = sss.interpolate_vectors(image.camera_pose, proj_list)
    elif args.pose == 'sba':
        pts_ned = sss.interpolate_vectors(image.camera_pose_sba, proj_list)
    # print "pts (ned):\n", pts_ned
    
    image.corner_list_ned = []
    image.corner_list_lla = []
    image.corner_list_xy = []
    for ned in pts_ned:
        #print p
        image.corner_list_ned.append( [ned[0], ned[1]] )
        image.corner_list_lla.append( navpy.ned2lla([ned], ref[0], ref[1], ref[2]) )
        image.corner_list_xy.append( [ned[1], ned[0]] )


dst_dir = proj.project_dir + "/Warped/"
if not os.path.exists(dst_dir):
    print "Notice: creating rubber sheeted texture directory =", dst_dir
    os.makedirs(dst_dir)

for image in proj.image_list:
    basename, ext = os.path.splitext(image.name)
    dst = dst_dir + basename + ".png"
    #if os.path.exists(dst):
    #    continue
    # print image.name
    scale = float(image.width) / float(camw)
    K = proj.cam.get_K(scale)
    x, y, warped = proj.render.drawImage(image, K, dist_coeffs,
                                         proj.source_dir, cm_per_pixel=20)
    #print image.coverage_ned()
    (minlon, minlat, maxlon, maxlat) = image.coverage_lla(ref)
    cv2.imshow('warped', warped)
    #print 'waiting for keyboard input...'
    #key = cv2.waitKey() & 0xff
    basename, ext = os.path.splitext(image.name)
    print "var imageUrl = " + "'Warped/" + basename + ".png'"
    print "imageBounds = [[%.10f, %.10f], [%.10f, %.10f]];" % (minlat, minlon, maxlat, maxlon)

    print "L.imageOverlay(imageUrl, imageBounds).addTo(map);"
    print ""
    cv2.imwrite(dst, warped)
    # make all black transparent (edges)
    command = "convert -transparent black %s %s" % ( dst, dst )
    commands.getstatusoutput( command )
