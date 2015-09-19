#!/usr/bin/python

# For all the feature matches and camera poses, estimate a mean
# reprojection error

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import cv2
import json
import numpy as np

sys.path.append('../lib')
import ProjectMgr

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()

f = open(args.project + "/Matches.json", 'r')
matches_dict = json.load(f)
f.close()

def mean_reprojection_error():
    # start with a clean slate
    for image in proj.image_list:
        image.img_pts = []
        image.obj_pts = []

    # iterate through the match dictionary and build a per image list of
    # obj_pts and img_pts
    for key in matches_dict:
        feature_dict = matches_dict[key]
        points = feature_dict['pts']
        ned = matches_dict[key]['ned']
        for p in points:
            image = proj.image_list[ p[0] ]
            kp = image.kp_list[ p[1] ]
            image.img_pts.append( kp.pt )
            image.obj_pts.append( ned )

    camw, camh = proj.cam.get_image_params()
    for image in proj.image_list:
        print image.name
        if len(image.img_pts) < 4:
            continue
        image_sum = 0.0
        image_max = 0.0
        scale = float(image.width) / float(camw)
        K = proj.cam.get_K(scale)
        rvec, tvec = image.get_proj()
        R, jac = cv2.Rodrigues(rvec)
        PROJ = np.concatenate((R, tvec), axis=1)
        for i, pt in enumerate(image.obj_pts):
            uvh = K.dot( PROJ.dot( np.hstack((pt, 1.0)) ).T )
            #print uvh
            uvh /= uvh[2]
            #print uvh
            #print "%s -> %s" % ( image.img_pts[i], [ np.squeeze(uvh[0,0]), np.squeeze(uvh[1,0]) ] )
            uv = np.array( [ np.squeeze(uvh[0,0]), np.squeeze(uvh[1,0]) ] )
            dist = np.linalg.norm(np.array(image.img_pts[i]) - uv)
            #print dist
            image_sum += dist
            if dist > image_max:
                image_max = dist
        print "%s mre = %.2f max = %.2f" % (image.name,
                                            image_sum / len(image.obj_pts),
                                            image_max)
        
mean_reprojection_error()
