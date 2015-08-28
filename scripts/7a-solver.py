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

# working on matching features ...

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

m = Matcher.Matcher()

f = open(args.project + "/Matches.json", 'r')
matches_dict = json.load(f)
f.close()

d2r = math.pi / 180.0

# figure out some stuff
camw, camh = proj.cam.get_image_params()
for image in proj.image_list:
    continue

    print image.name
    scale = float(image.width) / float(camw)
    K = proj.cam.get_K(scale)
    IK = np.linalg.inv(K)
    #quat = image.camera_pose['quat']
    #Rquat = transformations.quaternion_matrix(quat)
    #print "Rquat:\n", Rquat
    ypr = image.camera_pose['ypr']
    IR = transformations.euler_matrix(ypr[0]*d2r, ypr[1]*d2r, ypr[2]*d2r, 'rzyx')
    print "IR:\n", IR

    # interesting to notice that inv(R) == transpose(R), so taking the
    # transpose should be faster than inverting.
    #R_inv = np.linalg.inv(Reul)
    #print "R_inv:\n", R_inv
    R = np.transpose(IR[:3,:3]) # equivalent to inverting
    print "R\n", R

    uv = np.array([0.0, 0.0, 1.0])
    x = IR[:3,:3].dot(IK).dot(uv)
    print "x:\n", x

    rvec, jac = cv2.Rodrigues(R[:3,:3])
    print "rvec = ", rvec
    newR, jac = cv2.Rodrigues(rvec)
    print "newR:\n", newR

    ned = image.camera_pose['ned']
    print "ned = ", ned
    tvec = -np.matrix(R[:3,:3]) * np.matrix(ned).T
    print "tvec =", tvec
    pos = -np.matrix(R[:3,:3]).T * np.matrix(tvec)
    print "pos = ", pos

# start with a clean slate
for image in proj.image_list:
    image.img_pts = []
    image.obj_pts = []
    
# iterate through the match dictionary and build a per image list of
# obj_pts and img_pts
for key in matches_dict:
    feature_dict = matches_dict[key]
    points = feature_dict['pts']
    ned = feature_dict['ned']
    for p in points:
        image = proj.image_list[ p[0] ]
        kp = image.kp_list[ p[1] ]
        image.img_pts.append( kp.pt )
        image.obj_pts.append( ned )
        
M = np.array( [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float )
IM = np.linalg.inv(M)

camw, camh = proj.cam.get_image_params()
for image in proj.image_list:
    print image.name
    scale = float(image.width) / float(camw)
    K = proj.cam.get_K(scale)
    if hasattr(image, 'rvec'):
        (result, image.rvec, image.tvec) \
            = cv2.solvePnP(np.float32(image.obj_pts),
                           np.float32(image.img_pts),
                           K, None,
                           image.rvec, image.tvec,
                           useExtrinsicGuess=True)
    else:
        # first time
        (result, image.rvec, image.tvec) \
            = cv2.solvePnP(np.float32(image.obj_pts),
                           np.float32(image.img_pts),
                           K, None)
    Rraw, jac = cv2.Rodrigues(image.rvec)
    #print "Rraw (from SolvePNP):\n", Rraw

    ned = image.camera_pose['ned']
    print "original ned = ", ned
    #tvec = -np.matrix(R[:3,:3]) * np.matrix(ned).T
    #print "tvec =", tvec
    pos = -np.matrix(Rraw[:3,:3]).T * np.matrix(image.tvec)
    print "pos = ", pos.tolist()

    # Our Rcam matrix (in our coordinate system) is inv(M) * Rned, so
    # solvePnP returns this combination.  We can extract Rned by
    # premultiplying by M aka inv(IM).
    R = M.dot(Rraw)
    #print "R (after M * R):\n", R

    ypr = image.camera_pose['ypr']
    print "original ypr = ", ypr
    IR = np.matrix(R).T
    IRo = transformations.euler_matrix(ypr[0]*d2r, ypr[1]*d2r, ypr[2]*d2r, 'rzyx')
    IRq = transformations.quaternion_matrix(image.camera_pose['quat'])
    #print "Original IR:\n", IRo
    #print "Original IR (from quat)\n", IRq
    #print "IR (from SolvePNP):\n", IR
    
    (yaw, pitch, roll) = transformations.euler_from_matrix(IR, 'rzyx')
    print "ypr =", [yaw/d2r, pitch/d2r, roll/d2r]

    #print "Proj =", np.concatenate((R, image.tvec), axis=1)
