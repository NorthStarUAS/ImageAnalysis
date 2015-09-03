#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import json
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
proj.load_matches()
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

# iterate through the match dictionary and build a simple list of
# starting surface points
surface0 = []
for key in matches_dict:
    feature_dict = matches_dict[key]
    ned = feature_dict['ned']
    surface0.append( [ned[1], ned[0], -ned[2]] )

# iterate through the image list and build a simple list of camera locations
cam0 = []
for image in proj.image_list:
    ned, ypr, quat = image.get_camera_pose()
    cam0.append( [ned[1], ned[0], -ned[2]] )

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

camw, camh = proj.cam.get_image_params()
cam1 = []
for image in proj.image_list:
    # skip this step for the moment (so we can try triangulating from
    # original camera positions
    continue

    print image.name
    if len(image.img_pts) < 4:
        continue
    scale = float(image.width) / float(camw)
    K = proj.cam.get_K(scale)
    rvec, tvec = image.get_proj()
    (rvec, tvec, result) = cv2.solvePnPRansac(np.float32(image.obj_pts),
                                        np.float32(image.img_pts),
                                        K, None,
                                        rvec, tvec, useExtrinsicGuess=True)
    print "rvec=", rvec
    print "tvec=", tvec
    Rned2cam, jac = cv2.Rodrigues(rvec)
    #print "Rraw (from SolvePNP):\n", Rraw

    ned = image.camera_pose['ned']
    print "original ned = ", ned
    #tvec = -np.matrix(R[:3,:3]) * np.matrix(ned).T
    #print "tvec =", tvec
    pos = -np.matrix(Rned2cam[:3,:3]).T * np.matrix(tvec)
    newned = pos.T[0].tolist()[0]
    print "new ned =", newned
    cam1.append( [newned[1], newned[0], -newned[2]] )

    # Our Rcam matrix (in our ned coordinate system) is body2cam * Rned,
    # so solvePnP returns this combination.  We can extract Rned by
    # premultiplying by cam2body aka inv(body2cam).
    cam2body = image.get_cam2body()
    Rned2body = cam2body.dot(Rned2cam)
    #print "R (after M * R):\n", R

    ypr = image.camera_pose['ypr']
    print "original ypr = ", ypr
    Rbody2ned = np.matrix(Rned2body).T
    IRo = transformations.euler_matrix(ypr[0]*d2r, ypr[1]*d2r, ypr[2]*d2r, 'rzyx')
    IRq = transformations.quaternion_matrix(image.camera_pose['quat'])
    #print "Original IR:\n", IRo
    #print "Original IR (from quat)\n", IRq
    #print "IR (from SolvePNP):\n", IR
    
    (yaw, pitch, roll) = transformations.euler_from_matrix(Rbody2ned, 'rzyx')
    print "ypr =", [yaw/d2r, pitch/d2r, roll/d2r]

    image.set_camera_pose( pos.T[0].tolist(), [yaw/d2r, pitch/d2r, roll/d2r] )
    
    #print "Proj =", np.concatenate((R, tvec), axis=1)

IK = np.linalg.inv( proj.cam.get_K() )
surface1 = []
for i, i1 in enumerate(proj.image_list):
    rvec1, tvec1 = i1.get_proj()
    R1, jac = cv2.Rodrigues(rvec1)
    PROJ1 = np.concatenate((R1, tvec1), axis=1)
    for j, i2 in enumerate(proj.image_list):
        matches = i1.match_list[j]
        if len(matches) == 0:
            continue
        rvec2, tvec2 = i2.get_proj()
        R2, jac = cv2.Rodrigues(rvec2)
        PROJ2 = np.concatenate((R2, tvec2), axis=1)

        uv1 = []; uv2 = []
        for k, pair in enumerate(matches):
            p1 = i1.kp_list[ pair[0] ].pt
            p2 = i2.kp_list[ pair[1] ].pt
            uv1.append( [p1[0], p1[1], 1.0] )
            uv2.append( [p2[0], p2[1], 1.0] )
        pts1 = IK.dot(np.array(uv1).T)
        pts2 = IK.dot(np.array(uv2).T)
        points = cv2.triangulatePoints(PROJ1, PROJ2, pts1[:2], pts2[:2])
        
        #print "pts1 reproject:"
        #x1 = PROJ1.dot(points)
        #x1 /= x1[2]
        #for k in range(len(matches)):
        #    #print "p=%s x1=%s" % (p[:,k], x1[:,k])
        #    print "orig1=%s reproj=%s" % ( pts1[:,k].tolist(), x1[0:2,k].T.tolist()[0] )
        #print "pts2 reproject:"
        #x2 = PROJ2.dot(points)
        #x2 /= x2[2]
        #for k in range(len(matches)):
        #    #print "p=%s x1=%s" % (p[:,k], x1[:,k])
        #    print "orig1=%s reproj=%s" % ( pts2[:,k].tolist(), x2[0:2,k].T.tolist()[0] )
        
        points /= points[3]
        #print "points:\n", points[0:3].T
        
        print "%s vs %s" % (i1.name, i2.name)
        for k, p in enumerate(points[0:3].T):
            match = matches[k]
            key = "%d-%d" % (i, match[0])
            print key
            if not key in matches_dict:
                key = "%d-%d" % (j, match[1])
                print key
            pnew = p.tolist()
            print "new=", pnew
            print "1st guess=", matches_dict[key]['ned']
            surface1.append( [pnew[1], pnew[0], -pnew[2]] )

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
xs = []; ys = []; zs = []
for p in surface0:
    xs.append(p[0])
    ys.append(p[1])
    zs.append(p[2])
ax.scatter(np.array(xs), np.array(ys), np.array(zs), c='r', marker='.')

xs = []; ys = []; zs = []
for p in surface1:
    xs.append(p[0])
    ys.append(p[1])
    zs.append(p[2])
ax.scatter(np.array(xs), np.array(ys), np.array(zs), c='b', marker='.')

xs = []; ys = []; zs = []
for p in cam0:
    xs.append(p[0])
    ys.append(p[1])
    zs.append(p[2])
ax.scatter(np.array(xs), np.array(ys), np.array(zs), c='y', marker='^')

plt.show()
