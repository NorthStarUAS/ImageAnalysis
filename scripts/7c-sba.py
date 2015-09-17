#!/usr/bin/python

# write out the data in a form useful to pass to the sba (demo) program

# it appears camera poses are basically given as [ R | t ] where R is
# the same R we use throughout and t is the 'tvec'

# todo, run sba and automatically parse output ...

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

# return a quat (of the R matrix) for the given rvec
def rvec2quat(rvec):
    R1 = np.array( [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float )
    Rned2cam, jac = cv2.Rodrigues(rvec)
    Rned2sba = R1.dot(Rned2cam)
    # make 3x3 rotation matrix into 4x4 homogeneous matrix
    hIR = np.concatenate( (np.concatenate( (Rned2sba, np.zeros((3,1))),1),
                           np.mat([0,0,0,1])) )
    quat = transformations.quaternion_from_matrix(hIR)
    return quat

# iterate through the image list and build the camera pose dictionary
# (and a simple list of camera locations for plotting)
f = open( args.project + '/sba-cams.txt', 'w' )
for image in proj.image_list:
    # try #1
    body2cam = image.get_body2cam()
    ned2body = image.get_ned2body()
    # R1 = np.array( [[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float )
    Rtotal = body2cam.dot( ned2body )
    q = transformations.quaternion_from_matrix(Rtotal)

    # try #2
    rvec, tvec = image.get_proj()
    # q = rvec2quat(rvec)
    
    # try #3
    # ned2body = image.get_ned2body()
    # q = transformations.quaternion_from_matrix(ned2body)

    ned = np.array(image.camera_pose['ned']) / 1.0
    #s = "%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n" % (q[0], q[1], q[2], q[3],
    #                                              ned[1], ned[0], -ned[2])
    s = "%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n" % (q[0], q[1], q[2], q[3],
                                                  tvec[0,0], tvec[1,0], tvec[2,0])
    f.write(s)
    print "ned=%s tvec=%s" % (ned, np.squeeze(tvec))
f.close()

# iterate through the matches dictionary to produce a list of matches
f = open( args.project + '/sba-points.txt', 'w' )
for key in matches_dict:
    feat = matches_dict[key]
    ned = np.array(feat['ned']) / 1.0
    s = "%.8f %.8f %.8f " % (ned[0], ned[1], ned[2])
    f.write(s)
    pts = feat['pts']
    s = "%d " % (len(pts))
    f.write(s)
    for p in pts:
        image_num = p[0]
        kp = proj.image_list[image_num].kp_list[p[1]]
        s = "%d %.2f %.2f " % (image_num, kp.pt[0], kp.pt[1])
        f.write(s)
    f.write('\n')
f.close()

# print the calibration matrix "K"
f = open( args.project + '/sba-calib.txt', 'w' )
K = proj.cam.get_K()
s = "%.4f %.4f %.4f\n" % (K[0,0], K[0,1], K[0,2])
f.write(s)
s = "%.4f %.4f %.4f\n" % (K[1,0], K[1,1], K[1,2])
f.write(s)
s = "%.4f %.4f %.4f\n" % (K[2,0], K[2,1], K[2,2])
f.write(s)
