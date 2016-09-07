#!/usr/bin/python

# 1. Iterate through all the image pairs and triangulate the match points.
# 2. Set the 3d location of features to triangulated position (possibly
#    averaged if the feature is included in multiple matches
# 3. Compute new camera poses with solvePnP() using triangulated point locations
# 4. Repeat

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

import argparse
import commands
import cPickle as pickle
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

# constants
d2r = math.pi / 180.0

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()
#proj.load_match_pairs()

m = Matcher.Matcher()

matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )
print "unique features:", len(matches_direct)


# iterate through the project image list and triangulate the 3d
# location for all feature points, given the current camera pose.
# Returns a new matches_dict with update point positions
def triangulate(matches_direct, cam_dict):
    IK = np.linalg.inv( proj.cam.get_K() )

    match_pairs = proj.generate_match_pairs(matches_direct)

    # parallel to match_dict to accumulate point coordinate sums
    sum_dict = {}
    count_dict= {}

    for i, i1 in enumerate(proj.image_list):
        #rvec1, tvec1 = i1.get_proj()
        rvec1 = cam_dict[i1.name]['rvec']
        tvec1 = cam_dict[i1.name]['tvec']
        R1, jac = cv2.Rodrigues(rvec1)
        PROJ1 = np.concatenate((R1, tvec1), axis=1)
        for j, i2 in enumerate(proj.image_list):
            matches = match_pairs[i][j]
            if (j <= i) or (len(matches) == 0):
                continue
            #rvec2, tvec2 = i2.get_proj()
            rvec2 = cam_dict[i2.name]['rvec']
            tvec2 = cam_dict[i2.name]['tvec']
            R2, jac = cv2.Rodrigues(rvec2)
            PROJ2 = np.concatenate((R2, tvec2), axis=1)

            uv1 = []; uv2 = []; indices = []
            for pair in matches:
                p1 = i1.kp_list[ pair[0] ].pt
                p2 = i2.kp_list[ pair[1] ].pt
                uv1.append( [p1[0], p1[1], 1.0] )
                uv2.append( [p2[0], p2[1], 1.0] )
                # pair[2] is the index back into the matches_direct structure
                indices.append( pair[2] )
            pts1 = IK.dot(np.array(uv1).T)
            pts2 = IK.dot(np.array(uv2).T)
            points = cv2.triangulatePoints(PROJ1, PROJ2, pts1[:2], pts2[:2])
            points /= points[3]
            #print "points:\n", points[0:3].T

            # fixme: need to update result, sum_dict is no longer used
            print "%s vs %s" % (i1.name, i2.name)
            for k, p in enumerate(points[0:3].T):
                match = matches_direct[indices[k]]
                match[0] += p
                key = "%d-%d" % (i, match[0])
                print key
                if key in sum_dict:
                    sum_dict[key] += p
                    count_dict[key] += 1
                else:
                    sum_dict[key] = p
                    count_dict[key] = 1

    # divide each element of sum_dict by the count of pairs to get the
    # average point location (so sum_dict becomes a location_dict)
    for key in sum_dict:
        count = count_dict[key]
        sum_dict[key] /= count

    # return the new dictionary.
    return sum_dict


# Iterate through the project image list and run solvePnP on each
# image's feature set to derive new estimated camera locations
cam1 = []
def solvePnP(matches_direct):
    # start with a clean slate
    for image in proj.image_list:
        image.img_pts = []
        image.obj_pts = []
    # build a new cam_dict that is a copy of the current one
    cam_dict = {}
    for image in proj.image_list:
        cam_dict[image.name] = {}
        rvec, tvec = image.get_proj()
        ned, ypr, quat = image.get_camera_pose()
        cam_dict[image.name]['rvec'] = rvec
        cam_dict[image.name]['tvec'] = tvec
        cam_dict[image.name]['ned'] = ned

    # iterate through the match dictionary and build a per image list of
    # obj_pts and img_pts
    for match in matches_direct:
        ned = match[0]
        for p in match[1:]:
            image = proj.image_list[ p[0] ]
            kp = image.kp_list[ p[1] ]
            image.img_pts.append( kp.pt )
            image.obj_pts.append( ned )

    camw, camh = proj.cam.get_image_params()
    for image in proj.image_list:
        print image.name
        if len(image.img_pts) < 4:
            continue
        scale = float(image.width) / float(camw)
        K = proj.cam.get_K(scale)
        rvec, tvec = image.get_proj()
        (result, rvec, tvec) = cv2.solvePnP(np.float32(image.obj_pts),
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

        #image.set_camera_pose( pos.T[0].tolist(), [yaw/d2r, pitch/d2r, roll/d2r] )
        #print "Proj =", np.concatenate((R, tvec), axis=1)
        cam_dict[image.name] = {}
        cam_dict[image.name]['rvec'] = rvec
        cam_dict[image.name]['tvec'] = tvec
        cam_dict[image.name]['ned'] = newned
    return cam_dict

# compute a 3d affine tranformation between current camera locations
# and original camera locations, then transform the current cameras
# back.  This way the retain their new relative positioning without
# walking away from each other in scale or position.
def recenter(cam_dict):
    src = [[], [], [], []]      # current camera locations
    dst = [[], [], [], []]      # original camera locations
    for image in proj.image_list:
        if image.name in cam_dict:
            newned = cam_dict[image.name]['ned']
        else:
            newned, ypr, quat = image.get_camera_pose()
        src[0].append(newned[0])
        src[1].append(newned[1])
        src[2].append(newned[2])
        src[3].append(1.0)
        origned, ypr, quat = image.get_camera_pose()
        dst[0].append(origned[0])
        dst[1].append(origned[1])
        dst[2].append(origned[2])
        dst[3].append(1.0)
        print "%s %s" % (origned, newned)
    Aff3D = transformations.superimposition_matrix(src, dst, scale=True)
    print "Aff3D:\n", Aff3D
    scale, shear, angles, trans, persp = transformations.decompose_matrix(Aff3D)
    R = transformations.euler_matrix(*angles)
    print "R:\n", R
    # rotate, translate, scale the group of camera positions to best
    # align with original locations
    update_cams = Aff3D.dot( np.array(src) )
    print update_cams[:3]
    for i, p in enumerate(update_cams.T):
        key = proj.image_list[i].name
        if not key in cam_dict:
            cam_dict[key] = {}
        ned = [ p[0], p[1], p[2] ]
        print "ned:", ned
        cam_dict[key]['ned'] = ned
        # adjust the camera projection matrix (rvec) to rotate by the
        # amount of the affine transformation as well
        rvec = cam_dict[key]['rvec']
        tvec = cam_dict[key]['tvec']
        Rcam, jac = cv2.Rodrigues(rvec)
        print "Rcam:\n", Rcam
        Rcam_new = R[:3,:3].dot(Rcam)
        print "Rcam_new:\n", Rcam_new
        
        rvec, jac = cv2.Rodrigues(Rcam_new)
        cam_dict[key]['rvec'] = rvec
        tvec = -np.matrix(Rcam_new) * np.matrix(ned).T
        cam_dict[key]['tvec'] = tvec
        
# return a 3d affine tranformation between current camera locations
# and original camera locations.
def get_recenter_affine(cam_dict):
    src = [[], [], [], []]      # current camera locations
    dst = [[], [], [], []]      # original camera locations
    for image in proj.image_list:
        if image.name in cam_dict:
            newned = cam_dict[image.name]['ned']
        else:
            newned, ypr, quat = image.get_camera_pose()
        src[0].append(newned[0])
        src[1].append(newned[1])
        src[2].append(newned[2])
        src[3].append(1.0)
        origned, ypr, quat = image.get_camera_pose()
        dst[0].append(origned[0])
        dst[1].append(origned[1])
        dst[2].append(origned[2])
        dst[3].append(1.0)
        print "%s %s" % (origned, newned)
    A = transformations.superimposition_matrix(src, dst, scale=True)
    return A

def transform_points( A, pts_dict ):
    src = [[], [], [], []]
    for key in pts_dict:
        p = pts_dict[key]
        src[0].append(p[0])
        src[1].append(p[1])
        src[2].append(p[2])
        src[3].append(1.0)
    dst = A.dot( np.array(src) )
    result_dict = {}
    for i, key in enumerate(pts_dict):
        result_dict[key] = [ dst[0][i], dst[1][i], dst[2][i] ]
    return result_dict

def plot(surface0, cam0, surface1, cam1):
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

    xs = []; ys = []; zs = []
    for p in cam1:
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])
    ax.scatter(np.array(xs), np.array(ys), np.array(zs), c='b', marker='^')

    plt.show()

# iterate through the match dictionary and build a simple list of
# starting surface points
surface0 = []
for match in matches_direct:
    ned = match[0]
    surface0.append( [ned[1], ned[0], -ned[2]] )

cam0 = []
for image in proj.image_list:
    ned, ypr, quat = image.get_camera_pose()
    cam0.append( [ned[1], ned[0], -ned[2]] )

# iterate through the image list and build the camera pose dictionary
# (and a simple list of camera locations for plotting)
cam_dict = {}
for image in proj.image_list:
    rvec, tvec = image.get_proj()
    cam_dict[image.name] = {}
    cam_dict[image.name]['rvec'] = rvec
    cam_dict[image.name]['tvec'] = tvec

count = 0
while True:
    # run the triangulation step
    newpts_dict = triangulate(matches_direct, cam_dict)
    #print pts_dict

    surface1 = []
    for key in newpts_dict:
        p = newpts_dict[key]
        surface1.append( [ p[1], p[0], -p[2] ] )

    # find the optimal camera poses for the triangulation averaged
    # together.
    cam_dict = solvePnP(matches_direct)

    # get the affine transformation required to bring the new camera
    # locations back into a best fit with the original camera
    # locations
    A = get_recenter_affine(cam_dict)

    # transform all the feature points by the affine matrix
    newspts_dict = transform_points(A, newpts_dict)
    
    # run solvePnP now on the updated points (hopefully this will
    # naturally reorient the cameras as needed.)
    # 9/6/2016: shouldn't be needed since transform_points() now rotates
    # the camera orientation as well?
    # cam_dict = solvePnP(newpts_dict)
    
    cam1 = []
    for key in cam_dict:
        p = cam_dict[key]['ned']
        cam1.append( [ p[1], p[0], -p[2] ] )

    if count % 10 == 0:
        plot(surface0, cam0, surface1, cam1)

    count += 1
