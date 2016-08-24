#!/usr/bin/python

# TODO: work on cam_dict management so we keep track of the original
# cam_dict from gps/flight data for reorienting the entire pose set,
# but also correctly passes along refined camera pose information
# through the iterative process.

# triangulation estimates the 3d location of matching points from two
# camera poses.  However the orientation of the triangulated point
# surface is highly biased by orientation errors in the camera project
# matrices.  The scale of the resulting points are biased by errors in
# the 3d distance of the two camera poses.
#
# Any solver needs to simultaneously correct scale errors and
# orientation errors.
#
# solvePnP() estimates camera pose from 3d location of matching
# feature points, but it also highly biased by scale and orientation
# errors in the estimated 3d locations of the object points.
#
# (1) Estimating camera pose by iterating on triangulation, solvePnP
# never gets rid of orientation bias, but does appear to converge to
# the correct scale.
#
# (2)For each pair of matching images, we can triangulate the 3d
# points from current camera pose estimates, then solve pnp using the
# shared object points for each camera pose individually.  This gives
# us the correct relative poses between cameras.  Then we can find the
# centroid of 3d object points along with the before and after camera
# locations to define 3 matching points.  Then we can compute an
# affine (rotation, translation only) matrix to rotate the camera
# poses back to the better camera locations.  This corrects for
# orientation errors, but does not correct scale errors.
#
# This script attempts to combine global triangluation (and averaging
# the locations of all the resulting points + combined solvepnp (1) to
# improve scale errors + individual solvepnp and rotate back to
# correct for orientation errors.
#
# The hope is that this will yield a convergent iterative alternative
# to sparse bundle adjustment.

# older info ... 
# 1. Iterate through all the image pairs and triangulate the match points.
# 2. Set the 3d location of features to triangulated position (possibly
#    averaged if the feature is included in multiple matches
# 3. Compute new camera poses with solvePnP() using triangulated point locations
# 4. Repeat

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
proj.load_match_pairs()
proj.undistort_keypoints()

m = Matcher.Matcher()

f = open(args.project + "/Matches.json", 'r')
matches_dict = json.load(f)
f.close()

d2r = math.pi / 180.0

# iterate through the project image list and triangulate the 3d
# location for all feature points, given the current camera pose.
# Returns a new matches_dict with updated point positions (averaging
# point locations when they show up in multiple matches.)
def triangulate(cam_dict):
    IK = np.linalg.inv( proj.cam.get_K() )

    # parallel to match_dict to accumulate point coordinate sums
    sum_dict = {}

    for i, i1 in enumerate(proj.image_list):
        #rvec1, tvec1 = i1.get_proj()
        rvec1 = cam_dict[i1.name]['rvec']
        tvec1 = cam_dict[i1.name]['tvec']
        R1, jac = cv2.Rodrigues(rvec1)
        PROJ1 = np.concatenate((R1, tvec1), axis=1)
        for j, i2 in enumerate(proj.image_list):
            matches = i1.match_list[j]
            if (j <= i) or (len(matches) == 0):
                continue
            #rvec2, tvec2 = i2.get_proj()
            rvec2 = cam_dict[i2.name]['rvec']
            tvec2 = cam_dict[i2.name]['tvec']
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

            points /= points[3]
            #print "points:\n", points[0:3].T

            print "%s vs %s" % (i1.name, i2.name)
            for k, p in enumerate(points[0:3].T):
                match = matches[k]
                key = "%d-%d" % (i, match[0])
                print key
                if key in sum_dict:
                    sum_dict[key] += p
                else:
                    sum_dict[key] = p
                pnew = p.tolist()
                print "new=", pnew
                print "1st guess=", matches_dict[key]['ned']
                #surface1.append( [pnew[1], pnew[0], -pnew[2]] )

    # divide each element of sum_dict by the count of pairs to get the
    # average point location (so sum_dict becomes a location_dict)
    for key in matches_dict:
        count = len(matches_dict[key]['pts']) - 1
        sum_dict[key] /= count

    # return the new dictionary.
    return sum_dict


# Iterate through the project image list and run solvePnP on each
# image's feature set to derive new estimated camera locations.  This
# runs solvepnp for each image on all features associated with that
# image so the result is somewhat of an average camera pose for all
# matches.  This helps move the camera pose location so that scale is
# improved, but doesn't help fix orienation errors in camera poses.
cam1 = []
def solvePnP1( cam_dict_orig, cam_dict, pts_dict ):
    print "solvePnP1()"
    # start with a clean slate
    cam_dict_new = {}
    for image in proj.image_list:
        image.img_pts = []
        image.obj_pts = []

    # iterate through the match dictionary and build a per image list of
    # obj_pts and img_pts
    for key in matches_dict:
        feature_dict = matches_dict[key]
        points = feature_dict['pts']
        ned = pts_dict[key] # from separate provided point positions
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
        scale = float(image.width) / float(camw)
        K = proj.cam.get_K(scale)
        # initial guess for the cv2.solvePnP() algorithm, as well as
        # it will overwrite these with the answer, so we need to be
        # careful to create these as a copy of the original, not a
        # pointer to the original.
        rvec = np.copy(cam_dict[image.name]['rvec'])
        tvec = np.copy(cam_dict[image.name]['tvec'])
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

        # Our Rned2cam matrix (in our ned coordinate system) is
        # body2cam * Rned, so solvePnP returns this combination.  We
        # can extract Rned by premultiplying by cam2body aka
        # inv(body2cam).
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
        cam_dict_new[image.name] = {}
        cam_dict_new[image.name]['rvec'] = rvec
        cam_dict_new[image.name]['tvec'] = tvec
        cam_dict_new[image.name]['ned'] = newned
    return cam_dict_new

# return a quat (of the inv(R) matrix) for the given rvec
def rvec2quat(rvec, cam2body):
    Rned2cam, jac = cv2.Rodrigues(rvec)
    Rned2body = cam2body.dot(Rned2cam)
    Rbody2ned = np.matrix(Rned2body).T
    # make 3x3 rotation matrix into 4x4 homogeneous matrix
    hIR = np.concatenate( (np.concatenate( (Rbody2ned, np.zeros((3,1))),1),
                           np.mat([0,0,0,1])) )
    quat = transformations.quaternion_from_matrix(hIR)
    return quat
    
# Iterate through the project image list pairs and run solvePnP on
# each image pair individually to get their relative camera
# orientations.  Then use this information to correct the camera's
# pose.  This alone doesn't account for scaling which is why there is
# the earlier varient to help correct scaling.
cam1 = []
def solvePnP2( cam_dict_orig, cam_dict, pts_dict ):
    print "solvePnP2()"
    cam_dict_new = {}
    camw, camh = proj.cam.get_image_params()
    for i, i1 in enumerate(proj.image_list):
        print i1.name
        print "orig pose (at start of loop):\n", cam_dict[i1.name]
        scale = float(i1.width) / float(camw)
        K = proj.cam.get_K(scale)
        cam2body = i1.get_cam2body()
        body2cam = i1.get_body2cam()
        # include our own position in the average
        quat_start_weight = 100
        ned_start_weight = 50000
        count = 0
        rvec1_start = np.copy(cam_dict[i1.name]['rvec'])
        print "start_quat =", rvec2quat(rvec1_start, cam2body)
        sum_quat = rvec2quat(rvec1_start, cam2body) * quat_start_weight
        sum_ned = np.array(cam_dict[i1.name]['ned']) * ned_start_weight
        for j, i2 in enumerate(proj.image_list):
            matches = i1.match_list[j]
            if len(matches) < 8:
                continue
            count += 1
            # solvePnP() overwrites the guesses with the answer, so
            # let's make fresh copies each iteration (instead of just
            # copying a pointer to the original.)
            rvec1_guess = np.copy(cam_dict[i1.name]['rvec'])
            tvec1_guess = np.copy(cam_dict[i1.name]['tvec'])
            rvec2_guess = np.copy(cam_dict[i2.name]['rvec'])
            tvec2_guess = np.copy(cam_dict[i2.name]['tvec'])
            # build obj_pts and img_pts to position i1 relative to the
            # matches in i2.
            img1_pts = []
            img2_pts = []
            obj_pts = []
            for pair in matches:
                kp1 = i1.kp_list[ pair[0] ]
                img1_pts.append( kp1.pt )
                kp2 = i2.kp_list[ pair[1] ]
                img2_pts.append( kp2.pt )
                key = "%d-%d" % (i, pair[0])
                if not key in pts_dict:
                    key = "%d-%d" % (j, pair[1])
                # print key, pts_dict[key]
                obj_pts.append(pts_dict[key])

            # compute the centroid of obj_pts
            sum = np.zeros(3)
            for p in obj_pts:
                sum += p
            obj_center = sum / len(obj_pts)
            print "obj_pts center =", obj_center
            
            print "orig pose (before solvepnp):\n", cam_dict[i1.name]
        
            # given the previously computed triangulations (and
            # averages of point 3d locations if they are involved in
            # multiple triangulations), then compute an estimate for
            # both matching camera poses.  The relative positioning of
            # these camera poses should be pretty accurate.
            (result, rvec1, tvec1) = cv2.solvePnP(np.float32(obj_pts),
                                                  np.float32(img1_pts),
                                                  K, None,
                                                  rvec1_guess, tvec1_guess,
                                                  useExtrinsicGuess=True)
            print "orig pose (between solvepnp):\n", cam_dict[i1.name]
            (result, rvec2, tvec2) = cv2.solvePnP(np.float32(obj_pts),
                                                  np.float32(img2_pts),
                                                  K, None,
                                                  rvec2_guess, tvec2_guess,
                                                  useExtrinsicGuess=True)
            print "orig pose (after solvepnp):\n", cam_dict[i1.name]
            
            Rned2cam1, jac = cv2.Rodrigues(rvec1)
            Rned2cam2, jac = cv2.Rodrigues(rvec2)
            ned1 = -np.matrix(Rned2cam1[:3,:3]).T * np.matrix(tvec1)
            ned2 = -np.matrix(Rned2cam2[:3,:3]).T * np.matrix(tvec2)
            print "cam1 start=%s new=%s" % (cam_dict[i1.name]['ned'], ned1)
            print "cam2 start=%s new=%s" % (cam_dict[i2.name]['ned'], ned2)

            # compute a rigid transform (rotation + translation) to
            # align the estimated camera locations (projected from the
            # triangulation) back with the original camera points, and
            # roughly rotated around the centroid of the object
            # points.
            src = np.zeros( (3,3) ) # current camera locations
            dst = np.zeros( (3,3) ) # original camera locations
            src[0,:] = np.squeeze(ned1)
            src[1,:] = np.squeeze(ned2)
            src[2,:] = obj_center
            dst[0,:] = cam_dict[i1.name]['ned']
            dst[1,:] = cam_dict[i2.name]['ned']
            dst[2,:] = obj_center
            print "src:\n", src
            print "dst:\n", dst
            M = transformations.superimposition_matrix(src, dst)
            print "M:\n", M

            # Our (i1) Rned2cam matrix is body2cam * Rned, so solvePnP
            # returns this combination.  We can extract Rbody2ned by
            # premultiplying by cam2body aka inv(body2cam).
            Rned2body = cam2body.dot(Rned2cam1)
            # print "Rned2body:\n", Rned2body
            Rbody2ned = np.matrix(Rned2body).T # IR
            # print "Rbody2ned:\n", Rbody2ned
            # print "R (after M * R):\n", R

            # now transform by the earlier "M" affine transform to
            # rotate the work space closer to the actual camera points
            Rot = M[:3,:3].dot(Rbody2ned)
            
            # make 3x3 rotation matrix into 4x4 homogeneous matrix
            hRot = np.concatenate( (np.concatenate( (Rot, np.zeros((3,1))),1),
                                    np.mat([0,0,0,1])) )
            # print "hRbody2ned:\n", hRbody2ned

            quat = transformations.quaternion_from_matrix(hRot)
            sum_quat += quat
            
            sum_ned += np.squeeze(np.asarray(ned1))

        print "count = ", count
        newned = sum_ned / (ned_start_weight + count)
        print "orig ned =", i1.camera_pose['ned']
        print "new ned =", newned
        print "sum_quat=%s total=%s" % (sum_quat, quat_start_weight + count)
        newquat = sum_quat / (quat_start_weight + count)
        print "new quat =", newquat
        newIR = transformations.quaternion_matrix(newquat)
        print "flight data ypr = ", i1.camera_pose['ypr']
        (yaw, pitch, roll) = transformations.euler_from_quaternion(newquat, 'rzyx')
        print "new ypr =", [yaw/d2r, pitch/d2r, roll/d2r]

        newR = np.transpose(newIR[:3,:3]) # equivalent to inverting
        newRned2cam = body2cam.dot(newR[:3,:3])
        rvec, jac = cv2.Rodrigues(newRned2cam)
        #new_tvec = -np.matrix(newRned2cam) * np.matrix(newned).T

        print "orig pose (at end of loop):\n", cam_dict[i1.name]
        cam_dict_new[i1.name] = {}
        cam_dict_new[i1.name]['rvec'] = rvec
        # compute a tvec relative to the starting ned (we are just
        # adjusting orientation in this routine)
        ned_orig = cam_dict[i1.name]['ned']
        
        # move the camera ned locations
        tvec = -np.matrix(newRned2cam) * np.matrix(newned).T
        cam_dict_new[i1.name]['ned'] = newned
        # or don't move the camera ned locations
        #### tvec = -np.matrix(newRned2cam) * np.matrix(ned_orig).T
        #### cam_dict_new[i1.name]['ned'] = cam_dict[i1.name]['ned']
        
        cam_dict_new[i1.name]['tvec'] = tvec
        print "new pose:\n", cam_dict_new[i1.name]
    return cam_dict_new

# Iterate through the project image list pairs and run solvePnP on
# each image pair individually to get the estimated pose (from each
# match pair) for each camera.  The average the poses for each camera
# to come up with an average pose.
cam1 = []
def solvePnP3( cam_dict_orig, cam_dict, pts_dict ):
    print "solvePnP3()"
    cam_dict_new = {}
    camw, camh = proj.cam.get_image_params()
    for i, i1 in enumerate(proj.image_list):
        print i1.name
        print "orig pose (at start of loop):\n", cam_dict[i1.name]
        scale = float(i1.width) / float(camw)
        K = proj.cam.get_K(scale)
        cam2body = i1.get_cam2body()
        body2cam = i1.get_body2cam()
        # include our own position in the average
        quat_start_weight = 50
        count = 0
        rvec1_start = np.copy(cam_dict[i1.name]['rvec'])
        print "start_quat =", rvec2quat(rvec1_start, cam2body)
        sum_quat = rvec2quat(rvec1_start, cam2body) * quat_start_weight
        for j, i2 in enumerate(proj.image_list):
            matches = i1.match_list[j]
            if len(matches) < 8:
                continue
            count += 1
            # solvePnP() overwrites the guesses with the answer, so
            # let's make fresh copies each iteration (instead of just
            # copying a pointer to the original.)
            rvec1_guess = np.copy(cam_dict[i1.name]['rvec'])
            tvec1_guess = np.copy(cam_dict[i1.name]['tvec'])
            # build obj_pts and img_pts to position i1 relative to the
            # matches in i2.
            img1_pts = []
            obj_pts = []
            for pair in matches:
                kp1 = i1.kp_list[ pair[0] ]
                img1_pts.append( kp1.pt )
                key = "%d-%d" % (i, pair[0])
                if not key in pts_dict:
                    key = "%d-%d" % (j, pair[1])
                # print key, pts_dict[key]
                obj_pts.append(pts_dict[key])
            
            # given the previously computed triangulations (and
            # averages of point 3d locations if they are involved in
            # multiple triangulations), then compute an estimate for
            # both matching camera poses.  The relative positioning of
            # these camera poses should be pretty accurate.
            (result, rvec1, tvec1) = cv2.solvePnP(np.float32(obj_pts),
                                                  np.float32(img1_pts),
                                                  K, None,
                                                  rvec1_guess, tvec1_guess,
                                                  useExtrinsicGuess=True)
            quat = rvec2quat(rvec1, cam2body)
            sum_quat += quat

        print "count = ", count
        print "orig ned =", i1.camera_pose['ned']
        print "sum_quat=%s total=%s" % (sum_quat, quat_start_weight + count)
        newquat = sum_quat / (quat_start_weight + count)
        print "new quat =", newquat
        newIR = transformations.quaternion_matrix(newquat)
        print "flight data ypr = ", i1.camera_pose['ypr']
        (yaw, pitch, roll) = transformations.euler_from_quaternion(newquat, 'rzyx')
        print "new ypr =", [yaw/d2r, pitch/d2r, roll/d2r]

        newR = np.transpose(newIR[:3,:3]) # equivalent to inverting
        newRned2cam = body2cam.dot(newR[:3,:3])
        rvec, jac = cv2.Rodrigues(newRned2cam)
        #new_tvec = -np.matrix(newRned2cam) * np.matrix(newned).T

        print "orig pose (at end of loop):\n", cam_dict[i1.name]
        cam_dict_new[i1.name] = {}
        cam_dict_new[i1.name]['rvec'] = rvec
        # compute a tvec relative to the starting ned (we are just
        # adjusting orientation in this routine)
        ned_orig = cam_dict[i1.name]['ned']
        
        # move the camera ned locations
        tvec = -np.matrix(newRned2cam) * np.matrix(ned_orig).T
        cam_dict_new[i1.name]['ned'] = ned_orig
        # or don't move the camera ned locations
        #### tvec = -np.matrix(newRned2cam) * np.matrix(ned_orig).T
        #### cam_dict_new[i1.name]['ned'] = cam_dict[i1.name]['ned']
        
        cam_dict_new[i1.name]['tvec'] = tvec
        print "new pose:\n", cam_dict_new[i1.name]
    return cam_dict_new


# compute a 3d affine tranformation between current set of refined
# camera locations and original camera locations, then transform the
# current cameras back.  This way the retain their new relative
# positioning without walking away from each other in scale or
# position.
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
        if False:
            # adjust the camera projection matrix (rvec) to rotate by the
            # amount of the affine transformation as well (this should now
            # be better accounted for in solvePnP2()
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
for key in matches_dict:
    feature_dict = matches_dict[key]
    ned = feature_dict['ned']
    surface0.append( [ned[1], ned[0], -ned[2]] )

cam0 = []
for image in proj.image_list:
    ned, ypr, quat = image.get_camera_pose()
    cam0.append( [ned[1], ned[0], -ned[2]] )

# iterate through the image list and build:
# (1) an 'original' camera pose dictionary for later reference
# (2) a working camera pose dictionary for incrementally refining
# (3) a simple list of camera locations for plotting
cam_dict_orig = {}
cam_dict = {}
for image in proj.image_list:
    ned, ypr, quat = image.get_camera_pose()
    rvec, tvec = image.get_proj()
    cam_dict_orig[image.name] = {}
    cam_dict_orig[image.name]['rvec'] = rvec
    cam_dict_orig[image.name]['tvec'] = tvec
    cam_dict_orig[image.name]['ned'] = ned
    cam_dict[image.name] = {}
    cam_dict[image.name]['rvec'] = rvec
    cam_dict[image.name]['tvec'] = tvec
    cam_dict[image.name]['ned'] = ned

count = 0
while True:
    # run the triangulation step
    newpts_dict = triangulate(cam_dict)
    #print pts_dict

    surface1 = []
    for key in newpts_dict:
        p = newpts_dict[key]
        surface1.append( [ p[1], p[0], -p[2] ] )

    # improve camera pose position
    cam_dict = solvePnP1(cam_dict_orig, cam_dict, newpts_dict)

    # improve camera pose orientation
    # cam_dict = solvePnP3(cam_dict_orig, cam_dict, newpts_dict)
    
    # get the affine transformation required to bring the new camera
    # locations back into a best fit with the original camera
    # locations
    #### A = get_recenter_affine(cam_dict)

    # transform all the feature points by the affine matrix
    #### newpts_dict = transform_points(A, newpts_dict)
    
    # run solvePnP now on the updated points (hopefully this will
    # naturally reorient the cameras as needed.)
    #### cam_dict = solvePnP3(cam_dict_orig, cam_dict, newpts_dict)
    
    cam1 = []
    for key in cam_dict:
        p = cam_dict[key]['ned']
        cam1.append( [ p[1], p[0], -p[2] ] )

    if count % 10 == 0:
        plot(surface0, cam0, surface1, cam1)

    count += 1
