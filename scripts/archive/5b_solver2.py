#!/usr/bin/python

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

# iterate through the project image list pairs and triangulate the 3d
# location for all feature points for each matching pair, given the
# current camera poses.  Returns a new matches_dict with update point
# positions
def triangulate1(cam_dict):
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
                #print key
                if key in sum_dict:
                    sum_dict[key] += p
                else:
                    sum_dict[key] = p
                #if not key in matches_dict:
                #    key = "%d-%d" % (j, match[1])
                #    print key
                pnew = p.tolist()
                #print "new=", pnew
                #print "1st guess=", matches_dict[key]['ned']
                #surface1.append( [pnew[1], pnew[0], -pnew[2]] )

    # divide each element of sum_dict by the count of pairs to get the
    # average point location (so sum_dict becomes a location_dict)
    for key in matches_dict:
        count = len(matches_dict[key]['pts']) - 1
        sum_dict[key] /= count

    # return the new dictionary.
    return sum_dict



# iterate through the project image list pairs and triangulate the 3d
# location for all feature points for each matching pair, given the
# current camera poses.  Returns a new matches_dict with update point
# positions
def triangulate2(cam_dict):
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
                #print key
                if key in sum_dict:
                    sum_dict[key] += p
                else:
                    sum_dict[key] = p
                #if not key in matches_dict:
                #    key = "%d-%d" % (j, match[1])
                #    print key
                pnew = p.tolist()
                #print "new=", pnew
                #print "1st guess=", matches_dict[key]['ned']
                #surface1.append( [pnew[1], pnew[0], -pnew[2]] )

    # divide each element of sum_dict by the count of pairs to get the
    # average point location (so sum_dict becomes a location_dict)
    for key in matches_dict:
        count = len(matches_dict[key]['pts']) - 1
        sum_dict[key] /= count

    # return the new dictionary.
    return sum_dict


# Iterate through the project image list and run solvePnP on each
# image's feature set to derive new estimated camera locations
cam1 = []
def solvePnP1( pts_dict ):
    # start with a clean slate
    for i1 in proj.image_list:
        i1.img_pts = []
        i1.obj_pts = []

    # build a new cam_dict that is a copy of the current one
    cam_dict = {}
    for i1 in proj.image_list:
        cam_dict[i1.name] = {}
        rvec, tvec = i1.get_proj()
        ned, ypr, quat = i1.get_camera_pose()
        cam_dict[i1.name]['rvec'] = rvec
        cam_dict[i1.name]['tvec'] = tvec
        cam_dict[i1.name]['ned'] = ned

    camw, camh = proj.cam.get_image_params()
    for i, i1 in enumerate(proj.image_list):
        print i1.name
        scale = float(i1.width) / float(camw)
        K = proj.cam.get_K(scale)
        rvec, tvec = i1.get_proj()
        cam2body = i1.get_cam2body()
        body2cam = i1.get_body2cam()
        # include our own position in the average
        count = 1
        sum_quat = np.array(i1.camera_pose['quat'])
        sum_ned = np.array(i1.camera_pose['ned'])
        for j, i2 in enumerate(proj.image_list):
            matches = i1.match_list[j]
            if len(matches) < 8:
                continue
            count += 1
            # build obj_pts and img_pts to position i1 relative to the
            # matches in i2
            obj_pts = []
            img_pts = []
            for pair in matches:
                kp = i1.kp_list[ pair[0] ]
                img_pts.append( kp.pt )
                key = "%d-%d" % (i, pair[0])
                if not key in pts_dict:
                    key = "%d-%d" % (j, pair[1])
                # print key, pts_dict[key]
                obj_pts.append(pts_dict[key])
                #if key in pts_dict:
                #    sum_dict[key] += p
                #else:
                #    sum_dict[key] = p

            (result, rvec, tvec) = cv2.solvePnP(np.float32(obj_pts),
                                                np.float32(img_pts),
                                                K, None, rvec, tvec,
                                                useExtrinsicGuess=True)
            #print "rvec=%s tvec=%s" % (rvec, tvec)
            Rned2cam, jac = cv2.Rodrigues(rvec)
            #print "Rned2cam (from SolvePNP):\n", Rned2cam

            # Our Rcam matrix (in our ned coordinate system) is
            # body2cam * Rned, so solvePnP returns this combination.
            # We can extract Rned by premultiplying by cam2body aka
            # inv(body2cam).
            Rned2body = cam2body.dot(Rned2cam)
            # print "Rned2body:\n", Rned2body
            Rbody2ned = np.matrix(Rned2body).T # IR
            # print "Rbody2ned:\n", Rbody2ned
            # print "R (after M * R):\n", R

            # make 3x3 rotation matrix into 4x4 homogeneous matrix
            hRbody2ned = np.concatenate( (np.concatenate( (Rbody2ned, np.zeros((3,1))),1), np.mat([0,0,0,1])) )
            # print "hRbody2ned:\n", hRbody2ned

            quat = transformations.quaternion_from_matrix(hRbody2ned)
            sum_quat += quat
            
            pos = -np.matrix(Rned2cam[:3,:3]).T * np.matrix(tvec)
            sum_ned += np.squeeze(np.asarray(pos))
            print "ned =", np.squeeze(np.asarray(pos))

        print "count = ", count
        newned = sum_ned / float(count)
        print "orig ned =", i1.camera_pose['ned']
        print "new ned =", newned
        newquat = sum_quat / float(count)
        newIR = transformations.quaternion_matrix(newquat)
        print "orig ypr = ", i1.camera_pose['ypr']
        (yaw, pitch, roll) = transformations.euler_from_quaternion(newquat, 'rzyx')
        print "new ypr =", [yaw/d2r, pitch/d2r, roll/d2r]

        newR = np.transpose(newIR[:3,:3]) # equivalent to inverting
        newRned2cam = body2cam.dot(newR[:3,:3])
        rvec, jac = cv2.Rodrigues(newRned2cam)
        tvec = -np.matrix(newRned2cam) * np.matrix(newned).T

        #print "orig pose:\n", cam_dict[i1.name]
        cam_dict[i1.name]['rvec'] = rvec
        cam_dict[i1.name]['tvec'] = tvec
        cam_dict[i1.name]['ned'] = newned
        #print "new pose:\n", cam_dict[i1.name]
    return cam_dict

# Iterate through the project image list and run solvePnP on each
# image's feature set to derive new estimated camera locations
cam1 = []
def solvePnP2( pts_dict ):
    # build a new cam_dict that is a copy of the current one
    cam_dict = {}
    for i1 in proj.image_list:
        cam_dict[i1.name] = {}
        rvec, tvec = i1.get_proj()
        ned, ypr, quat = i1.get_camera_pose()
        cam_dict[i1.name]['rvec'] = rvec
        cam_dict[i1.name]['tvec'] = tvec
        cam_dict[i1.name]['ned'] = ned

    camw, camh = proj.cam.get_image_params()
    for i, i1 in enumerate(proj.image_list):
        print i1.name
        scale = float(i1.width) / float(camw)
        K = proj.cam.get_K(scale)
        rvec1, tvec1 = i1.get_proj()
        cam2body = i1.get_cam2body()
        body2cam = i1.get_body2cam()
        # include our own position in the average
        quat_start_weight = 100
        ned_start_weight = 2
        count = 0
        sum_quat = np.array(i1.camera_pose['quat']) * quat_start_weight
        sum_ned = np.array(i1.camera_pose['ned']) * ned_start_weight
        for j, i2 in enumerate(proj.image_list):
            matches = i1.match_list[j]
            if len(matches) < 8:
                continue
            count += 1
            rvec2, tvec2 = i2.get_proj()
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
            
            # given the previously computed triangulations (and
            # averages of point 3d locations if they are involved in
            # multiple triangulations), then compute an estimate for
            # both matching camera poses.  The relative positioning of
            # these camera poses should be pretty accurate.
            (result, rvec1, tvec1) = cv2.solvePnP(np.float32(obj_pts),
                                                  np.float32(img1_pts),
                                                  K, None, rvec1, tvec1,
                                                  useExtrinsicGuess=True)
            (result, rvec2, tvec2) = cv2.solvePnP(np.float32(obj_pts),
                                                  np.float32(img2_pts),
                                                  K, None, rvec2, tvec2,
                                                  useExtrinsicGuess=True)
            
            Rned2cam1, jac = cv2.Rodrigues(rvec1)
            Rned2cam2, jac = cv2.Rodrigues(rvec2)
            ned1 = -np.matrix(Rned2cam1[:3,:3]).T * np.matrix(tvec1)
            ned2 = -np.matrix(Rned2cam2[:3,:3]).T * np.matrix(tvec2)
            print "cam1 orig=%s new=%s" % (i1.camera_pose['ned'], ned1)
            print "cam2 orig=%s new=%s" % (i2.camera_pose['ned'], ned2)

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
            dst[0,:] = i1.camera_pose['ned']
            dst[1,:] = i2.camera_pose['ned']
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
        newquat = sum_quat / (quat_start_weight + count)
        newIR = transformations.quaternion_matrix(newquat)
        print "orig ypr = ", i1.camera_pose['ypr']
        (yaw, pitch, roll) = transformations.euler_from_quaternion(newquat, 'rzyx')
        print "new ypr =", [yaw/d2r, pitch/d2r, roll/d2r]

        newR = np.transpose(newIR[:3,:3]) # equivalent to inverting
        newRned2cam = body2cam.dot(newR[:3,:3])
        rvec, jac = cv2.Rodrigues(newRned2cam)
        tvec = -np.matrix(newRned2cam) * np.matrix(newned).T

        #print "orig pose:\n", cam_dict[i1.name]
        cam_dict[i1.name]['rvec'] = rvec
        cam_dict[i1.name]['tvec'] = tvec
        cam_dict[i1.name]['ned'] = newned
        #print "new pose:\n", cam_dict[i1.name]
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
for key in matches_dict:
    feature_dict = matches_dict[key]
    ned = feature_dict['ned']
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
    newpts_dict = triangulate1(cam_dict)
    #print pts_dict

    surface1 = []
    for key in newpts_dict:
        p = newpts_dict[key]
        surface1.append( [ p[1], p[0], -p[2] ] )

    # find the optimal camera poses for the triangulation averaged
    # together.
    cam_dict = solvePnP2(newpts_dict)

    # get the affine transformation required to bring the new camera
    # locations back into a best fit with the original camera
    # locations
    ##### A = get_recenter_affine(cam_dict)

    # transform all the feature points by the affine matrix
    ##### newspts_dict = transform_points(A, newpts_dict)
    
    # run solvePnP now on the updated points (hopefully this will
    # naturally reorient the cameras as needed.)
    ###### cam_dict = solvePnP(newpts_dict)
    
    cam1 = []
    for key in cam_dict:
        p = cam_dict[key]['ned']
        cam1.append( [ p[1], p[0], -p[2] ] )

    if count % 1 == 0:
        plot(surface0, cam0, surface1, cam1)

    count += 1
