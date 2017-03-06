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
parser.add_argument('--strategy', default='my_triangulate',
                    choices=['my_triangulate', 'triangulate', 'dem'], help='projection strategy')
parser.add_argument('--iterations', type=int, help='stop after this many solver iterations')
parser.add_argument('--target-mre', type=float, help='stop when mre meets this threshold')
parser.add_argument('--plot', action='store_true', help='plot the solution state')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )
print "unique features:", len(matches_direct)

# compute keypoint usage map
proj.compute_kp_usage_new(matches_direct)

# setup SRTM ground interpolator
ref = proj.ned_reference_lla
sss = SRTM.NEDGround( ref, 2000, 2000, 30 )

start_mre = -1.0

# iterate through the matches list and triangulate the 3d location for
# all feature points, given the associated camera poses.  Returns a
# new matches_dict with update point positions
import LineSolver

def my_triangulate(matches_direct, cam_dict):
    IK = np.linalg.inv( proj.cam.get_K() )

    for match in matches_direct:
        #print match
        points = []
        vectors = []
        for m in match[1:]:
            image = proj.image_list[m[0]]
            cam2body = image.get_cam2body()
            body2ned = image.rvec_to_body2ned(cam_dict[image.name]['rvec'])
            uv_list = [ image.uv_list[m[1]] ] # just one uv element
            vec_list = proj.projectVectors(IK, body2ned, cam2body, uv_list)
            points.append( cam_dict[image.name]['ned'] )
            vectors.append( vec_list[0] )
            #print ' ', image.name
            #print ' ', uv_list
            #print '  ', vec_list
        p = LineSolver.ls_lines_intersection(points, vectors, transpose=True).tolist()
        #print p, p[0]
        match[0] = [ p[0][0], p[1][0], p[2][0] ]

# iterate through the project image list and triangulate the 3d
# location for all feature points, given the current camera pose.
# Returns a new matches_dict with update point positions
def triangulate(matches_direct, cam_dict):
    IK = np.linalg.inv( proj.cam.get_K() )

    match_pairs = proj.generate_match_pairs(matches_direct)

    # zero the match NED coordinate and initialize the corresponding
    # count array
    counters = []
    for match in matches_direct:
        match[0] = np.array( [0.0, 0.0, 0.0] )
        counters.append( 0)
        
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
            # distance between two cameras
            ned1 = np.array(cam_dict[i1.name]['ned'])
            ned2 = np.array(cam_dict[i2.name]['ned'])
            dist = np.linalg.norm(ned2 - ned1)
            if dist < 40:
                # idea: the closer together two poses are, the greater
                # the triangulation error will be relative to small
                # attitude errors.  If we only compare more distance
                # camera views the solver will be more stable.
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
                counters[indices[k]] += 1

    # divide each NED coordinate (sum of triangulated point locations)
    # of matches_direct_dict by the count of references to produce an
    # average NED coordinate for each match.
    for i, match in enumerate(matches_direct):
        if counters[i] > 0:
            match[0] /= counters[i]
        else:
            print 'invalid match from images too close to each other:', match
            for j in range(1, len(match)):
                match[j] = [-1, -1]

    # return the new match structure
    return matches_direct


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
        # print image.name
        if len(image.img_pts) < 4:
            continue
        scale = float(image.width) / float(camw)
        K = proj.cam.get_K(scale)
        rvec, tvec = image.get_proj()
        (result, rvec, tvec) \
           = cv2.solvePnP(np.float32(image.obj_pts),
                                     np.float32(image.img_pts),
                                     K, None,
                                     rvec, tvec, useExtrinsicGuess=True)
        # The idea of using the Ransac version of solvePnP() is to
        # look past outliers instead of being affected by them.  We
        # don't use the outlier information at this point in the
        # process for outlier rejection.  However, it appears that
        # this process leads to divergence, not convergence.
        # (rvec, tvec, inliers) \
        #     = cv2.solvePnPRansac(np.float32(image.obj_pts),
        #                          np.float32(image.img_pts),
        #                          K, None,
        #                          rvec, tvec, useExtrinsicGuess=True)

        #print "rvec=", rvec
        #print "tvec=", tvec
        Rned2cam, jac = cv2.Rodrigues(rvec)
        #print "Rraw (from SolvePNP):\n", Rraw

        ned = image.camera_pose['ned']
        #print "original ned = ", ned
        #tvec = -np.matrix(R[:3,:3]) * np.matrix(ned).T
        #print "tvec =", tvec
        pos = -np.matrix(Rned2cam[:3,:3]).T * np.matrix(tvec)
        newned = pos.T[0].tolist()[0]
        #print "new ned =", newned

        # Our Rcam matrix (in our ned coordinate system) is body2cam * Rned,
        # so solvePnP returns this combination.  We can extract Rned by
        # premultiplying by cam2body aka inv(body2cam).
        cam2body = image.get_cam2body()
        Rned2body = cam2body.dot(Rned2cam)
        #print "R (after M * R):\n", R

        ypr = image.camera_pose['ypr']
        #print "original ypr = ", ypr
        Rbody2ned = np.matrix(Rned2body).T
        IRo = transformations.euler_matrix(ypr[0]*d2r, ypr[1]*d2r, ypr[2]*d2r, 'rzyx')
        IRq = transformations.quaternion_matrix(image.camera_pose['quat'])
        #print "Original IR:\n", IRo
        #print "Original IR (from quat)\n", IRq
        #print "IR (from SolvePNP):\n", IR

        (yaw, pitch, roll) = transformations.euler_from_matrix(Rbody2ned, 'rzyx')
        #print "ypr =", [yaw/d2r, pitch/d2r, roll/d2r]

        #image.set_camera_pose( pos.T[0].tolist(), [yaw/d2r, pitch/d2r, roll/d2r] )
        #print "Proj =", np.concatenate((R, tvec), axis=1)
        cam_dict[image.name] = {}
        cam_dict[image.name]['rvec'] = rvec
        cam_dict[image.name]['tvec'] = tvec
        cam_dict[image.name]['ned'] = newned
    return cam_dict

# return a 3d affine tranformation between fitted camera locations and
# original camera locations.
def get_recenter_affine(cam_dict):
    src = [[], [], [], []]      # current camera locations
    dst = [[], [], [], []]      # original camera locations
    for image in proj.image_list:
        if image.feature_count > 0:
            newned = cam_dict[image.name]['ned']
            src[0].append(newned[0])
            src[1].append(newned[1])
            src[2].append(newned[2])
            src[3].append(1.0)
            origned, ypr, quat = image.get_camera_pose()
            dst[0].append(origned[0])
            dst[1].append(origned[1])
            dst[2].append(origned[2])
            dst[3].append(1.0)
            #print image.name, '%s -> %s' % (origned, newned)
    A = transformations.superimposition_matrix(src, dst, scale=True)
    print "Affine 3D:\n", A
    return A

# transform the camera ned positions with the provided affine matrix
# to keep all the camera poses best fitted to the original camera
# locations.  Also rotate the camera poses by the rotational portion
# of the affine matrix to update the camera alignment.
def transform_cams(A, cam_dict):
    # construct an array of camera positions
    src = [[], [], [], []]
    for image in proj.image_list:
        new = cam_dict[image.name]['ned']
        src[0].append(new[0])
        src[1].append(new[1])
        src[2].append(new[2])
        src[3].append(1.0)
        
    # extract the rotational portion of the affine matrix
    scale, shear, angles, trans, persp = transformations.decompose_matrix(A)
    R = transformations.euler_matrix(*angles)
    #print "R:\n", R
    
    # full transform the camera ned positions to best align with
    # original locations
    update_cams = A.dot( np.array(src) )
    #print update_cams[:3]
    for i, p in enumerate(update_cams.T):
        key = proj.image_list[i].name
        if not key in cam_dict:
            cam_dict[key] = {}
        ned = [ p[0], p[1], p[2] ]
        # print "ned:", ned
        cam_dict[key]['ned'] = ned
        # adjust the camera projection matrix (rvec) to rotate by the
        # amount of the affine transformation as well
        rvec = cam_dict[key]['rvec']
        tvec = cam_dict[key]['tvec']
        Rcam, jac = cv2.Rodrigues(rvec)
        # print "Rcam:\n", Rcam
        Rcam_new = R[:3,:3].dot(Rcam)
        # print "Rcam_new:\n", Rcam_new
        
        rvec, jac = cv2.Rodrigues(Rcam_new)
        cam_dict[key]['rvec'] = rvec
        tvec = -np.matrix(Rcam_new) * np.matrix(ned).T
        cam_dict[key]['tvec'] = tvec

# transform all the match point locations
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

# mark items that exceed the cutoff reprojection error for deletion
def mark_outliers(result_list, cutoff, matches_direct):
    print " marking outliers..."
    mark_count = 0
    for line in result_list:
        # print "line:", line
        if line[0] > cutoff:
            print "  outlier index %d-%d err=%.2f" % (line[1], line[2],
                                                      line[0])
            #if args.show:
            #    draw_match(line[1], line[2])
            match = matches_direct[line[1]]
            match[line[2]+1] = [-1, -1]
            mark_count += 1

# mark matches not referencing images in the main group
def mark_non_group(main_group, matches_direct):
    # construct set of image indices in main_group
    group_dict = {}
    for image in main_group:
        for i, i1 in enumerate(proj.image_list):
            if image == i1:
                group_dict[i] = True
    #print 'group_dict:', group_dict
    
    print " marking non group..."
    mark_sum = 0
    for match in matches_direct:
        for j, p in enumerate(match[1:]):
            if not p[0] in group_dict:
                 match[j+1] = [-1, -1]
                 mark_sum += 1
    print 'marked:', mark_sum, 'matches for deletion'

# delete marked matches
def delete_marked_matches(matches_direct):
    print " deleting marked items..."
    for i in reversed(range(len(matches_direct))):
        match = matches_direct[i]
        has_bad_elem = False
        for j in reversed(range(1, len(match))):
            p = match[j]
            if p == [-1, -1]:
                has_bad_elem = True
                match.pop(j)
        if len(match) < 4:
            print "deleting match that is now in less than 3 images:", match
            matches_direct.pop(i)

# any image with less than 25 matches has all it's matches marked for
# deletion
def mark_weak_images(matches_direct):
    # count how many features show up in each image
    for i in proj.image_list:
        i.feature_count = 0
    for i, match in enumerate(matches_direct):
        for j, p in enumerate(match[1:]):
            if p[1] != [-1, -1]:
                image = proj.image_list[ p[0] ]
                image.feature_count += 1

    # make a dict of all images with less than 25 feature matches
    weak_dict = {}
    for i, img in enumerate(proj.image_list):
        if img.feature_count < 25:
            weak_dict[i] = True
            if img.feature_count > 0:
                print 'new weak image:', img.name
                img.feature_count = 0 # will be zero very soon
    print 'weak images:', weak_dict

    # mark any features in the weak images list
    mark_sum = 0
    for i, match in enumerate(matches_direct):
        #print 'before:', match
        for j, p in enumerate(match[1:]):
            if p[0] in weak_dict:
                 match[j+1] = [-1, -1]
                 mark_sum += 1
        #print 'after:', match
    
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

# temporary testing ....
# match_pairs = proj.generate_match_pairs(matches_direct)
# group_list = Matcher.groupByConnections(proj.image_list, matches_direct, match_pairs)
# mark_non_group(group_list[0], matches_direct)
# quit()

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
# cam_dict = {}
# for image in proj.image_list:
#     rvec, tvec, ned = image.get_proj()
#     cam_dict[image.name] = {}
#     cam_dict[image.name]['rvec'] = rvec
#     cam_dict[image.name]['tvec'] = tvec
#     cam_dict[image.name]['ned'] = ned

count = 0
while True:
    # find the 'best fit' camera poses for the triangulation averaged
    # together.
    cam_dict = solvePnP(matches_direct)

    # measure our current mean reprojection error and trim mre
    # outliers from the match set (any points with mre 4x stddev) as
    # well as any weak images with < 25 matches.
    (result_list, mre, stddev) \
        = proj.compute_reprojection_errors(cam_dict, matches_direct)
    if start_mre < 0.0: start_mre = mre
    print "mre = %.4f stddev = %.4f features = %d" % (mre, stddev, len(matches_direct))
    
    cull_outliers = False
    if cull_outliers:
        mark_outliers(result_list, mre + stddev*4, matches_direct)
        mark_weak_images(matches_direct)
        delete_marked_matches(matches_direct)

        # after outlier deletion, re-evalute matched pairs and connection
        # cycles.
        match_pairs = proj.generate_match_pairs(matches_direct)
        group_list = Matcher.groupByConnections(proj.image_list, matches_direct, match_pairs)
        mark_non_group(group_list[0], matches_direct)
        delete_marked_matches(matches_direct)
    else:
        # keep accounting structures happy
        mark_weak_images(matches_direct)
        
    # get the affine transformation required to bring the new camera
    # locations back inqto a best fit with the original camera
    # locations
    A = get_recenter_affine(cam_dict)

    # thought #1: if we are triangulating, this could be done once at the
    # end to fix up the solution, not every iteration?  But it doesn't
    # seem to harm the triangulation.

    # thought #2: if we are projecting onto the dem surface, we
    # probably shouldn't transform the cams back to the original
    # because this could perpetually pull things out of convergence
    transform_cams(A, cam_dict)

    if args.strategy == 'my_triangulate':
        # run the triangulation step (modifies NED coordinates in
        # place).  This computes a best fit for all the feature
        # locations based on the current best camera poses.
        my_triangulate(matches_direct, cam_dict)
    elif args.strategy == 'triangulate':
        # run the triangulation step (modifies NED coordinates in
        # place).  This computes a best fit for all the feature
        # locations based on the current best camera poses.
        triangulate(matches_direct, cam_dict)
    elif args.strategy == 'dem':
        # project the keypoints back onto the DEM surface from the
        # updated camera poses.
        proj.fastProjectKeypointsTo3d(sss, cam_dict)
        # estimate new world coordinates for each match point
        for match in matches_direct:
            sum = np.array( [0.0, 0.0, 0.0] )
            for p in match[1:]:
                sum += proj.image_list[ p[0] ].coord_list[ p[1] ]
            ned = sum / len(match[1:])
            # print "avg =", ned
            match[0] = ned.tolist()
    else:
        print 'unknown triangulation strategy, script will probably fail to do anything useful'

    surface1 = []
    for match in matches_direct:
        ned = match[0]
        print ned
        surface1.append( [ned[1], ned[0], -ned[2]] )


    # transform all the feature points by the affine matrix (modifies
    # matches_direct NED coordinates in place)
    # fixme: transform_points(A, matches_direct)

    # fixme: transform camera locations and orientations as well
    
    # run solvePnP now on the updated points (hopefully this will
    # naturally reorient the cameras as needed.)
    # 9/6/2016: shouldn't be needed since transform_points() now rotates
    # the camera orientation as well?
    # cam_dict = solvePnP(newpts_dict)
    
    cam1 = []
    for key in cam_dict:
        p = cam_dict[key]['ned']
        cam1.append( [ p[1], p[0], -p[2] ] )

    if args.plot:
        plot(surface0, cam0, surface1, cam1)

    count += 1

    # test stop conditions
    if args.iterations:
        if count >= args.iterations:
            print 'Stopping (by request) after', count, 'iterations.'
            break
    elif args.target_mre:
        if mre <= args.target_mre:
            print 'Stopping (by request) with mre:', mre
            break
    else:
        print 'No stop condition specified, running one iteration and stopping.'
        break
    
(result_list, mre, stddev) \
    = proj.compute_reprojection_errors(cam_dict, matches_direct)
 
print 'Start mre:', start_mre, 'end mre:', mre

result=raw_input('Update matches and camera poses? (y/n):')
if result == 'y' or result == 'Y':
    print 'Writing direct matches...'
    pickle.dump(matches_direct, open(args.project+"/matches_direct", "wb"))

    print 'Updating and saving camera poses...'
    for image in proj.image_list:
        pose = cam_dict[image.name]
        Rned2cam, jac = cv2.Rodrigues(pose['rvec'])
        pos = -np.matrix(Rned2cam[:3,:3]).T * np.matrix(pose['tvec'])
        ned = pos.T[0].tolist()[0]
        
        # Our Rcam matrix (in our ned coordinate system) is body2cam * Rned,
        # so solvePnP returns this combination.  We can extract Rned by
        # premultiplying by cam2body aka inv(body2cam).
        cam2body = image.get_cam2body()
        Rned2body = cam2body.dot(Rned2cam)
        Rbody2ned = np.matrix(Rned2body).T
        (yaw, pitch, roll) \
            = transformations.euler_from_matrix(Rbody2ned, 'rzyx')
        # print "ypr =", [yaw/d2r, pitch/d2r, roll/d2r]
        print 'orig:', image.get_camera_pose()
        image.set_camera_pose( ned, [yaw/d2r, pitch/d2r, roll/d2r] )
        print 'new: ', image.get_camera_pose()
        image.save_meta()
