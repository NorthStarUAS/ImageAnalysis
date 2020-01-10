#!/usr/bin/python3

# do pairwise essential matrix solve and back out relative poses.  Then look
# at relative ned heading vs. pose/essential/match heading and estimate a
# yaw error.  This can be fed back into the smart matcher to improve it's
# results (hopefully.)

# Assuming images are taking looking straight down, then given the
# pose of image1 and the matches/essential matrix relative to image2,
# we can compute the estimate direction in NED space image2 should be
# from image1.  We can also compute their actual direction based on
# gps coordinates.  If we trust gps, then the difference should be
# comparable to the EKF yaw error.

import argparse
import cv2
import math
import numpy as np

from props import getNode

from lib import camera
from lib import project
from lib import srtm
from lib import surface
from lib import transformations

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('project', help='project directory')
args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()
# proj.load_features(descriptors=False)
proj.load_match_pairs()

r2d = 180 / math.pi
d2r = math.pi / 180

# lookup ned reference
ref_node = getNode("/config/ned_reference", True)
ref = [ ref_node.getFloat('lat_deg'),
        ref_node.getFloat('lon_deg'),
        ref_node.getFloat('alt_m') ]
# setup SRTM ground interpolator
srtm.initialize( ref, 6000, 6000, 30 )

surface.load(proj.analysis_dir)

# compute the 3d triangulation of the matches between a pair of
# images.  This is super fancy when it works, but I'm struggling with
# some cases that break so I clearly don't understand some nuance, or
# have made a mistake somewhere.

#method = cv2.RANSAC
method = cv2.LMEDS
def find_essential(i1, i2):
    # quick sanity checks
    if i1 == i2:
        return None
    if not i2.name in i1.match_list:
        return None
    if len(i1.match_list[i2.name]) == 0:
        return None

    if not i1.kp_list or not len(i1.kp_list):
        i1.load_features()
    if not i2.kp_list or not len(i2.kp_list):
        i2.load_features()

    # camera calibration
    K = camera.get_K()
    IK = np.linalg.inv(K)

    # get poses
    rvec1, tvec1 = i1.get_proj()
    rvec2, tvec2 = i2.get_proj()
    R1, jac = cv2.Rodrigues(rvec1)
    PROJ1 = np.concatenate((R1, tvec1), axis=1)
    R2, jac = cv2.Rodrigues(rvec2)
    PROJ2 = np.concatenate((R2, tvec2), axis=1)

    # setup data structurs of cv2 call
    uv1 = []; uv2 = []; indices = []
    for pair in i1.match_list[i2.name]:
        uv1.append( i1.kp_list[ pair[0] ].pt )
        uv2.append( i2.kp_list[ pair[1] ].pt )
    uv1 = np.float32(uv1)
    uv2 = np.float32(uv2)
    E, mask = cv2.findEssentialMat(uv1, uv2, K, method=method)
    print(i1.name, 'vs', i2.name)
    print("E:\n", E)
    print()
    (n, R, tvec, mask) = cv2.recoverPose(E, uv1, uv2, K)
    print('  inliers:', n, 'of', len(uv1))
    print('  R:', R)
    print('  tvec:', tvec)
    
    # convert R to homogeonous
    #Rh = np.concatenate((R, np.zeros((3,1))), axis=1)
    #Rh = np.concatenate((Rh, np.zeros((1,4))), axis=0)
    #Rh[3,3] = 1
    # extract the equivalent quaternion, and invert
    q = transformations.quaternion_from_matrix(R)
    q_inv = transformations.quaternion_inverse(q)

    (ned1, ypr1, quat1) = i1.get_camera_pose()
    (ned2, ypr2, quat2) = i2.get_camera_pose()
    diff = np.array(ned2) - np.array(ned1)
    dist = np.linalg.norm( diff )
    dir = diff / dist
    print('dist:', dist, 'ned dir:', dir[0], dir[1], dir[2])
    crs_gps = 90 - math.atan2(dir[0], dir[1]) * r2d
    if crs_gps < 0: crs_gps += 360
    if crs_gps > 360: crs_gps -= 360
    print('crs_gps: %.1f' % crs_gps)

    Rbody2ned = i1.get_body2ned()
    cam2body = i1.get_cam2body()
    body2cam = i1.get_body2cam()
    est_dir = Rbody2ned.dot(cam2body).dot(R).dot(tvec)
    est_dir = est_dir / np.linalg.norm(est_dir) # normalize
    print('est dir:', est_dir.tolist())
    crs_fit = 90 - math.atan2(-est_dir[0], -est_dir[1]) * r2d
    if crs_fit < 0: crs_fit += 360
    if crs_fit > 360: crs_fit -= 360
    print('est crs_fit: %.1f' % crs_fit)
    print("est yaw error: %.1f" % (crs_fit - crs_gps))

# print('Computing essential matrix for pairs:')
# for i, i1 in enumerate(proj.image_list):
#     ned, ypr, quat = i1.get_camera_pose()
#     srtm_elev = srtm.ned_interp( [ned[0], ned[1]] )
#     i1_node = surface.surface_node.getChild(i1.name, True)
#     i1_node.setFloat("srtm_surface_m", "%.1f" % srtm_elev)
#     for j, i2 in enumerate(proj.image_list):
#         if j > i:
#             find_essential(i1, i2)

# project the center of image 2 into the uv space of image 1.  The
# 'heading' of the i2 center relative to the i1 center plus the i1 yaw
# estimate should =~ the gps ground course between the two image
# centers. The difference is our estimated ekf yaw error.

def find_affine(i1, i2):
    # quick sanity checks
    if i1 == i2:
        return None, None
    if not i2.name in i1.match_list:
        return None, None
    if len(i1.match_list[i2.name]) == 0:
        return None, None

    if not i1.kp_list or not len(i1.kp_list):
        i1.load_features()
    if not i2.kp_list or not len(i2.kp_list):
        i2.load_features()

    # affine transformation from i2 uv coordinate system to i1
    uv1 = []; uv2 = []; indices = []
    for pair in i1.match_list[i2.name]:
        uv1.append( i1.kp_list[ pair[0] ].pt )
        uv2.append( i2.kp_list[ pair[1] ].pt )
    uv1 = np.float32([uv1])
    uv2 = np.float32([uv2])
    affine, status = \
        cv2.estimateAffinePartial2D(uv2, uv1)
    print(i1.name, 'vs', i2.name)
    print(" affine:\n", affine)
    (rot, tx, ty, sx, sy) = decomposeAffine(affine)
    print(" ", rot, tx, ty, sx, sy)
    print()

    (ned1, ypr1, quat1) = i1.get_camera_pose()
    (ned2, ypr2, quat2) = i2.get_camera_pose()
    diff = np.array(ned2) - np.array(ned1)
    dist = np.linalg.norm( diff )
    dir = diff / dist
    print(" dist:", dist, 'ned dir:', dir[0], dir[1], dir[2])
    crs_gps = 90 - math.atan2(dir[0], dir[1]) * r2d
    if crs_gps < 0: crs_gps += 360
    if crs_gps > 360: crs_gps -= 360

    # center of i2 in i1's uv coordinate system
    (w, h) = camera.get_image_params()
    cx = int(w*0.5)
    cy = int(h*0.5)
    print("center:", [cx, cy])
    newc = affine.dot(np.float32([cx, cy, 1.0]))[:2]
    
    cdiff = [ newc[0] - cx, cy - newc[1] ]
    print("new center:", newc)
    print("center diff:", cdiff)

    crs_aff = 90 - math.atan2(cdiff[1], cdiff[0]) * r2d
    
    (_, air_ypr1, _) = i1.get_aircraft_pose()
    print(" aircraft yaw: %.1f" % air_ypr1[0])
    print(" affine rotation: %.1f" % rot)
    print(" affine course: %.1f" % crs_aff)
    print(" ground course: %.1f" % crs_gps)
    yaw_error = crs_gps - air_ypr1[0] - crs_aff
    if yaw_error < -180: yaw_error += 360
    if yaw_error > 180: yaw_error -= 360
    print(" error: %.1f" % yaw_error)

    # aircraft yaw (est) + affine course + yaw error = ground course
    
    # use affine matrix to project center of i1 into 
    # Rbody2ned = i1.get_body2ned()
    # cam2body = i1.get_cam2body()
    # body2cam = i1.get_body2cam()
    # est_dir = Rbody2ned.dot(cam2body).dot(R).dot(tvec)
    # est_dir = est_dir / np.linalg.norm(est_dir) # normalize
    # print('est dir:', est_dir.tolist())
    # crs_fit = 90 - math.atan2(-est_dir[0], -est_dir[1]) * r2d
    # if crs_fit < 0: crs_fit += 360
    # if crs_fit > 360: crs_fit -= 360
    # print('est crs_fit: %.1f' % crs_fit)
    # print("est yaw error: %.1f" % (crs_fit - crs_gps))

    return yaw_error, dist

def decomposeAffine(affine):
    tx = affine[0][2]
    ty = affine[1][2]

    a = affine[0][0]
    b = affine[0][1]
    c = affine[1][0]
    d = affine[1][1]

    sx = math.sqrt( a*a + b*b )
    if a < 0.0:
        sx = -sx
    sy = math.sqrt( c*c + d*d )
    if d < 0.0:
        sy = -sy

    rotate_deg = math.atan2(-b,a) * 180.0/math.pi
    if rotate_deg < -180.0:
        rotate_deg += 360.0
    if rotate_deg > 180.0:
        rotate_deg -= 360.0
    return (rotate_deg, tx, ty, sx, sy)

print("Computing affine matrix for pairs:")
for i, i1 in enumerate(proj.image_list):
    ned, ypr, quat = i1.get_camera_pose()
    srtm_elev = srtm.ned_interp( [ned[0], ned[1]] )
    i1_node = surface.surface_node.getChild(i1.name, True)
    i1_node.setFloat("srtm_surface_m", "%.1f" % srtm_elev)
    for j, i2 in enumerate(proj.image_list):
        if i != j:
            yaw_error, dist = find_affine(i1, i2)
            if not yaw_error is None:
                tri_node = i1_node.getChild("yaw_pairs", True)
                pair_node = tri_node.getChild(i2.name, True)
                pair_node.setFloat("yaw_error", "%.1f" % yaw_error)
                pair_node.setFloat("dist_m", "%.1f" % dist)
            
surface.surface_node.pretty_print()
#surface.save(proj.analysis_dir)

