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
from lib import smart
from lib import srtm
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

smart.load(proj.analysis_dir)

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
#     i1_node = smart.surface_node.getChild(i1.name, True)
#     i1_node.setFloat("srtm_surface_m", "%.1f" % srtm_elev)
#     for j, i2 in enumerate(proj.image_list):
#         if j > i:
#             find_essential(i1, i2)

print("Computing affine matrix for pairs:")
for i, i1 in enumerate(proj.image_list):
    ned, ypr, quat = i1.get_camera_pose()
    srtm_elev = srtm.ned_interp( [ned[0], ned[1]] )
    i1_node = smart.surface_node.getChild(i1.name, True)
    i1_node.setFloat("srtm_surface_m", "%.1f" % srtm_elev)
    for j, i2 in enumerate(proj.image_list):
        yaw_error = smart.update_yaw_error_estimate(i1, i2)
            
smart.surface_node.pretty_print()
smart.save(proj.analysis_dir)

