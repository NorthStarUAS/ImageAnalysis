# code to estimate world surface elevation and EKF yaw error from
# image direct pose informaation.

# - trianglulate image features (in 3d) based on camera poses
# - use essential/fundamental matrix + camera pose to estimate yaw error
# - use affine transformation + camera pose to estimate yaw error

import cv2
import math
import numpy as np
import os

from props import getNode
import props_json

from . import camera
from . import image
from .logger import log, qlog
from . import project
from . import srtm

r2d = 180 / math.pi
d2r = math.pi / 180

surface_node = getNode("/surface", True)

# compute the 3d triangulation of the matches between a pair of images
def triangulate_features(i1, i2):
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

    # setup data structures for cv2 call
    uv1 = []; uv2 = []; indices = []
    for pair in i1.match_list[i2.name]:
        p1 = i1.kp_list[ pair[0] ].pt
        p2 = i2.kp_list[ pair[1] ].pt
        uv1.append( [p1[0], p1[1], 1.0] )
        uv2.append( [p2[0], p2[1], 1.0] )
    pts1 = IK.dot(np.array(uv1).T)
    pts2 = IK.dot(np.array(uv2).T)
    points = cv2.triangulatePoints(PROJ1, PROJ2, pts1[:2], pts2[:2])
    points /= points[3]
    return points

# find (forward) affine transformation between feature pairs
def find_affine(i1, i2):
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

    # affine transformation from i2 uv coordinate system to i1
    uv1 = []; uv2 = []; indices = []
    for pair in i1.match_list[i2.name]:
        uv1.append( i1.kp_list[ pair[0] ].pt )
        uv2.append( i2.kp_list[ pair[1] ].pt )
    uv1 = np.float32([uv1])
    uv2 = np.float32([uv2])
    affine, status = \
        cv2.estimateAffinePartial2D(uv2, uv1)
    return affine

# return individual components of affine transform: rot, tx, ty, sx,
# sy (units are degrees and pixels)
def decompose_affine(affine):
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

    angle_deg = math.atan2(-b,a) * 180.0/math.pi
    if angle_deg < -180.0:
        angle_deg += 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    return (angle_deg, tx, ty, sx, sy)

# average of the triangulated points (converted to positive elevation)
def estimate_surface_elevation(i1, i2):
    points = triangulate_features(i1, i2)
    # num_matches = points.shape[1]
    if points is None:
        return None, None
    else:
        # points are are triangulated in the NED coordinates, so
        # invert the vertical (down) average before returning the
        # answer.
        return -np.average(points[2]), np.std(points[2])

# Estimate image pose yaw error (based on found pairs affine
# transform, original image pose, and gps positions; assumes a mostly
# nadir camara pose.)  After computering affine transform, project
# image 2 center uv into image1 uv space and compute approximate
# course in local uv space, then add this to direct pose yaw estimate
# and compare to gps course.
def estimate_yaw_error(i1, i2):
    affine = find_affine(i1, i2)
    if  affine is None:
        return None, None, None, None

    # fyi ...
    # print(i1.name, 'vs', i2.name)
    # print(" affine:\n", affine)
    (rot, tx, ty, sx, sy) = decompose_affine(affine)
    # print(" ", rot, tx, ty, sx, sy)
    if abs(ty) > 0:
        weight = abs(ty / tx)
    else:
        weight = abs(tx)

    # ground course between camera poses
    (ned1, ypr1, quat1) = i1.get_camera_pose()
    (ned2, ypr2, quat2) = i2.get_camera_pose()
    diff = np.array(ned2) - np.array(ned1)
    dist = np.linalg.norm( diff )
    dir = diff / dist
    print(" dist:", dist, 'ned dir:', dir[0], dir[1], dir[2])
    crs_gps = 90 - math.atan2(dir[0], dir[1]) * r2d
    if crs_gps < 0: crs_gps += 360
    if crs_gps > 360: crs_gps -= 360

    # center pixel of i2 in i1's uv coordinate system
    (w, h) = camera.get_image_params()
    cx = int(w*0.5)
    cy = int(h*0.5)
    print("center:", [cx, cy])
    newc = affine.dot(np.float32([cx, cy, 1.0]))[:2]
    cdiff = [ newc[0] - cx, cy - newc[1] ]
    #print("new center:", newc)
    #print("center diff:", cdiff)

    # estimated course based on i1 pose and [local uv coordinate
    # system] affine transform
    crs_aff = 90 - math.atan2(cdiff[1], cdiff[0]) * r2d
    (_, air_ypr1, _) = i1.get_aircraft_pose()
    #print(" aircraft yaw: %.1f" % air_ypr1[0])
    #print(" affine course: %.1f" % crs_aff)
    #print(" ground course: %.1f" % crs_gps)
    crs_fit = air_ypr1[0] + crs_aff
    
    yaw_error = crs_gps - crs_fit
    if yaw_error < -180: yaw_error += 360
    if yaw_error > 180: yaw_error -= 360
    print(" estimated yaw error: %.1f" % yaw_error)

    # aircraft yaw (est) + affine course + yaw error = ground course
    
    return yaw_error, dist, crs_aff, weight

# compute the pairwise surface estimate and then update the property
# tree records
def update_surface_estimate(i1, i2):
    avg, std = estimate_surface_elevation(i1, i2)
    if avg is None:
        return None, None

    i1_node = surface_node.getChild(i1.name, True)
    i2_node = surface_node.getChild(i2.name, True)
    tri1_node = i1_node.getChild("tri_surface_pairs", True)
    tri2_node = i2_node.getChild("tri_surface_pairs", True)
    
    # update pairwise info in the property tree
    weight = len(i1.match_list[i2.name])
    pair1_node = tri1_node.getChild(i2.name, True)
    pair2_node = tri2_node.getChild(i1.name, True)
    pair1_node.setFloat("surface_m", float("%.1f" % avg))
    pair1_node.setInt("weight", weight)
    pair1_node.setFloat("stddev", float("%.1f" % std))
    pair2_node.setFloat("surface_m", float("%.1f" % avg))
    pair2_node.setInt("weight", weight)
    pair2_node.setFloat("stddev", float("%.1f" % std))

    # update the average surface values
    cutoff_std = 25             # more than this suggests a bad set of matches

    sum1 = 0
    count1 = 0
    for child in tri1_node.getChildren():
        pair_node = tri1_node.getChild(child)
        surf = pair_node.getFloat("surface_m")
        weight = pair_node.getInt("weight")
        stddev = pair_node.getFloat("stddev")
        if stddev < cutoff_std:
            sum1 += surf * weight
            count1 += weight
    if count1 > 0:
        i1_node.setFloat("tri_surface_m", float("%.1f" % (sum1 / count1)))

    sum2 = 0
    count2 = 0
    for child in tri2_node.getChildren():
        pair_node = tri2_node.getChild(child)
        surf = pair_node.getFloat("surface_m")
        weight = pair_node.getInt("weight")
        stddev = pair_node.getFloat("stddev")
        if stddev < cutoff_std:
            sum2 += surf * weight
            count2 += weight
    if count2 > 0:
        i2_node.setFloat("tri_surface_m", float("%.1f" % (sum2 / count2)))

    return avg, std

# compute the pairwise surface estimate and then update the property
# tree records
def update_yaw_error_estimate(i1, i2):
    yaw_error, dist, crs_affine, weight = estimate_yaw_error(i1, i2)
    if yaw_error is None:
        return None, None

    i1_node = surface_node.getChild(i1.name, True)
    yaw_node = i1_node.getChild("yaw_pairs", True)
    
    # update pairwise info in the property tree
    pair_node = yaw_node.getChild(i2.name, True)
    pair_node.setFloat("yaw_error", "%.1f" % yaw_error)
    pair_node.setFloat("dist_m", "%.1f" % dist)
    pair_node.setFloat("relative_crs", "%.1f" % crs_affine)
    pair_node.setFloat("weight", "%.1f" % weight)

    sum = 0
    count = 0
    for child in yaw_node.getChildren():
        pair_node = yaw_node.getChild(child)
        yaw_error = pair_node.getFloat("yaw_error")
        weight = pair_node.getInt("weight")
        sum += yaw_error * weight
        count += weight
    if count > 0:
        i1_node.setFloat("yaw_error", float("%.1f" % (sum / count)))
        return sum / count
    else:
        return None

# return the average of estimated surfaces below the image pair
def get_estimate(i1, i2):
    i1_node = surface_node.getChild(i1.name, True)
    i2_node = surface_node.getChild(i2.name, True)
    tri1_node = i1_node.getChild("tri_surface_pairs", True)
    tri2_node = i2_node.getChild("tri_surface_pairs", True)

    count = 0
    sum = 0
    if i1_node.hasChild("tri_surface_m"):
        sum += i1_node.getFloat("tri_surface_m")
        count += 1
    if i2_node.hasChild("tri_surface_m"):
        sum += i2_node.getFloat("tri_surface_m")
        count += 1
        
    if count > 0:
        return sum / count

    # no triangulation estimate yet, fall back to SRTM lookup
    
    else:
        return None

# find srtm surface altitude under each camera pose
def update_srtm_elevations(proj):
    for image in proj.image_list:
        ned, ypr, quat = image.get_camera_pose()
        surface = srtm.ned_interp([ned[0], ned[1]])
        image_node = surface_node.getChild(image.name, True)
        image_node.setFloat("srtm_surface_m", float("%.1f" % surface))
        
def load(analysis_dir):
    surface_file = os.path.join(analysis_dir, "smart.json")
    props_json.load(surface_file, surface_node)

def save(analysis_dir):
    surface_file = os.path.join(analysis_dir, "smart.json")
    props_json.save(surface_file, surface_node)
    
