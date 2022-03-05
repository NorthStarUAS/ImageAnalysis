"""Estimate world surface el. & EKF yaw err. from image pose informaation."""

# - trianglulate image features (in 3d) based on camera poses
# - use essential/fundamental matrix + camera pose to estimate yaw error
# - use affine transformation + camera pose to estimate yaw error

import math
import os

import cv2
import numpy as np
from props import getNode, props_json

from . import camera, srtm
from .logger import qlog

r2d = 180 / math.pi
d2r = math.pi / 180

smart_node = getNode("/smart", True)


def triangulate_features(i1, i2):
    """Compute the 3d triangulation of the matches between a pair of images."""
    # quick sanity checks
    if i1 == i2:
        return None
    if i2.name not in i1.match_list:
        return None
    if len(i1.match_list[i2.name]) == 0:
        return None

    if not i1.kp_list or not len(i1.kp_list):
        i1.load_features()
    if not i2.kp_list or not len(i2.kp_list):
        i2.load_features()

    # camera calibration
    k = camera.get_K()
    i_k = np.linalg.inv(k)

    # get poses
    rvec1, tvec1 = i1.get_proj()
    rvec2, tvec2 = i2.get_proj()
    r1, jac = cv2.Rodrigues(rvec1)
    proj1 = np.concatenate((r1, tvec1), axis=1)
    r2, jac = cv2.Rodrigues(rvec2)
    proj2 = np.concatenate((r1, tvec2), axis=1)

    # setup data structures for cv2 call
    uv1 = []
    uv2 = []
    for pair in i1.match_list[i2.name]:
        p1 = i1.kp_list[pair[0]].pt
        p2 = i2.kp_list[pair[1]].pt
        uv1.append([p1[0], p1[1], 1.0])
        uv2.append([p2[0], p2[1], 1.0])
    pts1 = i_k.dot(np.array(uv1).T)
    pts2 = i_k.dot(np.array(uv2).T)
    points = cv2.triangulatePoints(proj1, proj2, pts1[:2], pts2[:2])
    points /= points[3]
    return points


def find_affine(i1, i2):
    """Find (forward) affine transformation between feature pairs."""
    # quick sanity checks
    if i1 == i2:
        return None
    if i2.name not in i1.match_list:
        return None
    if len(i1.match_list[i2.name]) == 0:
        return None

    if not i1.kp_list or not len(i1.kp_list):
        i1.load_features()
    if not i2.kp_list or not len(i2.kp_list):
        i2.load_features()

    # affine transformation from i2 uv coordinate system to i1
    uv1 = []
    uv2 = []
    for pair in i1.match_list[i2.name]:
        uv1.append(i1.kp_list[pair[0]].pt)
        uv2.append(i2.kp_list[pair[1]].pt)
    uv1 = np.float32([uv1])
    uv2 = np.float32([uv2])
    affine, status = cv2.estimateAffinePartial2D(uv2, uv1)
    return affine


#
def decompose_affine(affine):
    """Get individual components of affine transform: rot, tx, ty, sx, sy."""
    tx = affine[0][2]
    ty = affine[1][2]

    a = affine[0][0]
    b = affine[0][1]
    c = affine[1][0]
    d = affine[1][1]

    sx = math.sqrt(a * a + b * b)
    if a < 0.0:
        sx = -sx
    sy = math.sqrt(c * c + d * d)
    if d < 0.0:
        sy = -sy

    angle_deg = math.atan2(-b, a) * 180.0 / math.pi
    if angle_deg < -180.0:
        angle_deg += 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    return (angle_deg, tx, ty, sx, sy)


def estimate_surface_elevation(i1, i2):
    """Average triangulated point elevation."""
    points = triangulate_features(i1, i2)
    (ned1, ypr1, quat1) = i1.get_camera_pose()
    (ned2, ypr2, quat2) = i2.get_camera_pose()
    diff = np.array(ned2) - np.array(ned1)
    dist_m = np.linalg.norm(diff)
    # num_matches = points.shape[1]
    if points is None:
        return None, None, dist_m
    else:
        # points are are triangulated in the NED coordinates, so
        # invert the vertical (down) average before returning the
        # answer.
        return -np.average(points[2]), np.std(points[2]), dist_m


# Estimate image pose yaw error (based on found pairs affine
# transform, original image pose, and gps positions; assumes a mostly
# nadir camara pose.)  After computering affine transform, project
# image 2 center uv into image1 uv space and compute approximate
# course in local uv space, then add this to direct pose yaw estimate
# and compare to gps course.
def estimate_yaw_error(i1, i2):
    """Estimate image pose yaw error."""
    affine = find_affine(i1, i2)
    if affine is None:
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
    dist = np.linalg.norm(diff)
    direction = diff / dist
    print(" dist:", dist, "ned dir:", direction[0], direction[1], direction[2])
    crs_gps = 90 - math.atan2(direction[0], direction[1]) * r2d
    if crs_gps < 0:
        crs_gps += 360
    if crs_gps > 360:
        crs_gps -= 360

    # center pixel of i2 in i1's uv coordinate system
    (w, h) = camera.get_image_params()
    cx = int(w * 0.5)
    cy = int(h * 0.5)
    print("center:", [cx, cy])
    newc = affine.dot(np.float32([cx, cy, 1.0]))[:2]
    cdiff = [newc[0] - cx, cy - newc[1]]
    # print("new center:", newc)
    # print("center diff:", cdiff)

    # estimated course based on i1 pose and [local uv coordinate
    # system] affine transform
    crs_aff = 90 - math.atan2(cdiff[1], cdiff[0]) * r2d
    (_, air_ypr1, _) = i1.get_aircraft_pose()
    # print(" aircraft yaw: %.1f" % air_ypr1[0])
    # print(" affine course: %.1f" % crs_aff)
    # print(" ground course: %.1f" % crs_gps)
    crs_fit = air_ypr1[0] + crs_aff

    yaw_error = crs_gps - crs_fit
    if yaw_error < -180:
        yaw_error += 360
    if yaw_error > 180:
        yaw_error -= 360
    print(" estimated yaw error: %.1f" % yaw_error)

    # aircraft yaw (est) + affine course + yaw error = ground course

    return yaw_error, dist, crs_aff, weight


# compute the pairwise surface estimate and then update the property
# tree records
def update_surface_estimate(i1, i2):
    """Compute the pairwise surface estimate and update proptree."""
    avg, std, dist_m = estimate_surface_elevation(i1, i2)
    if avg is None:
        return None, None

    i1_node = smart_node.getChild(i1.name, True)
    i2_node = smart_node.getChild(i2.name, True)
    tri1_node = i1_node.getChild("tri_surface_pairs", True)
    tri2_node = i2_node.getChild("tri_surface_pairs", True)

    # update pairwise info in the property tree
    # weight = len(i1.match_list[i2.name])
    weight = dist_m * dist_m
    pair1_node = tri1_node.getChild(i2.name, True)
    pair2_node = tri2_node.getChild(i1.name, True)
    pair1_node.setFloat("surface_m", float("%.1f" % avg))
    pair1_node.setInt("weight", weight)
    pair1_node.setFloat("stddev", float("%.1f" % std))
    pair1_node.setInt("dist_m", dist_m)
    pair2_node.setFloat("surface_m", float("%.1f" % avg))
    pair2_node.setInt("weight", weight)
    pair2_node.setFloat("stddev", float("%.1f" % std))
    pair2_node.setInt("dist_m", dist_m)

    # update the average surface values
    cutoff_std = 25  # more than this suggests a bad set of matches

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


def update_yaw_error_estimate(i1, i2):
    """Update yaw error estimates."""
    yaw_error, dist, crs_affine, weight = estimate_yaw_error(i1, i2)
    if yaw_error is None:
        return 0

    i1_node = smart_node.getChild(i1.name, True)
    yaw_node = i1_node.getChild("yaw_pairs", True)

    # update pairwise info in the property tree
    pair_node = yaw_node.getChild(i2.name, True)
    pair_node.setFloat("yaw_error", "%.1f" % yaw_error)
    pair_node.setFloat("dist_m", "%.1f" % dist)
    pair_node.setFloat("relative_crs", "%.1f" % crs_affine)
    pair_node.setFloat("weight", "%.1f" % weight)

    total = 0
    count = 0
    for child in yaw_node.getChildren():
        pair_node = yaw_node.getChild(child)
        yaw_error = pair_node.getFloat("yaw_error")
        weight = pair_node.getInt("weight")
        dist_m = pair_node.getFloat("dist_m")
        if dist_m >= 0.5 and abs(yaw_error) <= 30:
            total += yaw_error * weight
            count += weight
        # else:
        #    log("yaw error ignored:", i1.name, i2.name, "%.1fm" % dist_m,
        #        "%.1f(deg)" % yaw_error)
    if count > 0:
        i1_node.setFloat("yaw_error", float("%.1f" % (total / count)))
        return total / count
    else:
        return 0


def get_yaw_error_estimate(i1):
    """Get a yaw error estimate for an image."""
    i1_node = smart_node.getChild(i1.name, True)
    if i1_node.hasChild("yaw_error"):
        return i1_node.getFloat("yaw_error")
    else:
        return 0.0


def get_surface_estimate(i1, i2):
    """Return the average of estimated surfaces below the image pair."""
    i1_node = smart_node.getChild(i1.name, True)
    i2_node = smart_node.getChild(i2.name, True)

    count = 0
    total = 0
    if i1_node.hasChild("tri_surface_m"):
        total += i1_node.getFloat("tri_surface_m")
        count += 1
    if i2_node.hasChild("tri_surface_m"):
        total += i2_node.getFloat("tri_surface_m")
        count += 1

    if count > 0:
        return total / count

    # no triangulation estimate yet, fall back to SRTM lookup
    g1 = i1_node.getFloat("srtm_surface_m")
    g2 = i2_node.getFloat("srtm_surface_m")
    ground_m = (g1 + g2) * 0.5
    qlog("  SRTM ground (no triangulation yet): %.1f" % ground_m)
    return ground_m


def update_srtm_elevations(proj):
    """Find srtm surface altitude under each camera pose."""
    for image in proj.image_list:
        ned, ypr, quat = image.get_camera_pose()
        surface = srtm.ned_interp([ned[0], ned[1]])
        image_node = smart_node.getChild(image.name, True)
        image_node.setFloat("srtm_surface_m", float("%.1f" % surface))


def set_yaw_error_estimates(proj):
    """Set yaw error estimates."""
    for image in proj.image_list:
        image_node = smart_node.getChild(image.name, True)
        yaw_node = image_node.getChild("yaw_pairs", True)
        yaw_error_deg = yaw_node.getFloat("yaw_error")
        image.set_aircraft_yaw_error_estimate(yaw_error_deg)


def load(analysis_dir):
    """Load a surface file."""
    surface_file = os.path.join(analysis_dir, "smart.json")
    props_json.load(surface_file, smart_node)


def save(analysis_dir):
    """Save a surface file."""
    surface_file = os.path.join(analysis_dir, "smart.json")
    props_json.save(surface_file, smart_node)
