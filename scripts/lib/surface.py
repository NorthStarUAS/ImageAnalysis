# code to trianglulate image features (in 3d) based on camera poses

import cv2
import numpy as np
import os

from props import getNode
import props_json

from . import camera
from . import image
from .logger import log, qlog
from . import project
from . import srtm

surface_node = getNode("/surface", True)

# compute the 3d triangulation of the matches between a pair of images
def triangulate_ned(i1, i2):
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
        p1 = i1.kp_list[ pair[0] ].pt
        p2 = i2.kp_list[ pair[1] ].pt
        uv1.append( [p1[0], p1[1], 1.0] )
        uv2.append( [p2[0], p2[1], 1.0] )
    pts1 = IK.dot(np.array(uv1).T)
    pts2 = IK.dot(np.array(uv2).T)
    points = cv2.triangulatePoints(PROJ1, PROJ2, pts1[:2], pts2[:2])
    points /= points[3]
    return points

def estimate_surface_ned(i1, i2):
    points = triangulate_ned(i1, i2)
    # num_matches = points.shape[1]
    if points is None:
        return None, None
    else:
        return np.average(points[2]), np.std(points[2])

# compute the pairwise surface estimate and then update the property
# tree records
def update_estimate(i1, i2):
    avg, std = estimate_surface_ned(i1, i2)
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
    pair1_node.setFloat("surface_m", float("%.1f" % -avg))
    pair1_node.setInt("weight", weight)
    pair1_node.setFloat("stddev", float("%.1f" % std))
    pair2_node.setFloat("surface_m", float("%.1f" % -avg))
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

    # in case the caller is interested. notice: work is done in NED so
    # returning the negative of the downbb
    return -avg, std

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
    surface_file = os.path.join(analysis_dir, "surface.json")
    props_json.load(surface_file, surface_node)

def save(analysis_dir):
    surface_file = os.path.join(analysis_dir, "surface.json")
    props_json.save(surface_file, surface_node)
    
