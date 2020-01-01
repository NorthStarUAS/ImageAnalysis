# code to trianglulate image features (in 3d) based on camera poses

import cv2
import numpy as np

from . import camera
from . import image
from . import project

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
    
