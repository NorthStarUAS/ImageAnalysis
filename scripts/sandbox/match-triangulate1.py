#!/usr/bin/python3

# bin distance, bin vector angle

import argparse
import cv2
import math
import numpy as np
from tqdm import tqdm

from lib import camera
from lib import project

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--scale', type=float, default=0.4, help='scale image before processing')
parser.add_argument('--ground', type=float, required=True, help="ground elevation")
args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()
# proj.load_features(descriptors=False)
proj.load_match_pairs()

ratio_cutoff = 0.60
grid_steps = 8

# common camera parameters
K = camera.get_K()
IK = np.linalg.inv(K)
dist_coeffs = camera.get_dist_coeffs()

print('Computing pair triangulations:')
for i, i1 in enumerate(proj.image_list):
    sum = 0.0
    count = 0
    for j, i2 in enumerate(proj.image_list):
        if j == i:
            continue
        if not i2.name in i1.match_list:
            continue
        
        num_matches = len(i1.match_list[i2.name])
        if num_matches == 0:
            continue

        if not i1.kp_list or not len(i1.kp_list):
            i1.load_features()
        if not i2.kp_list or not len(i2.kp_list):
            i2.load_features()

        rvec1, tvec1 = i1.get_proj()
        rvec2, tvec2 = i2.get_proj()
        R1, jac = cv2.Rodrigues(rvec1)
        PROJ1 = np.concatenate((R1, tvec1), axis=1)
        R2, jac = cv2.Rodrigues(rvec2)
        PROJ2 = np.concatenate((R2, tvec2), axis=1)

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
        
        sum += np.average(points[2])*num_matches
        count += num_matches
        
        print(" ", i1.name, "+", i2.name, "avg ground est:", np.average(points[2]))
    if count > 0:
        print(i1.name, "estimated surface below:", "%.1f" % (sum / count))
    else:
        print(i1.name, "no matches, no triangulation, no estimate")
