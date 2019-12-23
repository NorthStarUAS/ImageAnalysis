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

def gen_grid(w, h, steps):
    grid_list = []
    u_list = np.linspace(0, w, steps + 1)
    v_list = np.linspace(0, h, steps + 1)
    for v in v_list:
        for u in u_list:
            grid_list.append( [u, v] )
    return grid_list

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

def draw_inlier(src1, src2, kpt1, kpt2, inlier, drawing_type, scale):
    h, w = src1.shape[:2]
    src1 = cv2.resize(src1, (int(w*scale), int(h*scale)))
    src2 = cv2.resize(src2, (int(w*scale), int(h*scale)))
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == 'ONLY_LINES':
        for i in range(len(inlier)):
            left = np.array(kpt1[inlier[i].queryIdx].pt)*scale
            right = tuple(sum(x) for x in zip(np.array(kpt2[inlier[i].trainIdx].pt)*scale, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 
255, 255))

    elif drawing_type == 'LINES_AND_POINTS':
        for i in range(len(inlier)):
            left = kpt1[inlier[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kpt2[inlier[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))
        for i in range(len(inlier)):
            left = kpt1[inlier[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kpt2[inlier[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    cv2.imshow('show', output)

# common camera parameters
K = camera.get_K()
IK = np.linalg.inv(K)
dist_coeffs = camera.get_dist_coeffs()

print('Computing pair triangulations:')
for i, i1 in enumerate(proj.image_list):
    sum = 0.0
    count = 0
    for j, i2 in enumerate(proj.image_list):
        if j <= i:
            continue
        if not i2.name in i1.match_list:
            continue
        
        num_matches = len(i1.match_list[i2.name])
        if num_matches == 0:
            continue

        if not len(i1.kp_list) or not len(i1.des_list):
            i1.detect_features(args.scale)
        if not len(i2.kp_list) or not len(i2.des_list):
            i2.detect_features(args.scale)

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
