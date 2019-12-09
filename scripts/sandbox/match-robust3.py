#!/usr/bin/python3

# looks for homography, but uses only matches with "n" range, trying
# all the ranges, find the best hit

# v2 simplify lots / cut cruft, ignore angle

# v3 test angle ranges individually?

import argparse
import cv2
import math
import numpy as np
import os
import pyexiv2                  # dnf install python3-exiv2 (py3exiv2)
from tqdm import tqdm
import matplotlib.pyplot as plt

from props import root, getNode
import props_json

from lib import camera
from lib import image

parser = argparse.ArgumentParser(description='Align and combine sentera images.')
parser.add_argument('--scale', type=float, default=0.4, help='scale image before processing')
parser.add_argument('image1', help='image1 path')
parser.add_argument('image2', help='image1 path')
args = parser.parse_args()

ratio_cutoff = 0.60

def detect_camera(image_path):
    camera = ""
    exif = pyexiv2.ImageMetadata(image_path)
    exif.read()
    if 'Exif.Image.Make' in exif:
        camera = exif['Exif.Image.Make'].value
    if 'Exif.Image.Model' in exif:
        camera += '_' + exif['Exif.Image.Model'].value
    if 'Exif.Photo.LensModel' in exif:
        camera += '_' + exif['Exif.Photo.LensModel'].value
    camera = camera.replace(' ', '_')
    return camera

image1 = args.image1
image2 = args.image2
    
cam1 = detect_camera(image1)
cam2 = detect_camera(image2)
print(cam1)
print(cam2)

cam1_node = getNode("/camera1", True)
cam2_node = getNode("/camera2", True)

if props_json.load(os.path.join("../cameras", cam1 + ".json"), cam1_node):
    print("successfully loaded cam1 config")
if props_json.load(os.path.join("../cameras", cam2 + ".json"), cam2_node):
    print("successfully loaded cam2 config")

tmp = []
for i in range(9):
    tmp.append( cam1_node.getFloatEnum('K', i) )
K1 = np.copy(np.array(tmp)).reshape(3,3)
print("K1:", K1)

tmp = []
for i in range(5):
    tmp.append( cam1_node.getFloatEnum('dist_coeffs', i) )
dist1 = np.array(tmp)
print("dist1:", dist1)

tmp = []
for i in range(9):
    tmp.append( cam2_node.getFloatEnum('K', i) )
K2 = np.copy(np.array(tmp)).reshape(3,3)
print("K2:", K2)

tmp = []
for i in range(5):
    tmp.append( cam2_node.getFloatEnum('dist_coeffs', i) )
dist2 = np.array(tmp)
print("dist2:", dist2)

i1 = cv2.imread(image1, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
i2 = cv2.imread(image2, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
if i1 is None:
    print("Error loading:", image1)
    quit()
if i2 is None:
    print("Error loading:", image2)
    quit()

#i1 = cv2.undistort(i1, K1, dist1)
#i2 = cv2.undistort(i2, K1, dist1)

# scale images (anticipating images are identical dimensions, but this
# will force that assumption if the happen to not be.)
(h, w) = i1.shape[:2]
i1 = cv2.resize(i1, (int(w*args.scale), int(h*args.scale)))
i2 = cv2.resize(i2, (int(w*args.scale), int(h*args.scale)))
diag = int(math.sqrt(h*h + w*w) * args.scale)
print("h:", h, "w:", w)
print("scaled diag:", diag)
detector = cv2.xfeatures2d.SIFT_create()
kp1 = detector.detect(i1)
kp2 = detector.detect(i2)
print("Keypoints:", len(kp1), len(kp2))

kp1, des1 = detector.compute(i1, kp1)
kp2, des2 = detector.compute(i2, kp2)
print("Descriptors:", len(des1), len(des2))

FLANN_INDEX_KDTREE = 1
flann_params = {
    'algorithm': FLANN_INDEX_KDTREE,
    'trees': 5
}
search_params = dict(checks=100)
matcher = cv2.FlannBasedMatcher(flann_params, search_params)
matches = matcher.knnMatch(des1, des2, k=3)
print("Raw matches:", len(matches))

def draw_inlier(src1, src2, kpt1, kpt2, inlier, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == 'ONLY_LINES':
        for i in range(len(inlier)):
            left = kpt1[inlier[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kpt2[inlier[i].trainIdx].pt, (src1
.shape[1], 0)))
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
    cv2.waitKey()

print("collect stats...")
match_stats = []
for i, m in enumerate(tqdm(matches)):
    best_index = -1
    best_metric = 9
    best_angle = 0
    best_size = 0
    best_dist = 0
    for j in range(len(m)):
        if m[j].distance >= 290:
            break
        ratio = m[0].distance / m[j].distance
        if ratio < ratio_cutoff:
            break
        p1 = np.float32(kp1[m[j].queryIdx].pt)
        p2 = np.float32(kp2[m[j].trainIdx].pt)
        raw_dist = np.linalg.norm(p2 - p1)
        # angle difference mapped to +/- 90
        a1 = np.array(kp1[m[j].queryIdx].angle)
        a2 = np.array(kp2[m[j].trainIdx].angle)
        angle_diff = abs((a1-a2+90) % 180 - 90)
        s1 = np.array(kp1[m[j].queryIdx].size)
        s2 = np.array(kp2[m[j].trainIdx].size)
        if s1 > s2:
            size_diff = s1 / s2
        else:
            size_diff = s2 / s1
        if size_diff > 1.5:
            continue
        metric = (size_diff + 1) / ratio
        #print(" ", j, m[j].distance, size_diff, metric)
        if best_index < 0 or metric < best_metric:
            best_metric = metric
            best_index = j
            best_angle = angle_diff
            best_size = size_diff
            best_dist = raw_dist
    if best_index >= 0:
        #print(i, best_index, m[best_index].distance, best_size, best_metric)
        match_stats.append( [ m[best_index], best_index, ratio, best_metric,
                              best_angle, best_size, best_dist ] )

# sort match_stats by best first (according to 'metric')
# match_stats = sorted( match_stats, key=lambda fields: fields[3])

maxrange = int(diag*0.02)
step = int(maxrange / 2)
tol = int(diag*0.005)
if tol < 5: tol = 5
maxdist = int(diag*0.55)
best_fitted_matches = 0
for target_dist in range(0, maxdist, step):
    dist_matches = []
    for line in match_stats:
        best_metric = line[3]
        best_dist = line[6]
        if abs(best_dist - target_dist) > maxrange:
            continue
        dist_matches.append(line)
    print("Target distance:", target_dist, "candidates:", len(dist_matches))
    astep = 10
    for angle in range(0, 90, astep):
        print(angle)
        angle_matches = []
        for line in dist_matches:
            match = line[0]
            best_metric = line[3]
            best_angle = line[4]
            if abs(angle - best_angle) > astep*2:
                continue
            angle_matches.append(match)
        if len(angle_matches) > 7:
            src = []
            dst = []
            for m in angle_matches:
                src.append( kp1[m.queryIdx].pt )
                dst.append( kp2[m.trainIdx].pt )
            H, status = cv2.findHomography(np.array([src]).astype(np.float32),
                                           np.array([dst]).astype(np.float32),
                                           cv2.RANSAC,
                                           tol)
            matches_fit = []
            matches_dist = []
            for i, m in enumerate(angle_matches):
                if status[i]:
                    matches_fit.append(m)
                    matches_dist.append(m.distance)
            if len(matches_fit) > best_fitted_matches:
                best_fitted_matches = len(matches_fit)
                print("Filtered matches:", len(angle_matches),
                      "Fitted matches:", len(matches_fit))
                print("metric cutoff:", best_metric)
                matches_dist = np.array(matches_dist)
                print("avg match quality:", np.average(matches_dist))
                print("max match quality:", np.max(matches_dist))
                i1_new = cv2.warpPerspective(i1, H, (i1.shape[1], i1.shape[0]))
                blend = cv2.addWeighted(i1_new, 0.5, i2, 0.5, 0)
                cv2.imshow('blend', blend)
                draw_inlier(i1, i2, kp1, kp2, matches_fit, 'ONLY_LINES')

if False:
    cv2.imshow('i1', i1)
    cv2.imshow('i1_new', i1_new)
    cv2.imshow('i2', i2)
    cv2.imshow('blend', blend)
    
cv2.waitKey()
