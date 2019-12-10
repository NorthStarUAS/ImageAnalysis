#!/usr/bin/python3

import argparse
import cv2
import math
import numpy as np
from tqdm import tqdm

from lib import project

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--scale', type=float, default=0.4, help='scale image before processing')
args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()
# proj.load_features(descriptors=False)
proj.load_match_pairs()

ratio_cutoff = 0.60

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

print('Computing pair distances:')
dist_list = []
for i, i1 in enumerate(tqdm(proj.image_list)):
    for j, i2 in enumerate(proj.image_list):
        if j <= i:
            continue
        # camera pose distance check
        ned1, ypr1, q1 = i1.get_camera_pose()
        ned2, ypr2, q2 = i2.get_camera_pose()
        dist = np.linalg.norm(np.array(ned2) - np.array(ned1))
        dist_list.append( [dist, i, j] )

dist_list = sorted(dist_list, key=lambda fields: fields[0])

for line in dist_list:
    dist = line[0]
    i = line[1]
    j = line[2]
    i1 = proj.image_list[i]
    i2 = proj.image_list[j]
    # print(i1.match_list)
    num_matches = len(i1.match_list[i2.name])
    print("dist: %.1f" % dist, i1.name, i2.name, num_matches)
    print("rev matches:", len(i2.match_list[i1.name]))

    if not len(i1.kp_list) or not len(i1.des_list):
        i1.detect_features(args.scale)
    if not len(i2.kp_list) or not len(i2.des_list):
        i2.detect_features(args.scale)

    (w, h) = i1.get_size()
    diag = int(math.sqrt(h*h + w*w))
    print("h:", h, "w:", w)
    print("scaled diag:", diag)
    
    rgb1 = i1.load_rgb()
    rgb2 = i2.load_rgb()

    FLANN_INDEX_KDTREE = 1
    flann_params = {
        'algorithm': FLANN_INDEX_KDTREE,
        'trees': 5
    }
    search_params = dict(checks=100)
    matcher = cv2.FlannBasedMatcher(flann_params, search_params)
    matches = matcher.knnMatch(i1.des_list, i2.des_list, k=3)
    print("Raw matches:", len(matches))
    
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
            p1 = np.float32(i1.kp_list[m[j].queryIdx].pt)
            p2 = np.float32(i2.kp_list[m[j].trainIdx].pt)
            raw_dist = np.linalg.norm(p2 - p1)
            # angle difference mapped to +/- 90
            a1 = np.array(i1.kp_list[m[j].queryIdx].angle)
            a2 = np.array(i2.kp_list[m[j].trainIdx].angle)
            angle_diff = abs((a1-a2+90) % 180 - 90)
            s1 = np.array(i1.kp_list[m[j].queryIdx].size)
            s2 = np.array(i2.kp_list[m[j].trainIdx].size)
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

    maxrange = int(diag*0.02)
    step = int(maxrange / 2)
    tol = int(diag*0.005)
    if tol < 5: tol = 5
    maxdist = int(diag*0.55)
    best_fitted_matches = 0
    match_bins = [[] for i in range(int(maxdist/step)+1)]
    print("bins:", len(match_bins))
    for line in match_stats:
        best_metric = line[3]
        best_dist = line[6]
        bin = int(round(best_dist / step))
        if bin < len(match_bins):
            match_bins[bin].append(line)
            if bin > 0:
                match_bins[bin-1].append(line)
            if bin < len(match_bins) - 1:
                match_bins[bin+1].append(line)
        
    for i, dist_matches in enumerate(match_bins):
        astep = 10
        print("bin:", i, "len:", len(dist_matches),
              "angles 0-90, step:", astep, )
        best_of_bin = 0
        for angle in range(0, 90, astep):
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
                    src.append( i1.kp_list[m.queryIdx].pt )
                    dst.append( i2.kp_list[m.trainIdx].pt )
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
                if len(matches_fit) > best_of_bin:
                       best_of_bin = len(matches_fit)
                if len(matches_fit) > best_fitted_matches:
                    best_fitted_matches = len(matches_fit)
                    print("Filtered matches:", len(angle_matches),
                          "Fitted matches:", len(matches_fit))
                    print("metric cutoff:", best_metric)
                    matches_dist = np.array(matches_dist)
                    print("avg match quality:", np.average(matches_dist))
                    print("max match quality:", np.max(matches_dist))
                    i1_new = cv2.warpPerspective(rgb1, H, (rgb1.shape[1], rgb1.shape[0]))
                    blend = cv2.addWeighted(i1_new, 0.5, rgb2, 0.5, 0)
                    blend = cv2.resize(blend, (int(w*args.scale), int(h*args.scale)))
                    cv2.imshow('blend', blend)
                    draw_inlier(rgb1, rgb2, i1.kp_list, i2.kp_list, matches_fit, 'ONLY_LINES', args.scale)
                       
        # check for diminishing returns and bail early
        print(best_fitted_matches, best_of_bin)
        if best_fitted_matches > 50 and best_of_bin < 10:
            break
                    
    cv2.waitKey()
