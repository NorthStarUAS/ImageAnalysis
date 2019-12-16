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
        lla1, ypr1, q1 = i1.get_aircraft_pose()
        lla2, ypr2, q2 = i2.get_aircraft_pose()
        yaw_diff = ypr1[0] - ypr2[0]
        if yaw_diff < -180: yaw_diff += 360
        if yaw_diff > 180: yaw_diff -= 360
        dist_list.append( [dist, yaw_diff, i, j] )
dist_list = sorted(dist_list, key=lambda fields: fields[0])

for line in dist_list:
    dist = line[0]
    yaw_diff = line[1]
    i = line[2]
    j = line[3]
    i1 = proj.image_list[i]
    i2 = proj.image_list[j]
    # print(i1.match_list)
    num_matches = len(i1.match_list[i2.name])
    print("dist: %.1f" % dist, "yaw: %.1f" % yaw_diff, i1.name, i2.name, num_matches)
    print("rev matches:", len(i2.match_list[i1.name]))
    if num_matches >= 25:
        continue

    if not len(i1.kp_list) or not len(i1.des_list):
        i1.detect_features(args.scale)
    if not len(i2.kp_list) or not len(i2.des_list):
        i2.detect_features(args.scale)

    w, h = camera.get_image_params()
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
    matches = matcher.knnMatch(i1.des_list, i2.des_list, k=2)
    print("Raw matches:", len(matches))

    # numpy rotation matrix
    c, s = np.cos(yaw_diff*np.pi/180), np.sin(yaw_diff*np.pi/180)
    R = np.matrix([[c, s], [-s, c]])
    print("R:", R)
    
    print("collect stats...")
    match_stats = []
    for i, m in enumerate(matches):
        best_index = -1
        best_metric = 9
        best_angle = 0
        best_size = 0
        best_dist = 0
        best_vangle = 0
        for j in range(len(m)):
            if m[j].distance >= 290:
                break
            ratio = m[0].distance / m[j].distance
            if ratio < ratio_cutoff:
                break
            p1 = np.float32(i1.kp_list[m[j].queryIdx].pt)
            p2 = np.float32(i2.kp_list[m[j].trainIdx].pt)
            v = p2 - p1
            raw_dist = np.linalg.norm(v)
            vangle = math.atan2(v[1], v[0])
            if vangle < 0: vangle += 2*math.pi
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
            if size_diff > 1.25:
                continue
            metric = size_diff / ratio
            #print(" ", j, m[j].distance, size_diff, metric)
            if best_index < 0 or metric < best_metric:
                best_metric = metric
                best_index = j
                best_angle = angle_diff
                best_size = size_diff
                best_dist = raw_dist
                best_vangle = vangle
        if best_index >= 0:
            #print(i, best_index, m[best_index].distance, best_size, best_metric)
            match_stats.append( [ m[best_index], best_index, ratio, best_metric,
                                  best_angle, best_size, best_dist, best_vangle ] )

    min_pairs = 25
    maxdist = int(diag*0.55)
    divs = 40
    step = maxdist / divs       # 0.1
    tol = int(diag*0.005)
    if tol < 5: tol = 5
    best_fitted_matches = 0
    dist_bins = [[] for i in range(divs + 1)]
    print("bins:", len(dist_bins))
    for line in match_stats:
        best_metric = line[3]
        best_dist = line[6]
        bin = int(round(best_dist / step))
        if bin < len(dist_bins):
            dist_bins[bin].append(line)
            if bin > 0:
                dist_bins[bin-1].append(line)
            if bin < len(dist_bins) - 1:
                dist_bins[bin+1].append(line)
        
    matches_fit = []
    for i, dist_matches in enumerate(dist_bins):
        print("bin:", i, "len:", len(dist_matches))
        best_of_bin = 0
        divs = 20
        step = 2*math.pi / divs
        angle_bins = [[] for i in range(divs + 1)]
        for line in dist_matches:
            match = line[0]
            vangle = line[7]
            bin = int(round(vangle / step))
            angle_bins[bin].append(match)
            if bin == 0:
                angle_bins[-1].append(match)
                angle_bins[bin+1].append(match)
            elif bin == divs:
                angle_bins[bin-1].append(match)
                angle_bins[0].append(match)
            else:
                angle_bins[bin-1].append(match)
                angle_bins[bin+1].append(match)
        for angle_matches in angle_bins:
            if len(angle_matches) >= min_pairs:
                src = []
                dst = []
                for m in angle_matches:
                    src.append( i1.kp_list[m.queryIdx].pt )
                    dst.append( i2.kp_list[m.trainIdx].pt )
                H, status = cv2.findHomography(np.array([src]).astype(np.float32),
                                               np.array([dst]).astype(np.float32),
                                               cv2.RANSAC,
                                               tol)
                num_fit = np.count_nonzero(status)
                if num_fit > best_of_bin:
                    best_of_bin = num_fit
                if num_fit > best_fitted_matches:
                    matches_fit = []
                    matches_dist = []
                    for i, m in enumerate(angle_matches):
                        if status[i]:
                            matches_fit.append(m)
                            matches_dist.append(m.distance)
                    best_fitted_matches = num_fit
                    print("Filtered matches:", len(angle_matches),
                          "Fitted matches:", num_fit)
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
