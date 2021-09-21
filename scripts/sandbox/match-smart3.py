#!/usr/bin/python3

# bin distance, bin vector angle

import argparse
import cv2
import math
import numpy as np
from tqdm import tqdm

from props import getNode

from lib import camera
from lib import project
from lib import srtm

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('project', help='project directory')
parser.add_argument('image1')
parser.add_argument('image2')
parser.add_argument('--scale', type=float, default=0.4, help='scale image before processing')
args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()
# proj.load_features(descriptors=False)
# proj.load_match_pairs()

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

# lookup ned reference
ref_node = getNode("/config/ned_reference", True)
ref = [ ref_node.getFloat('lat_deg'),
        ref_node.getFloat('lon_deg'),
        ref_node.getFloat('alt_m') ]
# setup SRTM ground interpolator
srtm.initialize( ref, 6000, 6000, 30 )

# print('Computing pair distances:')
# dist_list = []
# for i, i1 in enumerate(tqdm(proj.image_list)):
#     for j, i2 in enumerate(proj.image_list):
#         if j <= i:
#             continue
#         # temporary, just consider images with few matches
#         count = 0
#         for p in i1.match_list:
#             if len(i1.match_list[p]) >= 25:
#                 count += 1
#         if count >= 3:
#             continue
#         # camera pose distance check
#         ned1, ypr1, q1 = i1.get_camera_pose()
#         ned2, ypr2, q2 = i2.get_camera_pose()
#         dist = np.linalg.norm(np.array(ned2) - np.array(ned1))
#         lla1, ypr1, q1 = i1.get_aircraft_pose()
#         lla2, ypr2, q2 = i2.get_aircraft_pose()
#         yaw_diff = ypr1[0] - ypr2[0]
#         if yaw_diff < -180: yaw_diff += 360
#         if yaw_diff > 180: yaw_diff -= 360
#         dist_list.append( [dist, yaw_diff, i, j] )
# dist_list = sorted(dist_list, key=lambda fields: fields[0])

# common camera parameters
K = camera.get_K()
IK = np.linalg.inv(K)
dist_coeffs = camera.get_dist_coeffs()
w, h = camera.get_image_params()
diag = int(math.sqrt(h*h + w*w))
print("h:", h, "w:", w)
print("scaled diag:", diag)
grid_list = gen_grid(w, h, grid_steps)

i1 = proj.findImageByName(args.image1)
i2 = proj.findImageByName(args.image2)

# project a grid of uv coordinates from image 2 out onto the
# supposed ground plane.  Then back project these 3d world points
# into image 1 uv coordinates.  Compute an estimated 'ideal'
# homography relationship between the two images as a starting
# search point for feature matches.
    
proj_list = project.projectVectors( IK, i2.get_body2ned(),
                                    i2.get_cam2body(), grid_list )
ned1, ypr1, quat1 = i1.get_camera_pose(opt=False)
ned2, ypr2, quat2 = i2.get_camera_pose(opt=False)
g1 = srtm.ned_interp( [ned1[0], ned1[1]] )
g2 = srtm.ned_interp( [ned2[0], ned2[1]] )
ground = (g1 + g2) * 0.5
print("SRTM ground elevation:", ground)
    
pts_ned = project.intersectVectorsWithGroundPlane(ned2, ground, proj_list)
rvec1, tvec1 = i1.get_proj()
reproj_points, jac = cv2.projectPoints(np.array(pts_ned), rvec1, tvec1,
                                       K, dist_coeffs)
reproj_list = reproj_points.reshape(-1,2).tolist()
# print("reprojected points:", reproj_list)

print("Should filter points outside of 2nd image space here and now!")

affine, status = \
    cv2.estimateAffinePartial2D(np.array([reproj_list]).astype(np.float32),
                                np.array([grid_list]).astype(np.float32))
(rot, tx, ty, sx, sy) = decomposeAffine(affine)
print("Affine:")
print("Rotation (deg):", rot)
print("Translation (pixels):", tx, ty)
print("Skew:", sx, sy)

#if rot < 10:
#    continue

H, status = cv2.findHomography(np.array([reproj_list]).astype(np.float32),
                               np.array([grid_list]).astype(np.float32),
                               0)
print("Preliminary H:", H)

if not len(i1.kp_list) or not len(i1.des_list):
    i1.detect_features(args.scale)
if not len(i2.kp_list) or not len(i2.des_list):
    i2.detect_features(args.scale)

rgb1 = i1.load_rgb()
rgb2 = i2.load_rgb()

i1_new = cv2.warpPerspective(rgb1, H, (rgb1.shape[1], rgb1.shape[0]))
blend = cv2.addWeighted(i1_new, 0.5, rgb2, 0.5, 0)
blend = cv2.resize(blend, (int(w*args.scale), int(h*args.scale)))
cv2.imshow('blend', blend)
cv2.waitKey()

FLANN_INDEX_KDTREE = 1
flann_params = {
    'algorithm': FLANN_INDEX_KDTREE,
    'trees': 5
}
search_params = dict(checks=100)
matcher = cv2.FlannBasedMatcher(flann_params, search_params)
matches = matcher.knnMatch(i1.des_list, i2.des_list, k=2)
print("Raw matches:", len(matches))

best_fitted_matches = 20    # don't proceed if we can't beat this value

src_pts = np.float32([i1.kp_list[i].pt for i in range(len(i1.kp_list))]).reshape(-1, 1, 2)
dst_pts = np.float32([i2.kp_list[i].pt for i in range(len(i2.kp_list))]).reshape(-1, 1, 2)

while True:
    print('H:', H)
    trans_pts = cv2.perspectiveTransform(src_pts, H)

    print("collect stats...")
    match_stats = []
    for i, m in enumerate(matches):
        #print("dist:", m[0].distance)
        if m[0].distance >= 300:
            continue
        ratio = m[0].distance / m[1].distance
        #print("ratio:", ratio)
        if ratio > 0.9:
            continue
        p1 = trans_pts[m[0].queryIdx]
        p2 = dst_pts[m[0].trainIdx]
        #print(p1, p2)
        raw_dist = np.linalg.norm(p2 - p1)
        match_stats.append( [ m[0], raw_dist ] )
    #print(match_stats)
    min_pairs = 25
    tol = int(diag*0.005)
    if tol < 5: tol = 5

    done = True
    shifts = [ 0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000 ]
    for shift in shifts:
        print("shift:", shift)
        cutoffs = [ 16, 32, 64, 128, 256, 512, 1024, 2048 ]
        dist_bins = [[] for i in range(len(cutoffs))]
        print("bins:", len(dist_bins))
        for line in match_stats:
            m = line[0]
            best_metric = line[0]
            best_dist = line[1]
            for i, d in enumerate(cutoffs):
                if abs(best_dist-shift) < cutoffs[i]:
                    dist_bins[i].append(m)

        wait_for_key = False
        for i, dist_matches in enumerate(dist_bins):
            print("bin:", i, "cutoff:", cutoffs[i], "len:", len(dist_matches))
            if len(dist_matches) >= min_pairs:
                src = np.float32([src_pts[m.queryIdx] for m in dist_matches]).reshape(1, -1, 2)
                dst = np.float32([dst_pts[m.trainIdx] for m in dist_matches]).reshape(1, -1, 2)
                H_test, status = cv2.findHomography(src, dst, cv2.RANSAC, tol)
                num_fit = np.count_nonzero(status)
                print("  num_fit:",num_fit)
                if num_fit > best_fitted_matches:
                    done = False
                    matches_fit = []
                    matches_dist = []
                    # affine, astatus = \
                    #     cv2.estimateAffinePartial2D(np.array([src]).astype(np.float32),
                    #                                 np.array([dst]).astype(np.float32))
                    # (rot, tx, ty, sx, sy) = decomposeAffine(affine)
                    # print("Affine:")
                    # print("Rotation (deg):", rot)
                    # print("Translation (pixels):", tx, ty)
                    # print("Skew:", sx, sy)
                    H = np.copy(H_test)
                    print("H:", H)
                    for i, m in enumerate(dist_matches):
                        if status[i]:
                            matches_fit.append(m)
                            matches_dist.append(m.distance)
                    best_fitted_matches = len(matches_fit)
                    print("Filtered matches:", len(dist_matches),
                          "Fitted matches:", len(matches_fit))
                    print("metric cutoff:", best_metric)
                    matches_dist = np.array(matches_dist)
                    print("avg match quality:", np.average(matches_dist))
                    print("max match quality:", np.max(matches_dist))
                    i1_new = cv2.warpPerspective(rgb1, H, (rgb1.shape[1], rgb1.shape[0]))
                    blend = cv2.addWeighted(i1_new, 0.5, rgb2, 0.5, 0)
                    blend = cv2.resize(blend, (int(w*args.scale), int(h*args.scale)))
                    cv2.imshow('blend', blend)
                    wait_for_key = True
                    draw_inlier(rgb1, rgb2, i1.kp_list, i2.kp_list, matches_fit, 'ONLY_LINES', args.scale)

            # check for diminishing returns and bail early
            print(best_fitted_matches)
            if best_fitted_matches > 50:
                break

        if wait_for_key:
            cv2.waitKey()
    if done:
        break
