#!/usr/bin/python3

# put most of our eggs in the gms matching basket:
# https://github.com/JiawangBian/GMS-Feature-Matcher/blob/master/python/gms_matcher.py

import copy
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import random
import time

from props import getNode

from . import camera
from .find_obj import explore_match
from . import image_list
from .logger import log, qlog
from . import project
from . import smart
from . import transformations

detector_node = getNode('/config/detector', True)
matcher_node = getNode('/config/matcher', True)

detect_scale = 0.40
the_matcher = None
max_distance = None
min_pairs = 25

d2r = math.pi / 180.0

# the flann based matcher uses random starting points so some
# borderline matching results may change from one run to the next.
random.seed(time.time())

# Configure the matching session (setup the values in the property
# tree and call this function before any others.  Note, putting the
# parameters in the property tree simplifies the parameter list and
# lets us save a record of these in the config.json file
def configure():
    global detect_scale
    global the_matcher
    global max_distance
    global min_pairs

    detect_scale = detector_node.getFloat('scale')
    detector_str = detector_node.getString('detector')
    if detector_str == 'SIFT' or detector_str == 'SURF':
        norm = cv2.NORM_L2
        max_distance = 270.0
    elif detector_str == 'ORB' or detector_str == 'Star':
        norm = cv2.NORM_HAMMING
        max_distance = 64
    else:
        log("Detector not specified or not known:", detector_str)
        quit()

    # work around a feature/bug: flann enums don't exist
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH    = 6
    if norm == cv2.NORM_L2:
        flann_params = {
            'algorithm': FLANN_INDEX_KDTREE,
            'trees': 5
        }
    else:
        flann_params = {
            'algorithm': FLANN_INDEX_LSH,
            'table_number': 6,     # 12
            'key_size': 12,        # 20
            'multi_probe_level': 1 #2
        }
    search_params = {
        'checks': 100
    }
    the_matcher = cv2.FlannBasedMatcher(flann_params, search_params)
    min_pairs = matcher_node.getFloat('min_pairs')

# Iterate through all the matches for the specified image and
# delete keypoints that don't satisfy the homography (or
# fundamental) relationship.  Returns true if match set is clean, false
# if keypoints were removed.
#
# Notice: this tends to eliminate matches that aren't all on the
# same plane, so if the scene has a lot of depth, this could knock
# out a lot of good matches.
def filter_by_transform(K, i1, i2, transform):
    clean = True

    # tol = float(i1.width) / 200.0 # rejection range in pixels
    tol = math.pow(i1.width, 0.25)
    if tol < 1.0:
        tol = 1.0
    # print "tol = %.4f" % tol 
    matches = i1.match_list[i2.name]
    if len(matches) < min_pairs:
        i1.match_list[i2.name] = []
        return True
    p1 = []
    p2 = []
    for k, pair in enumerate(matches):
        use_raw_uv = False
        if use_raw_uv:
            p1.append( i1.kp_list[pair[0]].pt )
            p2.append( i2.kp_list[pair[1]].pt )
        else:
            # undistorted uv points should be better if the camera
            # calibration is known, right?
            p1.append( i1.uv_list[pair[0]] )
            p2.append( i2.uv_list[pair[1]] )

    p1 = np.float32(p1)
    p2 = np.float32(p2)
    #print "p1 = %s" % str(p1)
    #print "p2 = %s" % str(p2)
    method = cv2.RANSAC
    #method = cv2.LMEDS
    if transform == "homography":
        M, status = cv2.findHomography(p1, p2, method, tol)
    elif transform == "fundamental":
        M, status = cv2.findFundamentalMat(p1, p2, method, tol)
    elif transform == "essential":
        M, status = cv2.findEssentialMat(p1, p2, K, method, threshold=tol)
    elif transform == "none":
        status = np.ones(len(matches))
    else:
        # fail
        M, status = None, None
    log("  %s vs %s: %d / %d  inliers/matched" % (i1.name, i2.name, np.sum(status), len(status)))
    # remove outliers
    for k, flag in enumerate(status):
        if not flag:
            # print("    deleting: " + str(matches[k]))
            clean = False
            matches[k] = (-1, -1)
    for pair in reversed(matches):
        if pair == (-1, -1):
            matches.remove(pair)
    return clean

# return a count of unique matches
def count_unique(i1, i2, matches_fit):
    idx_pairs = []
    for m in matches_fit:
        idx_pairs.append( [m.queryIdx, m.trainIdx] )
    idx_pairs = filter_duplicates(i1, i2, idx_pairs)
    return len(idx_pairs)

    
# Filter duplicate features.  SIFT (for example) can detect the same
# feature at different scales/orientations which can lead to duplicate
# match pairs, or possibly one feature in image1 matching two or more
# features in images2.  Find and remove these from the set.
def filter_duplicates(i1, i2, idx_pairs):
    count = 0
    result = []
    kp1_dict = {}
    kp2_dict = {}
    for pair in idx_pairs:
        kp1 = i1.kp_list[pair[0]]
        kp2 = i2.kp_list[pair[1]]
        key1 = "%.2f-%.2f" % (kp1.pt[0], kp1.pt[1])
        key2 = "%.2f-%.2f" % (kp2.pt[0], kp2.pt[1])
        if key1 in kp1_dict and key2 in kp2_dict:
            # print("image1 and image2 key point already used:", key1, key2)
            count += 1
        elif key1 in kp1_dict:
            # print("image1 key point already used:", key1)
            count += 1
        elif key2 in kp2_dict:
            # print( "image2 key point already used:", key2)
            count += 1
        else:
            kp1_dict[key1] = True
            kp2_dict[key2] = True
            result.append(pair)
    if count > 0:
        qlog("  removed %d/%d duplicate features" % (count, len(idx_pairs)))
    return result

# iterate through idx_pairs1 and mark/remove any pairs that don't
# exist in idx_pairs2.  Then recreate idx_pairs2 as the inverse of
# idx_pairs1
def filter_cross_check(idx_pairs1, idx_pairs2):
    new1 = []
    new2 = []
    for k, pair in enumerate(idx_pairs1):
        rpair = [pair[1], pair[0]]
        for r in idx_pairs2:
            #print "%s - %s" % (rpair, r)
            if rpair == r:
                new1.append( pair )
                new2.append( rpair )
                break
    if len(idx_pairs1) != len(new1) or len(idx_pairs2) != len(new2):
        qlog("  cross check: (%d, %d) => (%d, %d)" % (len(idx_pairs1), len(idx_pairs2), len(new1), len(new2)))
    return new1, new2                                               

# run the knn matcher for the two sets of keypoints
def raw_matches(i1, i2, k=2):
    # sanity check
    if i1.des_list is None or i2.des_list is None:
        return []
    if len(i1.des_list.shape) == 0 or i1.des_list.shape[0] <= 1:
        return []
    if len(i2.des_list.shape) == 0 or i2.des_list.shape[0] <= 1:
        return []

    matches = the_matcher.knnMatch(np.array(i1.des_list),
                                   np.array(i2.des_list),
                                   k=k)
    qlog("  raw matches:", len(matches))
    return matches

def basic_pair_matches(i1, i2):
    matches = raw_matches(i1, i2)
    match_ratio = matcher_node.getFloat('match_ratio')
    sum = 0.0
    max_good = 0
    sum_good = 0.0
    count_good = 0
    for m in matches:
        sum += m[0].distance
        if m[0].distance <= m[1].distance * match_ratio:
            sum_good += m[0].distance
            count_good += 1
            if m[0].distance > max_good:
                max_good = m[0].distance
    qlog("  avg dist:", sum / len(matches))
    if count_good:
        qlog("  avg good dist:", sum_good / count_good, "(%d)" % count_good)
    qlog("  max good dist:", max_good)

    if False:
        # filter by absolute distance (for ORB, statistically all real
        # matches will have a distance < 64, for SIFT I don't know,
        # but I'm guessing anything more than 270.0 is a bad match.
        matches_thresh = []
        for m in matches:
            if m[0].distance < max_distance and m[0].distance <= m[1].distance * match_ratio:
                matches_thresh.append(m[0])
        qlog("  quality matches:", len(matches_thresh))

    if True:
        # generate a quality metric for each match, sort and only
        # pass along the best 'n' matches that pass the distance
        # ratio test.  (Testing the idea that 2000 matches aren't
        # better than 20 if they are good matches with respect to
        # optimizing the fit.)
        by_metric = []
        for m in matches:
            ratio = m[0].distance / m[1].distance # smaller is better
            metric = m[0].distance * ratio
            by_metric.append( [metric, m[0]] )
        by_metric = sorted(by_metric, key=lambda fields: fields[0])
        matches_thresh = []
        for line in by_metric:
            if line[0] < max_distance * match_ratio:
                matches_thresh.append(line[1])
        qlog("  quality matches:", len(matches_thresh))
        # fixme, make this a command line option or parameter?
        mymax = 2000
        if len(matches_thresh) > mymax:
            # clip list to n best rated matches
            matches_thresh = matches_thresh[:mymax]
            qlog("  clipping to:", mymax)

    if len(matches_thresh) < min_pairs:
        # just quit now
        return []

    w, h = camera.get_image_params()
    if not w or not h:
        log("Zero image sizes will crash matchGMS():", w, h)
        log("Recommend removing all meta/*.feat files and")
        log("rerun the matching step.")
        log("... or do some coding to add this information to the")
        log("ImageAnalysis/meta/<image_name>.json files")
        quit()
    size = (w, h)

    matchesGMS = cv2.xfeatures2d.matchGMS(size, size, i1.kp_list, i2.kp_list, matches_thresh, withRotation=True, withScale=False, thresholdFactor=5.0)
    #matchesGMS = cv2.xfeatures2d.matchGMS(size, size, i1.uv_list, i2.uv_list, matches_thresh, withRotation=True, withScale=False)
    #print('matchesGMS:', matchesGMS)

    idx_pairs = []
    for i, m in enumerate(matchesGMS):
        idx_pairs.append( [m.queryIdx, m.trainIdx] )

    # check for duplicate matches (based on different scales or attributes)
    idx_pairs = filter_duplicates(i1, i2, idx_pairs)
    qlog("  initial matches =", len(idx_pairs))
    if len(idx_pairs) < min_pairs:
        # so sorry
        return []
    else:
        return idx_pairs

# do initial feature matching (both ways) for the specified image
# pair.
def bidirectional_pair_matches(i1, i2, review=False):
    if i1 == i2:
        log("We shouldn't see this, but i1 == i2", i1.name, i2.name)
        return [], []

    # all vs. all match between overlapping i1 keypoints and i2
    # keypoints (forward match)
    idx_pairs1 = basic_pair_matches(i1, i2)
    if len(idx_pairs1) >= min_pairs:
        idx_pairs2 = basic_pair_matches(i2, i1)
    else:
        # save some time
        idx_pairs2 = []

    idx_pairs1, idx_pairs2 = filter_cross_check(idx_pairs1, idx_pairs2)

    if False:
        plot_matches(i1, i2, idx_pairs1)
        plot_matches(i2, i1, idx_pairs2)

    if review:
        if len(idx_pairs1):
            status, key = self.showMatchOrient(i1, i2, idx_pairs1)
            # remove deselected pairs
            for k, flag in enumerate(status):
                if not flag:
                    print("    deleting: " + str(idx_pairs1[k]))
                    idx_pairs1[k] = (-1, -1)
            for pair in reversed(idx_pairs1):
                if pair == (-1, -1):
                    idx_pairs1.remove(pair)

        if len(idx_pairs2):
            status, key = self.showMatchOrient(i2, i1, idx_pairs2)
            # remove deselected pairs
            for k, flag in enumerate(status):
                if not flag:
                    print("    deleting: " + str(idx_pairs2[k]))
                    idx_pairs2[k] = (-1, -1)
            for pair in reversed(idx_pairs2):
                if pair == (-1, -1):
                    idx_pairs2.remove(pair)

    return idx_pairs1, idx_pairs2

def gen_grid(w, h, steps):
    grid_list = []
    u_list = np.linspace(0, w, steps + 1)
    v_list = np.linspace(0, h, steps + 1)
    for v in v_list:
        for u in u_list:
            grid_list.append( [u, v] )
    return grid_list

def smart_pair_matches(i1, i2, review=False, est_rotation=False):
    # common camera parameters
    K = camera.get_K()
    IK = np.linalg.inv(K)
    dist_coeffs = camera.get_dist_coeffs()
    w, h = camera.get_image_params()
    diag = int(math.sqrt(h*h + w*w))
    print("h:", h, "w:", w, "diag:", diag)
    grid_steps = 8
    grid_list = gen_grid(w, h, grid_steps)

    # consider estimated yaw error and estimated surface elevation
    # from previous successful matches.
    
    if matcher_node.hasChild("ground_m"):
        ground_m = matcher_node.getFloat("ground_m")
        log("Forced ground:", ground_m)
    else:
        ground_m = smart.get_surface_estimate(i1, i2)
        # if ground_m is None:
        #     g1 = i1.node.getFloat("srtm_surface_m")
        #     g2 = i2.node.getFloat("srtm_surface_m")
        #     ground_m = (g1 + g2) * 0.5
        #     qlog("  SRTM ground (no triangulation yet): %.1f" % ground_m)
        # else:
        log("  Ground estimate: %.1f" % ground_m)
        
    i1_yaw_error = smart.get_yaw_error_estimate(i1)
    i2_yaw_error = smart.get_yaw_error_estimate(i2)
    # inherit partner yaw error if none computed yet.
    if abs(i1_yaw_error) < 0.0001 and abs(i2_yaw_error) > 0.0001:
        i1_yaw_error = i2_yaw_error
    if abs(i1_yaw_error) > 0.0001 and abs(i2_yaw_error) < 0.0001:
        i2_yaw_error = i1_yaw_error
    print("smart yaw errors:", i1_yaw_error, i2_yaw_error)
    R2 = transformations.rotation_matrix(i2_yaw_error*d2r, [1, 0, 0])[:3,:3]
    print("R2:\n", R2)
    
    match_ratio = matcher_node.getFloat("match_ratio")
    
    if review:
        rgb1 = i1.load_rgb()
        rgb2 = i2.load_rgb()

    # project a grid of uv coordinates from image 2 out onto the
    # supposed ground plane.  Then back project these 3d world points
    # into image 1 uv coordinates.  Compute an estimated 'ideal'
    # homography relationship between the two images as a starting
    # search point for feature matches.

    if est_rotation:
        print("body2ned:\n", i2.get_body2ned())
        smart_body2ned = np.dot(i2.get_body2ned(), R2)
        print("smart body2ned:\n", smart_body2ned)
    else:
        smart_body2ned = i2.get_body2ned()

    proj_list = project.projectVectors( IK, smart_body2ned,
                                        i2.get_cam2body(),
                                        grid_list )
    ned2, ypr2, quat2 = i2.get_camera_pose()
    if -ned2[2] > ground_m:
        ground_m = -ned2[2] - 2
    pts_ned = project.intersectVectorsWithGroundPlane(ned2, ground_m,
                                                      proj_list)
    if False and review:
        plot_list = []
        for p in pts_ned:
            plot_list.append( [p[1], p[0]] )
        plot_list = np.array(plot_list)
        plt.figure()
        plt.plot(plot_list[:,0], plot_list[:,1], 'ro')
        plt.show()
                            
    if est_rotation:
        rvec1, tvec1 = i1.get_proj(opt=False, yaw_error_est=i1_yaw_error)
    else:
        rvec1, tvec1 = i1.get_proj(opt=False)
    reproj_points, jac = cv2.projectPoints(np.array(pts_ned), rvec1, tvec1,
                                           K, dist_coeffs)
    reproj_list = reproj_points.reshape(-1,2).tolist()
    # print("reprojected points:", reproj_list)

    # print("Should filter points outside of 2nd image space here and now!")

    # affine, status = \
    #     cv2.estimateAffinePartial2D(np.array([reproj_list]).astype(np.float32),
    #                                 np.array([grid_list]).astype(np.float32))
    # (rot, tx, ty, sx, sy) = decomposeAffine(affine)
    # print("Affine:")
    # print("Rotation (deg):", rot)
    # print("Translation (pixels):", tx, ty)
    # print("Skew:", sx, sy)

    H, status = cv2.findHomography(np.array([reproj_list]).astype(np.float32),
                                   np.array([grid_list]).astype(np.float32),
                                   0)
    if review:
        # draw what we estimated
        print("Preliminary H:", H)
        i1_new = cv2.warpPerspective(rgb1, H, (rgb1.shape[1], rgb1.shape[0]))
        blend = cv2.addWeighted(i1_new, 0.5, rgb2, 0.5, 0)
        blend = cv2.resize(blend, (int(w*detect_scale), int(h*detect_scale)))
        cv2.imshow('blend', blend)
        print("Press a key:")
        cv2.waitKey()

    matches = raw_matches(i1, i2, k=3)
    print("Raw matches:", len(matches))

    best_fitted_matches = 20    # don't proceed if we can't beat this value
    matches_best = []
    
    src_pts = np.float32([i1.kp_list[i].pt for i in range(len(i1.kp_list))]).reshape(-1, 1, 2)
    dst_pts = np.float32([i2.kp_list[i].pt for i in range(len(i2.kp_list))]).reshape(-1, 1, 2)
    
    while True:
        # print('H:', H)
        trans_pts = cv2.perspectiveTransform(src_pts, H)

        print("collect stats...")
        match_stats = []
        for i, m in enumerate(matches):
            best_index = -1
            best_metric = 9
            best_angle = 0
            best_size = 0
            best_dist = 0
            for j in range(len(m)):
                if m[j].distance >= 300:
                    break
                ratio = m[0].distance / m[j].distance
                if ratio < match_ratio:
                    break
                p1 = trans_pts[m[j].queryIdx]
                p2 = dst_pts[m[j].trainIdx]
                #print(p1, p2)
                raw_dist = np.linalg.norm(p2 - p1)
                s1 = np.array(i1.kp_list[m[j].queryIdx].size)
                s2 = np.array(i2.kp_list[m[j].trainIdx].size)
                if s1 > s2:
                    size_diff = s1 / s2
                else:
                    size_diff = s2 / s1
                if size_diff > 1.25:
                    continue
                metric = raw_dist * size_diff / ratio
                #print(" ", j, m[j].distance, size_diff, metric)
                if best_index < 0 or metric < best_metric:
                    best_metric = metric
                    best_index = j
                    best_dist = raw_dist
            if best_index >= 0:
                match_stats.append( [ m[best_index], best_dist ] )

        tol = int(diag*0.005)
        if tol < 5: tol = 5

        cutoffs = [ 32, 64, 128, 256, 512, 1024, 2048 ]
        dist_bins = [[] for i in range(len(cutoffs))]
        print("bins:", len(dist_bins))
        for line in match_stats:
            m = line[0]
            best_dist = line[1]
            for i, d in enumerate(cutoffs):
                if best_dist < cutoffs[i]:
                    dist_bins[i].append(m)

        done = True
        for i, dist_matches in enumerate(dist_bins):
            print("bin:", i, "cutoff:", cutoffs[i], "len:", len(dist_matches))
            if len(dist_matches) >= min_pairs:
                src = np.float32([src_pts[m.queryIdx] for m in dist_matches]).reshape(1, -1, 2)
                dst = np.float32([dst_pts[m.trainIdx] for m in dist_matches]).reshape(1, -1, 2)
                H_test, status = cv2.findHomography(src, dst, cv2.RANSAC, tol)
                num_fit = np.count_nonzero(status)
                matches_fit = []
                matches_dist = []
                for i, m in enumerate(dist_matches):
                    if status[i]:
                        matches_fit.append(m)
                        matches_dist.append(m.distance)
                num_unique = count_unique(i1, i2, matches_fit)
                print(" fit:", num_fit, "unique:", num_unique)
                if num_unique > best_fitted_matches:
                    done = False
                    # affine, astatus = \
                    #     cv2.estimateAffinePartial2D(np.array([src]).astype(np.float32),
                    #                                 np.array([dst]).astype(np.float32))
                    # (rot, tx, ty, sx, sy) = decomposeAffine(affine)
                    # print("Affine:")
                    # print("Rotation (deg):", rot)
                    # print("Translation (pixels):", tx, ty)
                    # print("Skew:", sx, sy)
                    H = np.copy(H_test)
                    matches_best = list(matches_fit) # copy
                    # print("H:", H)
                    best_fitted_matches = num_unique
                    print("Filtered matches:", len(dist_matches),
                          "Fitted matches:", len(matches_fit),
                          "Unique matches:", num_unique)
                    #print("metric cutoff:", best_metric)
                    matches_dist = np.array(matches_dist)
                    print("avg match quality:", np.average(matches_dist))
                    print("max match quality:", np.max(matches_dist))
                    if review:
                        i1_new = cv2.warpPerspective(rgb1, H, (rgb1.shape[1], rgb1.shape[0]))
                        blend = cv2.addWeighted(i1_new, 0.5, rgb2, 0.5, 0)
                        blend = cv2.resize(blend, (int(w*detect_scale), int(h*detect_scale)))
                        #draw_inlier(rgb1, rgb2, i1.kp_list, i2.kp_list, matches_fit, 'ONLY_LINES', args.scale)
                        cv2.imshow('blend', blend)

            # check for diminishing returns and bail early
            #print(best_fitted_matches)
            #if best_fitted_matches > 50:
            #    break

        if review:
            print("Press a key:")
            cv2.waitKey()
            
        if done:
            break
        
    if len(matches_best) >= min_pairs:
        idx_pairs = []
        for m in matches_best:
            idx_pairs.append( [m.queryIdx, m.trainIdx] )
        idx_pairs = filter_duplicates(i1, i2, idx_pairs)
        if len(idx_pairs) >= min_pairs:
            rev_pairs = []
            for p in idx_pairs:
                rev_pairs.append( [p[1], p[0]] )
            qlog("  found matches =", len(idx_pairs))
            return idx_pairs, rev_pairs
    return [], []

def bruteforce_pair_matches(i1, i2, review=False):
    match_ratio = matcher_node.getFloat('match_ratio')
    w, h = camera.get_image_params()
    diag = int(math.sqrt(h*h + w*w))
    print("h:", h, "w:", w)
    print("scaled diag:", diag)

    if review:
        rgb1 = i1.load_rgb()
        rgb2 = i2.load_rgb()

    matches = raw_matches(i1, i2, k=3)
    
    qlog("  collect stats...")
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
            if ratio < match_ratio:
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

    maxdist = int(diag*0.55)
    maxrange = int(diag*0.02)
    divs = 40
    step = maxdist / divs       # 0.1
    tol = int(diag*0.005)
    if tol < 5: tol = 5
    best_fitted_matches = 0
    dist_bins = [[] for i in range(divs + 1)]
    print("bins:", len(dist_bins))
    for line in match_stats:
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
                    if review:
                        i1_new = cv2.warpPerspective(rgb1, H, (rgb1.shape[1], rgb1.shape[0]))
                        blend = cv2.addWeighted(i1_new, 0.5, rgb2, 0.5, 0)
                        blend = cv2.resize(blend, (int(w*detect_scale), int(h*detect_scale)))
                        cv2.imshow('blend', blend)
                        #draw_inlier(rgb1, rgb2, i1.kp_list, i2.kp_list, matches_fit, 'ONLY_LINES', detect_scale)
                       
        # check for diminishing returns and bail early
        print("bin:", i, "len:", len(dist_matches),
              best_fitted_matches, best_of_bin)
        if best_fitted_matches > 50 and best_of_bin < 10:
            break

    if review:
        cv2.waitKey()

    if len(matches_fit) >= min_pairs:
        idx_pairs = []
        for m in matches_fit:
            idx_pairs.append( [m.queryIdx, m.trainIdx] )
        idx_pairs = filter_duplicates(i1, i2, idx_pairs)
        if len(idx_pairs) >= min_pairs:
            rev_pairs = []
            for p in idx_pairs:
                rev_pairs.append( [p[1], p[0]] )
            qlog("  initial matches =", len(idx_pairs))
            return idx_pairs, rev_pairs
    return [], []
    
def find_matches(proj, K, strategy="smart", transform="homography",
                 sort=False, review=False):
    n = len(proj.image_list) - 1
    n_work = float(n*(n+1)/2)
    t_start = time.time()

    intervals = []
    for i in range(len(proj.image_list)-1):
        ned1, ypr1, q1 = proj.image_list[i].get_camera_pose()
        ned2, ypr2, q2 = proj.image_list[i+1].get_camera_pose()
        dist = np.linalg.norm(np.array(ned2) - np.array(ned1))
        intervals.append(dist)
        print(i, dist)
    median = np.median(intervals)
    log("Median pair interval: %.1f m" % median)
    median_int = int(round(median))
    if median_int == 0:
        median_int = 1

    if matcher_node.hasChild("min_dist"):
        min_dist = matcher_node.getFloat("min_dist")
    else:
        min_dist = 0
    if matcher_node.hasChild("max_dist"):
        max_dist = matcher_node.getFloat("max_dist")
    else:
        max_dist = median_int * 4

    log('Generating work list for range:', min_dist, '-', max_dist)
    work_list = []
    for i, i1 in enumerate(tqdm(proj.image_list, smoothing=0.05)):
        ned1, ypr1, q1 = i1.get_camera_pose()
        for j, i2 in enumerate(proj.image_list):
            if j <= i:
                continue
            # camera pose distance check
            ned2, ypr2, q2 = i2.get_camera_pose()
            dist = np.linalg.norm(np.array(ned2) - np.array(ned1))
            dist = int(round(dist/median_int))*median_int # discretized sorting/cache friendlier
            if dist >= min_dist and dist <= max_dist:
                work_list.append( [dist, i, j] )

    if sort:
        # (optional) sort worklist from closest pairs to furthest pairs
        # (caution, this is less cache friendly, but hopefully mitigated
        # a bit by the discritized sorting scheme.)
        #
        # benefits of sorting by distance: more important work is done
        # first (chance to quit early)
        #
        # benefits of sorting by order: for large memory usage, active
        # memory pool decreases as work progresses (becoming more and
        # more system friendly as the match progresses.)
        work_list = sorted(work_list, key=lambda fields: fields[0])

    # note: image.desc_timestamp is used to unload not recently
    # used descriptors ... these burn a ton of memory so unloading
    # things not recently used should help our memory foot print
    # at hopefully not too much of a performance expense.

    # process the work list
    n_count = 0
    save_time = time.time()
    save_interval = 300     # seconds
    log("Processing worklist matches:")
    for line in tqdm(work_list, smoothing=0.05):
        dist = line[0]
        i = line[1]
        j = line[2]
        i1 = proj.image_list[i]
        i2 = proj.image_list[j]

        # eta estimation
        percent = n_count / float(len(work_list))
        n_count += 1
        t_elapsed = time.time() - t_start
        if percent > 0:
            t_end = t_elapsed / percent
        else:
            t_end = t_start
        t_remain = t_end - t_elapsed

        # skip if match has already been computed
        if i2.name in i1.match_list and i1.name in i2.match_list:
            if (True or strategy == "smart" or strategy == "bruteforce") and len(i1.match_list[i2.name]) == 0:
                log("Retrying: ", i1.name, "vs", i2.name, "(no matches found previously)")
            else:
                log("Skipping: ", i1.name, "vs", i2.name, "already done.")
                continue

        msg = "Matching %s vs %s - %.1f%% done: " % (i1.name, i2.name, percent * 100.0)
        if t_remain < 3600:
            msg += "%.1f (min)" % (t_remain / 60.0)
        else:
            msg += "%.1f (hr)" % (t_remain / 3600.0)
        qlog(msg)
        qlog("  separation (approx) = %.0f (m)" % dist)

        # update cache timers and make sure features/descriptors are loaded
        i1.desc_timestamp = time.time()
        i2.desc_timestamp = time.time()
        if i1.kp_list is None or i1.des_list is None or not len(i1.kp_list) or not len(i1.des_list):
            i1.detect_features(detect_scale)
        if i2.kp_list is None or i2.des_list is None or not len(i2.kp_list) or not len(i2.des_list):
            i2.detect_features(detect_scale)

        if strategy == "smart":
            review = False
            #match_fwd, match_rev = smart_pair_matches(i1, i2, review, False)
            match_fwd, match_rev = smart_pair_matches(i1, i2, review, True)
        elif strategy == "traditional":
            match_fwd, match_rev = bidirectional_pair_matches(i1, i2, review)
        elif strategy == "bruteforce":
            match_fwd, match_rev = bruteforce_pair_matches(i1, i2)
        i1.match_list[i2.name] = match_fwd
        i2.match_list[i1.name] = match_rev

        # mark these images as matches_dirty
        i1.matches_clean = False
        i2.matches_clean = False

        # update surface triangulation (estimate)
        avg, std = smart.update_surface_estimate(i1, i2)
        if avg and std:
            qlog(" ", i1.name, i2.name, "surface est: %.1f" % avg, "std: %.1f" % std)
        yaw1_error = smart.update_yaw_error_estimate(i1, i2)
        i1.set_aircraft_yaw_error_estimate(yaw1_error)
        yaw2_error = smart.update_yaw_error_estimate(i2, i1)
        i2.set_aircraft_yaw_error_estimate(yaw2_error)

        # new feature, depends on a reasonably quality initial camera
        # pose!  caution: I've put a policy setting here in the middle
        # of a capability for initial testing.  if we find a match,
        # but the std dev of the altitude of the triangulated features
        # > 25 (m) then we think this is a bad match and we delete the
        # pairs.
        if std and std >= 50 and len(i1.match_list[i2.name]) < 100:
            log("Std dev of surface triangulation blew up, matches are probably bad so discarding them!", i1.name, i2.name, "avg:", avg, "std:", std, "count:", len(match_fwd))
            #showMatchOrient(i1, i2, i1.match_list[i2.name])
            i1.match_list[i2.name] = []
            i2.match_list[i1.name] = []
            
        # save our work so far, and flush descriptor cache
        if time.time() >= save_time + save_interval:
            log('saving matches and image meta data ...')
            saveMatches(proj.image_list, check_if_dirty=True)
            smart.save(proj.analysis_dir)
            save_time = time.time()
            time_list = []
            for i3 in proj.image_list:
                if not i3.des_list is None:
                    time_list.append( [i3.desc_timestamp, i3] )
            time_list = sorted(time_list, key=lambda fields: fields[0],
                               reverse=True)
            # may wish to monitor and update cache_size formula
            cache_size = 20 + 5 * (int(math.sqrt(len(proj.image_list))) + 1)
            flush_list = time_list[cache_size:]
            qlog("flushing keypoint/descriptor cache - size: %d (over by: %d)" % (cache_size, len(flush_list)) )
            for line in flush_list:
                qlog('  clearing descriptors for:', line[1].name)
                line[1].kp_list = None
                line[1].des_list = None
                line[1].uv_list = None

    # and the final save
    saveMatches(proj.image_list)
    smart.save(proj.analysis_dir)
    print('Pair-wise matches successfully saved.')

def saveMatches(image_list, check_if_dirty=False):
    for image in image_list:
        if check_if_dirty:
            if not image.matches_clean:
                image.save_matches()
        else:
            image.save_matches()

# for visualizing matches
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

# pasted from stackoverflow.com ....
def rotateAndScale(img, degreesCCW=30, scaleFactor=1.0 ):
    (oldY,oldX) = img.shape[:2]

    # rotate about center of image.
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW,
                                scale=scaleFactor)

    # choose a new image size.
    newX, newY = oldX * scaleFactor, oldY * scaleFactor

    # include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX, newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),
                  abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the
    # center of that region.
    (tx, ty) = ((newX-oldX)/2, (newY-oldY)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
    return rotatedImg, M

def copyKeyPoint(k):
    return cv2.KeyPoint(x=k.pt[0], y=k.pt[1],
                        _size=k.size, _angle=k.angle,
                        _response=k.response, _octave=k.octave,
                        _class_id=k.class_id)
    
def showMatchOrient(i1, i2, idx_pairs, status=None, orient='relative'):
    #print " -- idx_pairs = " + str(idx_pairs)
    img1 = i1.load_gray()
    img2 = i2.load_gray()

    # compute the affine transformation between points.  This is
    # used to determine relative orientation of the two images,
    # and possibly estimate outliers if no status array is
    # provided.

    src = []
    dst = []
    for pair in idx_pairs:
        src.append( i1.kp_list[pair[0]].pt )
        dst.append( i2.kp_list[pair[1]].pt )
    affine, status = \
        cv2.estimateAffinePartial2D(np.array([src]).astype(np.float32),
                                    np.array([dst]).astype(np.float32))
    print('affine:', affine)
    if affine is None:
        affine = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
        # auto reject any pairs where we can't determine a proper affine transformation at all
        status = np.zeros(len(idx_pairs), np.bool_)
        return status, ord(' ')
    (rot, tx, ty, sx, sy) = decomposeAffine(affine)
    print(' ', rot, tx, ty, sx, sy)

    if status is None:
        status = np.ones(len(idx_pairs), np.bool_)
        # for each src point, compute dst_est[i] = src[i] * affine
        error = []
        for i, p in enumerate(src):
            p_est = affine.dot( np.hstack((p, 1.0)) )[:2]
            # print('p est:', p_est, 'act:', dst[i])
            #np1 = np.array(i1.coord_list[pair[0]])
            #np2 = np.array(i2.coord_list[pair[1]])
            d = np.linalg.norm(p_est - dst[i])
            # print('dist:', d)
            error.append(d)
        # print('errors:', error)
        error = np.array(error)
        avg = np.mean(error)
        std = np.std(error)
        print('avg:', avg, 'std:', std)

        # mark the potential outliers
        for i in range(len(idx_pairs)):
            if error[i] > avg + 3*std:
                status[i] = False

    print('orientation:', orient)
    if orient == 'relative':
        # estimate relative orientation between features
        yaw1 = 0
        yaw2 = rot
    elif orient == 'aircraft':
        yaw1 = i1.aircraft_pose['ypr'][0]
        yaw2 = i2.aircraft_pose['ypr'][0]
    elif orient == 'camera':
        yaw1 = i1.camera_pose['ypr'][0]
        yaw2 = i2.camera_pose['ypr'][0]
    elif orient == 'sba':
        yaw1 = i1.camera_pose_sba['ypr'][0]
        yaw2 = i2.camera_pose_sba['ypr'][0]
    else:
        yaw1 = 0.0
        yaw2 = 0.0
    print( yaw1, yaw2)
    h, w = img1.shape[:2]
    scale = 790.0/float(w)
    si1, M1 = rotateAndScale(img1, yaw1, scale)
    si2, M2 = rotateAndScale(img2, yaw2, scale)

    kp_pairs = []
    for p in idx_pairs:
        kp1 = copyKeyPoint(i1.kp_list[p[0]])
        p1 = M1.dot( np.hstack((kp1.pt, 1.0)) )[:2]
        kp1.pt = (p1[0], p1[1])
        kp2 = copyKeyPoint(i2.kp_list[p[1]])
        p2 = M2.dot( np.hstack((kp2.pt, 1.0)) )[:2]
        kp2.pt = (p2[0], p2[1])
        # print p1, p2
        kp_pairs.append( (kp1, kp2) )

    key = explore_match('find_obj', si1, si2, kp_pairs,
                        hscale=1.0, wscale=1.0, status=status)

    # status structure represents in/outlier choices of user.
    # explore_match() modifies the status array in place.

    cv2.destroyAllWindows()

    # status is an array of booleans that parallels the pair array
    # and represents the users choice to keep or discard the
    # respective pairs.
    return status, key


#########################################################################
###   STUFF BELOW HERE IS OLD, RARELY USED, AND POSSIBLY DEPRECATED   ###
#########################################################################

class Matcher():
    def __init__(self):
        pass

    def plot_matches(self, i1, i2, idx_pairs):
        # This can be plotted in gnuplot with:
        # plot "c1.txt", "c2.txt", "vector.txt" u 1:2:($3-$1):($4-$2) title "pairs" with vectors
        f = open('c1.txt', 'w')
        for c1 in i1.coord_list:
            f.write("%.3f %.3f %.3f\n" % (c1[1], c1[0], -c1[2]))
        f.close()
        f = open('c2.txt', 'w')
        for c2 in i2.coord_list:
            f.write("%.3f %.3f %.3f\n" % (c2[1], c2[0], -c2[2]))
        f.close()
        f = open('vector.txt', 'w')
        for pair in idx_pairs:
            c1 = i1.coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            f.write("%.3f %.3f %.3f %.3f %.3f %.3f\n" % ( c2[1], c2[0], -c2[2], c1[1], c1[0], -c1[2] ))
        f.close()
        # a = raw_input("Press Enter to continue...")
        
    def showMatch(self, i1, i2, idx_pairs, status=None):
        #print " -- idx_pairs = " + str(idx_pairs)
        kp_pairs = []
        for p in idx_pairs:
            kp1 = i1.kp_list[p[0]]
            kp2 = i2.kp_list[p[1]]
            kp_pairs.append( (kp1, kp2) )
        img1 = i1.load_gray()
        img2 = i2.load_gray()
        if status == None:
            status = np.ones(len(kp_pairs), np.bool_)
        h, w = img1.shape[:2]
        scale = 790.0/float(w)
        si1 = cv2.resize(img1, (0,0), fx=scale, fy=scale)
        si2 = cv2.resize(img2, (0,0), fx=scale, fy=scale)
        explore_match('find_obj', si1, si2, kp_pairs,
                      hscale=scale, wscale=scale, status=status)
        # status structure will be correct here and represent
        # in/outlier choices of user
        cv2.destroyAllWindows()

        # status is an array of booleans that parallels the pair array
        # and represents the users choice to keep or discard the
        # respective pairs.
        return status

    def showMatches(self, i1):
        for key in i1.match_list:
            idx_pairs = i1.match_list[key]
            if len(idx_pairs) >= min_pairs:
                i2 = self.findImageByName(key)
                print("Showing matches for image %s and %s" % (i1.name, i2.name))
                self.showMatch( i1, i2, idx_pairs )

    def showAllMatches(self):
        # O(n,n) compare
        for i, i1 in enumerate(self.image_list):
            self.showMatches(i1)

    # compute the error between a pair of images
    def imagePairError(self, i, alt_coord_list, j, match, emax=False):
        i1 = self.image_list[i]
        i2 = self.image_list[j]
        #print "%s %s" % (i1.name, i2.name)
        coord_list = i1.coord_list
        if alt_coord_list != None:
            coord_list = alt_coord_list
        emax_value = 0.0
        dist2_sum = 0.0
        error_sum = 0.0
        for pair in match:
            c1 = coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            dist2 = dx*dx + dy*dy
            dist2_sum += dist2
            error = math.sqrt(dist2)
            if emax_value < error:
                emax_value = error
        if emax:
            return emax_value
        else:
            return math.sqrt(dist2_sum / len(match))

    # considers only total distance between points (thanks to Knuth
    # and Welford)
    def imagePairVariance1(self, i, alt_coord_list, j, match):
        i1 = self.image_list[i]
        i2 = self.image_list[j]
        coord_list = i1.coord_list
        if alt_coord_list != None:
            coord_list = alt_coord_list
        mean = 0.0
        M2 = 0.0
        n = 0
        for pair in match:
            c1 = coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            n += 1
            x = math.sqrt(dx*dx + dy*dy)
            delta = x - mean
            mean += delta/n
            M2 = M2 + delta*(x - mean)

        if n < 2:
            return 0.0
 
        variance = M2/(n - 1)
        return variance

    # considers x, y errors separated (thanks to Knuth and Welford)
    def imagePairVariance2(self, i, alt_coord_list, j, match):
        i1 = self.image_list[i]
        i2 = self.image_list[j]
        coord_list = i1.coord_list
        if alt_coord_list != None:
            coord_list = alt_coord_list
        xsum = 0.0
        ysum = 0.0
        # pass 1, compute x and y means
        for pair in match:
            c1 = coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            xsum += dx
            ysum += dy
        xmean = xsum / len(match)
        ymean = ysum / len(match)

        # pass 2, compute average error in x and y
        xsum2 = 0.0
        ysum2 = 0.0
        for pair in match:
            c1 = coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            ex = xmean - dx
            ey = ymean - dy
            xsum2 += ex * ex
            ysum2 += ey * ey
        xerror = math.sqrt( xsum2 / len(match) )
        yerror = math.sqrt( ysum2 / len(match) )
        return xerror*xerror + yerror*yerror

    # Compute an error metric related to image placement among the
    # group.  If an alternate coordinate list is provided, that is
    # used to compute the error metric (useful for doing test fits.)
    # if max=True then return the maximum pair error, not the weighted
    # average error
    def imageError(self, i, alt_coord_list=None, method="average",
                   variance=False, max=False):
        if method == "average":
            variance = False
            emax = False
        elif method == "stddev":
            variance = True
            emax = False
        elif method == "max":
            variance = False
            emax = True

        i1 = self.image_list[i]
        emax_value = 0.0
        dist2_sum = 0.0
        var_sum = 0.0
        weight_sum = i1.weight  # give ourselves an appropriate weight
        i1.num_matches = 0
        for key in enumerate(i1.match_list):
            matches = i1.match_list[key]
            if len(matches):
                i1.num_matches += 1
                i2 = self.findImageByName[key]
                #print "Matching %s vs %s " % (i1.name, i2.name)
                error = 0.0
                if variance:
                    var = self.imagePairVariance2(i, alt_coord_list, j, matches)
                    #print "  %s var = %.2f" % (i1.name, var)
                    var_sum += var * i2.weight
                else:
                    error = self.imagePairError(i, alt_coord_list, j,
                                                matches, emax)
                    dist2_sum += error * error * i2.weight
                weight_sum += i2.weight
                if emax_value < error:
                    emax_value = error
        if emax:
            return emax_value
        elif variance:
            #print "  var_sum = %.2f  weight_sum = %.2f" % (var_sum, weight_sum)
            return math.sqrt(var_sum / weight_sum)
        else:
            return math.sqrt(dist2_sum / weight_sum)

    # delete the pair (and it's reciprocal if it can be found)
    # i = image1 index, j = image2 index
    def deletePair(self, i, j, pair):
        i1 = self.image_list[i]
        i2 = self.image_list[j]
        #print "%s v. %s" % (i1.name, i2.name)
        #print "i1 pairs before = %s" % str(i1.match_list[j])
        #print "pair = %s" % str(pair)
        i1.match_list[i2.name].remove(pair)
        #print "i1 pairs after = %s" % str(i1.match_list[j])
        pair_rev = (pair[1], pair[0])
        #print "i2 pairs before = %s" % str(i2.match_list[i])
        i2.match_list[i1.name].remove(pair_rev)
        #print "i2 pairs after = %s" % str(i2.match_list[i])

    # compute the error between a pair of images
    def pairErrorReport(self, i, alt_coord_list, j, minError):
        i1 = self.image_list[i]
        i2 = self.image_list[j]
        match = i1.match_list[i2.name]
        print("pair error report %s v. %s (%d)" % (i1.name, i2.name, len(match)))
        if len(match) == 0:
            return
        report_list = []
        coord_list = i1.coord_list
        if alt_coord_list != None:
            coord_list = alt_coord_list
        dist2_sum = 0.0
        for k, pair in enumerate(match):
            c1 = coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            dist2 = dx*dx + dy*dy
            dist2_sum += dist2
            error = math.sqrt(dist2)
            report_list.append( (error, k) )
        report_list = sorted(report_list,
                             key=lambda fields: fields[0],
                             reverse=True)

        # meta stats on error values
        error_avg = math.sqrt(dist2_sum / len(match))
        stddev_sum = 0.0
        for line in report_list:
            error = line[0]
            stddev_sum += (error_avg-error)*(error_avg-error)
        stddev = math.sqrt(stddev_sum / len(match))
        print("   error avg = %.2f stddev = %.2f" % (error_avg, stddev))

        # compute best estimation of valid vs. suspect pairs
        dirty = False
        if error_avg >= minError:
            dirty = True
        if minError < 0.1:
            dirty = True
        status = np.ones(len(match), np.bool_)
        for line in report_list:
            if line[0] > 50.0 or line[0] > (error_avg + 2*stddev):
                status[line[1]] = False
                dirty = True

        if dirty:
            status = self.showMatch(i1, i2, match, status)
            delete_list = []
            for k, flag in enumerate(status):
                if not flag:
                    print("    deleting: " + str(match[k]))
                    #match[i] = (-1, -1)
                    delete_list.append(match[k])

        if False: # for line in report_list:
            print("    %.1f %s" % (line[0], str(match[line[1]])))
            if line[0] > 50.0 or line[0] > (error_avg + 5*stddev):
                # if positional error > 50m delete pair
                done = False
                while not done:
                    print("Found a suspect match: d)elete v)iew [o]k: ",)
                    reply = find_getch()
                    print
                    if reply == 'd':
                        match[line[1]] = (-1, -1)
                        dirty = True
                        done = True;
                        print("    (deleted) " + str(match[line[1]]))
                    elif reply == 'v':
                        self.showMatch(i1, i2, match, line[1])
                    else:
                        done = True

        if dirty:
            # update match list to remove the marked pairs
            #print "before = " + str(match)
            #for pair in reversed(match):
            #    if pair == (-1, -1):
            #        match.remove(pair)
            for pair in delete_list:
                self.deletePair(i, j, pair)
            #print "after = " + str(match)

    def findImageIndex(self, search):
        for i, image in enumerate(self.image_list):
            if search == image:
                return i
        return None

    def findImageByName(self, search):
        for image in self.image_list:
            if search == image.name:
                return image
        return None

    # fuzz factor increases (decreases) the ransac tolerance and is in
    # pixel units so it makes sense to bump this up or down in integer
    # increments.
    def reviewFundamentalErrors(self, fuzz_factor=1.0, interactive=True):
        total_removed = 0

        # Test fundametal matrix constraint
        for i, i1 in enumerate(self.image_list):
            # rejection range in pixels
            tol = float(i1.width) / 800.0 + fuzz_factor
            print("tol = %.4f" % tol)
            if tol < 0.0:
                tol = 0.0
            for key in i1.match_list:
                matches = i1.match_list[key]
                i2 = self.findImageByName[key]
                if i1.name == i2.name:
                    continue
                if len(matches) < min_pairs:
                    i1.match_list[i2.name] = []
                    continue
                p1 = []
                p2 = []
                for k, pair in enumerate(matches):
                    p1.append( i1.kp_list[pair[0]].pt )
                    p2.append( i2.kp_list[pair[1]].pt )

                p1 = np.float32(p1)
                p2 = np.float32(p2)
                #print "p1 = %s" % str(p1)
                #print "p2 = %s" % str(p2)
                M, status = cv2.findFundamentalMat(p1, p2, cv2.RANSAC, tol)

                size = len(status)
                inliers = np.sum(status)
                print('  %s vs %s: %d / %d  inliers/matched' \
                    % (i1.name, i2.name, inliers, size))

                if inliers < size:
                    total_removed += (size - inliers)
                    if interactive:
                        status = self.showMatch(i1, i2, matches, status)

                    delete_list = []
                    for k, flag in enumerate(status):
                        if not flag:
                            print("    deleting: " + str(matches[k]))
                            #match[i] = (-1, -1)
                            delete_list.append(matches[k])

                    for pair in delete_list:
                        self.deletePair(i, j, pair)
        return total_removed

    # return true if point set is pretty close to linear
    def isLinear(self, points, threshold=20.0):
        x = []
        y = []
        for pt in points:
            x.append(pt[0])
            y.append(pt[1])
        z = np.polyfit(x, y, 1) 
        p = np.poly1d(z)
        sum = 0.0
        for pt in points:
            e = pt[1] - p(pt[0])
            sum += e*e
        return math.sqrt(sum) <= threshold

    # look for linear/degenerate match sets
    def reviewLinearSets(self, threshold=20.0):
        # Test fundametal matrix constraint
        for i, i1 in enumerate(self.image_list):
            for key in i1.match_list:
                matches = i1.match_list[key]
                i2 = self.findImageByName[key]
                if i1.name == i2.name:
                    continue
                if len(matches) < min_pairs:
                    i1.match_list[i2.name] = []
                    continue
                pts = []
                status = []
                for k, pair in enumerate(matches):
                    pts.append( i1.kp_list[pair[0]].pt )
                    status.append(False)

                # check for degenerate case of all matches being
                # pretty close to a straight line
                if self.isLinear(pts, threshold):
                    print("%s vs %s is a linear match, probably should discard" % (i1.name, i2.name))
                    
                    status = self.showMatch(i1, i2, matches, status)

                    delete_list = []
                    for k, flag in enumerate(status):
                        if not flag:
                            print("    deleting: " + str(matches[k]))
                            #match[i] = (-1, -1)
                            delete_list.append(matches[k])

                    for pair in delete_list:
                        self.deletePair(i, j, pair)

    # Review matches by fewest pairs of keypoints.  The fewer
    # keypoints that join a pair of images, the greater the chance
    # that we could have stumbled on a degenerate or incorrect set.
    def reviewByFewestPairs(self, maxpairs=8):
        print("Review matches by fewest number of pairs")
        if len(self.image_list):
            report_list = []
            for i, image in enumerate(self.image_list):
                for key in image.match_list:
                    matches = image.match_list[key]
                    e = len(matches)
                    if e > 0 and e <= maxpairs:
                        report_list.append( (e, i, key) )
            report_list = sorted(report_list, key=lambda fields: fields[0],
                                 reverse=False)
            # show images sorted by largest positional disagreement first
            for line in report_list:
                i1 = self.image_list[line[1]]
                i2 = self.findImageByName(line[2])
                pairs = i1.match_list[line[2]]
                if len(pairs) == 0:
                    # probably already deleted by the other match order
                    continue
                print("showing %s vs %s: %d pairs" \
                    % (i1.name, i2.name, len(pairs)))
                status = self.showMatch(i1, i2, pairs)
                delete_list = []
                for k, flag in enumerate(status):
                    if not flag:
                        print("    deleting: " + str(pairs[k]))
                        #match[i] = (-1, -1)
                        delete_list.append(pairs[k])
                for pair in delete_list:
                    self.deletePair(line[1], line[2], pair)

    # sort and review match pairs by worst positional error
    def matchErrorReport(self, i, minError=20.0):
        i1 = self.image_list[i]
        # now for each image, find/show worst individual matches
        report_list = []
        for key in i1.match_list:
            matches = i1.match_list[key]
            if len(matches):
                i2 = self.findImageByName[key]
                #print "Matching %s vs %s " % (i1.name, i2.name)
                e = self.imagePairError(i, None, key, matches)
                if e > minError:
                    report_list.append( (e, i, key) )

        report_list = sorted(report_list,
                             key=lambda fields: fields[0],
                             reverse=True)

        for line in report_list:
            i1 = self.image_list[line[1]]
            i2 = self.findImageByName(line[2])
            print("min error = %.3f" % minError)
            print("%.1f %s %s" % (line[0], i1.name, i2.name))
            if line[0] > minError:
                #print "  %s" % str(pairs)
                self.pairErrorReport(line[1], None, line[2], minError)
                #print "  after %s" % str(match)
                #self.showMatch(i1, i2, match)

    # sort and review all match pairs by worst positional error
    def fullMatchErrorReport(self, minError=20.0):
        report_list = []
        for i, image in enumerate(self.image_list):
            i1 = self.image_list[i]
            # now for each image, find/show worst individual matches
            for key in i1.match_list:
                matches = i1.match_list[key]
                if len(matches):
                    i2 = self.findImageByName(key)
                    #print "Matching %s vs %s " % (i1.name, i2.name)
                    e = self.imagePairError(i, None, key, matches)
                    if e > minError:
                        report_list.append( (e, i, key) )

        report_list = sorted(report_list,
                             key=lambda fields: fields[0],
                             reverse=True)

        for line in report_list:
            i1 = self.image_list[line[1]]
            i2 = self.findImageByName(line[2])
            print("min error = %.3f" % minError)
            print("%.1f %s %s" % (line[0], i1.name, i2.name))
            if line[0] > minError:
                #print "  %s" % str(pairs)
                self.pairErrorReport(line[1], None, line[2], minError)
                #print "  after %s" % str(match)
                #self.showMatch(i1, i2, match)

    def reviewPoint(self, lon_deg, lat_deg, ref_lon, ref_lat):
        (x, y) = image_list.wgs842cart(lon_deg, lat_deg, ref_lon, ref_lat)
        print("Review images touching %.2f %.2f" % (x, y))
        review_list = image_list.getImagesCoveringPoint(self.image_list, x, y, pad=25.0, only_placed=False)
        #print "  Images = %s" % str(review_list)
        for image in review_list:
            print("    %s -> " % image.name,)
            for key in image.match_list:
                matches = image.match_list[key]
                if len(matches):
                    print("%s (%d) " % (self.image_list[j].name, len(matches)),)
            print
            r2 = image.coverage()
            p = image_list.getImagesCoveringRectangle(self.image_list, r2)
            p_names = []
            for i in p:
                p_names.append(i.name)
            print("      possible matches: %d" % len(p_names))

            
# collect and group match chains that refer to the same keypoint
def group_matches(matches_direct):
    # this match grouping function appears to product more entries
    # rather than reduce the entries.  I don't understand how that
    # could be, but it needs to be fixed bore the function is used for
    # anything.
    print('WARNING: this function does not seem to work correctly, need to debug it before using!!!!')
    print('Number of pair-wise matches:', len(matches_direct))
    matches_group = list(matches_direct) # shallow copy
    count = 0
    done = False
    while not done:
        print("Iteration:", count, len(matches_group))
        count += 1
        matches_new = []
        matches_lookup = {}
        for i, match in enumerate(matches_group):
            # scan if any of these match points have been previously seen
            # and record the match index
            index = -1
            for p in match[1:]:
                key = "%d-%d" % (p[0], p[1])
                if key in matches_lookup:
                    index = matches_lookup[key]
                    break
            if index < 0:
                # not found, append to the new list
                for p in match[1:]:
                    key = "%d-%d" % (p[0], p[1])
                    matches_lookup[key] = len(matches_new)
                    matches_new.append(list(match)) # shallow copy
            else:
                # found a previous reference, append these match items
                existing = matches_new[index]
                # only append items that don't already exist in the early
                # match, and only one match per image (!)
                for p in match[1:]:
                    key = "%d-%d" % (p[0], p[1])
                    found = False
                    for e in existing[1:]:
                        if p[0] == e[0]:
                            found = True
                            break
                    if not found:
                        # add
                        existing.append(list(p)) # shallow copy
                        matches_lookup[key] = index
                        # print "new:", existing
                        # print 
        if len(matches_new) == len(matches_group):
            done = True
        else:
            matches_group = list(matches_new) # shallow copy
    print("Unique features (after grouping):", len(matches_group))
    return matches_group
            
