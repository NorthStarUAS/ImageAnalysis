#!/usr/bin/env python3

# compute neighbors in pixel space, then compute 'cone' shape metric
# in 3d sba space.  Outliers will typically be separted from their
# neighbors in 3d space ... i.e. long cone depth relative to base
# size.

# Build a list of all 3d vs. 2d points for each image.  Because a
# feature could have matches in multiple images, there will be
# repeated 2d (uv) coordinates with numerically distinct 3d positions.
# If the SBA worked well, all the 3d coordinates should be "close" for
# all matching UV's and there should be a fairly consistant surface
# topolgy.  If there is a bad match, because we are just doing match
# pairs at this point, the SBA solver can usually move the 3d
# coordinate somewhere that satisfies the constraint, but in doing so,
# will (hopefully) be forced to move the point away from a
# topologically consistant surface.

# If we can find these outliers that don't make sense with the local
# topology, we may have matched a point at the top of a flag pole or
# at the bottom of a well ... but chances are more likely it's a bad
# match and we can remove it.

# this approach uses kdtrees and nearest neighbors rather than a
# delauney triangulation.  Delauney triangulation was cool until we
# had to deal with multiple copies of the same uv coordinates.

import argparse
import commands
import cPickle as pickle
import cv2
import fnmatch
import itertools
#import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
import scipy.spatial
import sys

sys.path.append('../lib')
import project
import transformations

def meta_stats(report):
    sum = 0.0
    count = len(report)
    for line in report:
        value = line[0]
        sum += value
    average = sum / len(report)
    print "average value = %.2f" % (average)

    sum = 0.0
    for line in report:
        value = line[0]
        diff = average - value
        sum += diff**2
    stddev = math.sqrt(sum / count)
    print "standard deviation = %.2f" % (stddev)
    return average, stddev

parser = argparse.ArgumentParser(description='Compute Delauney triangulation of matches.')
parser.add_argument('project', help='project directory')
parser.add_argument('--stddev', default=5, type=int, help='standard dev threshold')
parser.add_argument('--checkpoint', action='store_true', help='auto save results after each iteration')
parser.add_argument('--show', action='store_true', help='show most extreme reprojection errors with matches.')
args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

print "Loading original (direct) matches ..."
matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )

print "Loading fitted (sba) matches..."
matches_sba = pickle.load( open( args.project + "/matches_sba", "rb" ) )
print "features:", len(matches_sba)

def compute_surface_outliers():
    # start with a clean slate
    for image in proj.image_list:
        image.feat_3d = []
        image.feat_uv = []
        image.match_sba_idx = []
        image.kdtree = None
    matches_fit_sum = np.zeros(len(matches_sba))
    matches_fit_count = np.zeros(len(matches_sba))
    
    # iterate through the sba match dictionary and build a per-image
    # list of 3d feature points with corresponding 2d uv coordinates
    print "Sorting matches into per-image structures..."
    for i, match in enumerate(matches_sba):
        # print i, match
        ned = match[0]
        for p in match[1:]:
            image = proj.image_list[ p[0] ]
            image.feat_3d.append( np.array(ned) )
            image.feat_uv.append( image.uv_list[p[1]] )
            image.match_sba_idx.append( i )
            uv = image.uv_list[p[1]]
            if abs(uv[0]-1861.316) < 0.1:
                if abs(uv[1] - 839.172) < 0.1:
                       print image.name, 'match_sba index:', i
                       print 'sba:', matches_sba[i]
                       print 'direct:', matches_direct[i]
                       draw_match(i, -1)
    print 'Constructing kd trees...'
    for image in proj.image_list:
        if len(image.feat_uv):
            image.kdtree = scipy.spatial.KDTree(image.feat_uv)
            
    print "Processing images..."
    report = []
    for image in proj.image_list:
        print image.name, len(image.feat_uv)
        if len(image.feat_uv) and len(image.feat_uv) < 3:
            print "Image with > 0, but < 3 features"
            continue
        for i, uv in enumerate(image.feat_uv):
            #print 'uv:', uv, image.feat_3d[i]
            (dist_uv, index) = image.kdtree.query(uv, k=30)
            #print dist_uv, index
            # compare the 2d distance vs 3d distance for nearby neighbors
            dist_2d = []
            dist_3d = []
            count = 0
            for j, d2d in enumerate(dist_uv):
                if d2d > 0:
                    count += 1
                if count > 10:
                    break
                index3d = index[j]
                d3d = np.linalg.norm(image.feat_3d[i] - image.feat_3d[index3d])
                dist_2d.append(d2d)
                dist_3d.append(d3d)
                if image.match_sba_idx[i] == 67469:
                    print image.feat_uv[index3d], image.feat_3d[index3d]
            z = np.polyfit(np.array(dist_2d), np.array(dist_3d), 1)
            p = np.poly1d(z)
            if image.match_sba_idx[i] == 67469:
                print 'uv:', uv, image.feat_3d[i]
                print dist_2d
                print dist_3d
                print z
            #print ' z:', z
            # evaluate the 2d vs. 3d fit for each point
            count = 0
            for j, d2d in enumerate(dist_uv):
                if d2d > 0:
                    count += 1
                if count > 10:
                    break
                index3d = index[j]
                d3d = np.linalg.norm(image.feat_3d[i] - image.feat_3d[index3d])
                est_d3d = p(d2d)
                if image.match_sba_idx[i] == 67469:
                    print match_idx, 'diff:', abs(est_d3d - d3d)
                match_idx = image.match_sba_idx[index3d]
                matches_fit_sum[match_idx] += abs(est_d3d - d3d)
                matches_fit_count[match_idx] += 1
                
            # Do a least squares fit of neighbor 2d dist vs neighbor
            # 3d distance.  In a well behaved system, we would expect
            # a fairly linear relationship and significant outliers
            # are suspect.
            #match_idx = image.match_sba_idx[i]
            #print 'match_idx:', match_idx
            #matches_fit_sum[match_idx] += abs(z[1])
            #matches_fit_count[match_idx] += 1

    print 'Evaluating surface consistency results...'
    for i in range(len(matches_sba)):
        if matches_fit_count[i] < 1:
            print "Hey, match index", i, "count is zero!"
            metric = 0
        else:
            metric = matches_fit_sum[i] / matches_fit_count[i]
        report.append( (metric, i) )

    average, stddev = meta_stats(report)
    report = sorted(report, key=lambda fields: abs(fields[0]), reverse=True)

    delete_list = []
    for line in report:
        metric = line[0]
        index = line[1]
        if abs(average - metric) >= args.stddev * stddev:
            print "index=", index, "metric=", metric
            if args.show:
                draw_match(index, -1)
            delete_list.append( index )


    # set(delete_list) eliminates potential duplicates (if an outlier
    # is an outlier in both images it matches.)
    delete_list = sorted(set(delete_list), reverse=True)
    for index in delete_list:
        print "deleting", index
        matches_direct.pop(index)
        matches_sba.pop(index)

    return len(delete_list)

# experimental, draw a visual of a match point in all it's images
def draw_match(i, index):
    green = (0, 255, 0)
    red = (0, 0, 255)
    match = matches_sba[i]
    print 'match:', match, 'index:', index
    for j, m in enumerate(match[1:]):
        print ' ', m, proj.image_list[m[0]]
        img = proj.image_list[m[0]]
        # kp = img.kp_list[m[1]].pt # distorted
        print 'm[1]:', m[1], len(img.uv_list)
        kp = img.uv_list[m[1]]  # undistored
        print ' ', kp
        rgb = img.load_rgb()
        h, w = rgb.shape[:2]
        crop = True
        range = 300
        if crop:
            cx = int(round(kp[0]))
            cy = int(round(kp[1]))
            if cx < range:
                xshift = range - cx
                cx = range
            elif cx > (w - range):
                xshift = (w - range) - cx
                cx = w - range
            else:
                xshift = 0
            if cy < range:
                yshift = range - cy
                cy = range
            elif cy > (h - range):
                yshift = (h - range) - cy
                cy = h - range
            else:
                yshift = 0
            print 'size:', w, h, 'shift:', xshift, yshift
            rgb1 = rgb[cy-range:cy+range, cx-range:cx+range]
            if ( j == index ):
                color = red
            else:
                color = green
            cv2.circle(rgb1, (range-xshift,range-yshift), 2, color, thickness=2)
        else:
            scale = 790.0/float(w)
            rgb1 = cv2.resize(rgb, (0,0), fx=scale, fy=scale)
            cv2.circle(rgb1, (int(round(kp[0]*scale)), int(round(kp[1]*scale))), 2, green, thickness=2)
        cv2.imshow(img.name, rgb1)
    print 'waiting for keyboard input...'
    key = cv2.waitKey() & 0xff
    cv2.destroyAllWindows()

def save_results():
    # write out the updated match dictionaries
    print "Writing original matches..."
    pickle.dump(matches_direct, open(args.project+"/matches_direct", "wb"))

    print "Writing sba matches..."
    pickle.dump(matches_sba, open(args.project+"/matches_sba", "wb"))

deleted_sum = 0
result = compute_surface_outliers()
while result > 0:
    deleted_sum += result
    result = compute_surface_outliers()
    if args.checkpoint:
        save_results()

if deleted_sum > 0:
    result=raw_input('Remove ' + str(deleted_sum) + ' outliers from the original matches? (y/n):')
    if result == 'y' or result == 'Y':
        save_results()
