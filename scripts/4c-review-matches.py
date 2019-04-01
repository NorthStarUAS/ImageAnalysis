#!/usr/bin/python3

# for all the images in the project image_dir, compute the camera
# poses from the aircraft pose (and camera mounting transform).
# Project the image plane onto an SRTM (DEM) surface for our best
# layout guess (at this point before we do any matching/bundle
# adjustment work.)

# TODO: after each delete, recompute outliers on the fly and start
# over.  Also perhaps show all the outliers in decending order by
# worst image, then features within that image.

import argparse
import pickle
import cv2
import numpy as np
import os.path

from lib import Groups
from lib import Matcher
from lib import ProjectMgr

import match_culling as cull

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--stddev', type=float, default=5, help='how many stddevs above the mean for auto discarding features')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()
proj.load_features()
proj.undistort_keypoints()

matcher = Matcher.Matcher()

print("Loading match points (direct)...")
matches = pickle.load( open( os.path.join(proj.analysis_dir, "matches_direct"), "rb" ) )

print('num images:', len(proj.image_list))

# traverse the matches structure and create a pair-wise match
# structure.  (Start with an empty n x n list of empty pair lists,
# then fill in the structures.)
pairs = []
homography = []
averages = []
stddevs = []
status_flags = []
dsts = []
deltas = []
for i in range(len(proj.image_list)):
    p = [ [] for j in range(len(proj.image_list)) ]
    pairs.append(p)
    homography.append( [None] * len(proj.image_list) )
    averages.append( [0] * len(proj.image_list) )
    stddevs.append( [0] * len(proj.image_list) )
    status_flags.append( [None] * len(proj.image_list) )
    dsts.append( [None] * len(proj.image_list) )
    deltas.append( [None] * len(proj.image_list) )
    
for match in matches:
    for p1 in match[2:]:
        for p2 in match[3:]:
            if p1[0] >= p2[0]:
                # ignore the reciprocal matches
                continue
            i1 = p1[0]
            i2 = p2[0]
            pairs[ p1[0] ][ p2[0] ].append( [ p1[1], p2[1] ] )
# print pairs

# sanity check (display pairs)
# import Matcher
# m = Matcher.Matcher()
# for i in range(len(proj.image_list)):
#     for j in range(len(proj.image_list)):
#         if len(pairs[i][j]):
#             print i, j, len(pairs[i][j])
#             i1 = proj.image_list[i]
#             i2 = proj.image_list[j]
#             status = matcher.showMatchOrient(i1, i2, pairs[i][j] )

# compute the consensus homography matrix for each pairing, then
# compute projection error stats for each pairing.  (We are looking
# for outliers that don't fit the homography relationship very well or
# at all.)
bypair = []
for i in range(len(proj.image_list)):
    for j in range(len(proj.image_list)):
        if i >= j:
            # only worry about matching one direction (don't duplicate
            # effort with the reciprocal matches.)
            continue
        if len(pairs[i][j]) < 4:
            continue
        i1 = proj.image_list[i]
        i2 = proj.image_list[j]
        src = []
        dst = []
        for pair in pairs[i][j]:
            # undistorted uv points
            src.append( i1.uv_list[pair[0]] )
            dst.append( i2.uv_list[pair[1]] )
            # original points
            #src.append( i1.kp_list[pair[0]].pt )
            #dst.append( i2.kp_list[pair[1]].pt )
        src = np.float32(src)
        dst = np.float32(dst)
        dsts[i][j] = dst
        delta = []
        error = []
        filter = 'homography'
        #filter = 'affine'
        # filter = 'margin' (just a test, we probably want the margins for better distorting parameter fitting.)
        if filter == 'affine':
            fullAffine = False
            affine = cv2.estimateRigidTransform(src, dst, fullAffine)
            # print('affine:', affine)
            if affine is None:
                print("Affine failed, pair:", i, j, "num pairs:",
                      len(pairs[i][j]), pairs[i][j])
                affine = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0]])
            # for each src point, compute dst_est[i] = src[i] * affine
            for k, p in enumerate(src):
                p_est = affine.dot( np.hstack((p, 1.0)) )[:2]
                #print('p est:', p_est, 'act:', dst[k])
                #np1 = np.array(i1.coord_list[pair[0]])
                #np2 = np.array(i2.coord_list[pair[1]])
                diff = dst[k] - p_est
                dist = np.linalg.norm(diff)
                #print('dist:', d)
                delta.append(diff)
                error.append(dist)
            deltas[i][j] = delta
        elif filter == 'homography':
            method = 0          # a regular method using all the points
            #method = cv2.RANSAC
            #method = cv2.LMEDS
            M, status = cv2.findHomography(src, dst, method)
            if M is None:
                print("Homography failed, pair:", i, j, "num pairs:",
                      len(pairs[i][j]), pairs[i][j])
                continue
            #print('len:', len(pairs[i][j]))
            #print('M:', M)
            homography[i][j] = M
            for k, p in enumerate(src):
                tmp = M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]
                if abs(tmp) > 0.000001:
                    x = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / tmp
                    y = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / tmp
                    p_est = np.array([x, y])
                else:
                    print('what?')
                    p_est = np.array([0.0, 0.0])
                # print('p est:', p_est, 'act:', dst[k])
                diff = dst[k] - p_est
                dist = np.linalg.norm(diff)
                #print 'dist:', d
                delta.append(diff)
                error.append(dist)
            deltas[i][j] = delta
        elif filter == 'margin':
            margin = 0.1
            hmargin = int(i1.width * margin)
            vmargin = int(i1.height * margin)
            for k in range(len(src)):
                p1 = src[k]
                p2 = dst[k]
                flag = False
                if p1[0] < hmargin or p1[0] > i1.width - hmargin:
                    flag = True
                elif p2[0] < hmargin or p2[0] > i1.width - hmargin:
                    flag = True
                elif p1[1] < vmargin or p1[1] > i1.height - vmargin:
                    flag = True
                elif p2[1] < vmargin or p2[1] > i1.height - vmargin:
                    flag = True
                if flag:
                    error.append(1)
                else:
                    error.append(0)
        error = np.array(error)
        max_error = np.amax(error)    # maximum
        max_index = np.argmax(error)
        print('max:', max_error, '@', max_index)
        avg = np.mean(error)    # average of the errors
        std = np.std(error)     # standard dev of the errors
        averages[i][j] = avg
        stddevs[i][j] = std
        
        status = np.ones(len(pairs[i][j]), np.bool_)

        # flag any outliers by std deviation
        for k in range(len(pairs[i][j])):
            if error[k] > avg + args.stddev * std:
                status[k] = False

        # also make sure the max outlier is at least flagged
        if filter == 'homography' or filter == 'affine':
            # flag only the worst error
            status[max_index] = False
        elif filter == 'margin':
            # flag any non-zero
            for k in range(len(error)):
                if error[k] > 0.5:
                    status[k] = False
            
        status_flags[i][j] = status
        error_metric = max_error       # pure max error
        #error_metric = max_error / std # max error relative to std 
        print('pair:', i1.name, i2.name, 'max:', max_error, 'avg:', avg, 'std:', std)
        bypair.append( [error_metric, avg, std, i, j] )

sort_by = 'worst'
if sort_by == 'best':
    # best first shows our most likely co-located image pairs.  These
    # introduce feature location volatility into the solution so we'd
    # prefer to not have these.
    bypair = sorted(bypair, key=lambda fields: fields[0])
else:
    # worst to first shows our most likely outliers (bad matches)
    bypair = sorted(bypair, key=lambda fields: fields[0], reverse=True)
    
mark_list = []
for line in bypair:
    print(line)
    i = line[3]
    j = line[4]
    i1 = proj.image_list[i]
    i2 = proj.image_list[j]

    debug = True
    if debug:
        file = os.path.join(args.project, 'projection.gnuplot')
        f = open(file, 'w')
        for k in range(len(deltas[i][j])):
            dst = dsts[i][j][k]
            delta = deltas[i][j][k]
            f.write("%.2f %.2f %.2f %.2f\n" % (dst[0], dst[1],
                                               delta[0], delta[1]))
        f.close()
        
    # pass in our own status array
    print(i1.name, 'vs', i2.name)
    status, key = matcher.showMatchOrient(i1, i2, pairs[i][j],
                                          status=status_flags[i][j])
    if key == ord('q'):
        # request quit
        break
    elif key == ord(' '):
        # accept the outliers, find them, and construct a mark list
        for k, flag in enumerate(status):
            if not flag:
                p0 = pairs[i][j][k][0]
                p1 = pairs[i][j][k][1]
                print('outlier:', i, j, p0, p1)
                for k, match in enumerate(matches):
                    pos1 = pos2 = -1
                    for pos, m in enumerate(match[1:]):
                        if m[0] == i and m[1] == p0:
                            pos1 = pos
                        if m[0] == j and m[1] == p1:
                            pos2 = pos
                    if pos1 >= 0 and pos2 >= 0:
                        print("found in match: ", k)
                        mark_list.append([k, pos1])
                        mark_list.append([k, pos2])


if len(mark_list):
    print('Outliers marked:', len(mark_list))
    result=input('Save these changes? (y/n):')
    if result == 'y' or result == 'Y':
        # mark and delete the outliers
        cull.mark_using_list(mark_list, matches)
        cull.delete_marked_features(matches)
 
        # write out the updated match dictionaries
        print("Writing matches (direct) ...")
        pickle.dump(matches, open(os.path.join(args.project, "matches_direct"), "wb"))

print('Number of features:', len(matches))

print('Hard coded exit in mid script...')
quit()

# run through the match list and compute the homography error (in image space)
error_list = []
for k, match in enumerate(matches):
    max_dist = 0.0
    for p1 in match[1:]:
        for p2 in match[1:]:
            i = p1[0]
            j = p2[0]
            if i >= j:
                # ignore reciprocal matches
                continue
            i1 = proj.image_list[i]
            i2 = proj.image_list[j]
            # print i, j, i1.name, i2.name
            np1 = np.array(i1.uv_list[p1[1]])
            np2 = np.array(i2.uv_list[p2[1]])
            M = homography[i][j]
            tmp = M[2][0]*np1[0] + M[2][1]*np1[1] + M[2][2]
            if abs(tmp) > 0.000001:
                x = (M[0][0]*np1[0] + M[0][1]*np1[1] + M[0][2]) / tmp
                y = (M[1][0]*np1[0] + M[1][1]*np1[1] + M[1][2]) / tmp
                p_est = np.array([x, y])
            else:
                p_est = np.array([0.0, 0.0])
            # print 'p est:', p_est, 'act:', np2
            d = np.linalg.norm(p_est - np2)
            # print 'dist:', d
            if d > max_dist:
                max_dist = d
    error_list.append( [max_dist, k, 0] )
    
error_list = sorted(error_list, key=lambda fields: fields[0], reverse=True)

#for line in result:
#    print 'index:', line[1], 'error:', line[0]
#    cull.draw_match(line[1], 0, matches, proj.image_list)
    
mark_list = cull.show_outliers(error_list, matches, proj.image_list)
mark_sum = len(mark_list)

# mark both direct and/or sba match lists as requested
cull.mark_using_list(mark_list, matches)

if mark_sum > 0:
    print('Outliers removed from match lists:', mark_sum)
    result=input('Save these changes? (y/n):')
    if result == 'y' or result == 'Y':
        cull.delete_marked_features(matches)
        print('Number of matches:', len(matches))
        
        # write out the updated match dictionaries
        print("Writing direct matches...")
        pickle.dump(matches, open(os.path.join(args.project, "matches_direct"), "wb"))
else:
    print('Number of matches:', len(matches))
