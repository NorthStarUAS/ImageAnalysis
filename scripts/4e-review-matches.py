#!/usr/bin/python3

# for all the images in the project image_dir, compute the camera
# poses from the aircraft pose (and camera mounting transform).
# Project the image plane onto an SRTM (DEM) surface for our best
# layout guess (at this point before we do any matching/bundle
# adjustment work.)

# TODO: after each delete, recompute outliers on the fly and start
# over.  Also perhaps show all the outliers in decending order by
# worst image, then features within that image.

import sys
#sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import argparse
import pickle
import cv2
import numpy as np
import os.path

sys.path.append('../lib')
import Groups
import Matcher
import ProjectMgr

import match_culling as cull

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

matcher = Matcher.Matcher()

print("Loading match points (direct)...")
matches_direct = pickle.load( open( os.path.join(args.project, "matches_direct"), "rb" ) )

# load the group connections within the image set
groups = Groups.load(args.project)

# traverse the matches_direct structure and create a pair-wise match
# structure.  (Start with an empty n x n list of empty pair lists,
# then fill in the structures.)
pairs = []
homography = []
averages = []
stddevs = []
status_flags = []
for i in range(len(proj.image_list)):
    p = [ [] for j in range(len(proj.image_list)) ]
    pairs.append(p)
    homography.append( [None] * len(proj.image_list) )
    averages.append( [0] * len(proj.image_list) )
    stddevs.append( [0] * len(proj.image_list) )
    status_flags.append( [None] * len(proj.image_list) )
    
for match in matches_direct:
    for p1 in match[1:]:
        for p2 in match[2:]:
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
        if len(pairs[i][j]):
            i1 = proj.image_list[i]
            i2 = proj.image_list[j]
            src = []
            dst = []
            for pair in pairs[i][j]:
                # undistorted uv points
                src.append( i1.uv_list[pair[0]] )
                dst.append( i2.uv_list[pair[1]] )
            src = np.float32(src)
            dst = np.float32(dst)
            method = 0          # a regular method using all the points
            #method = cv2.RANSAC
            #method = cv2.LMEDS
            M, status = cv2.findHomography(src, dst, method)
            homography[i][j] = M
            error = []
            for k, p in enumerate(src):
                # *wrong* for homography: p_est = M.dot( np.hstack((p, 1.0)) )[:2]
                tmp = M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]
                if abs(tmp) > 0.000001:
                    x = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / tmp
                    y = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / tmp
                    p_est = np.array([x, y])
                else:
                    p_est = np.array([0.0, 0.0])
                #print 'p est:', p_est, 'act:', dst[k]
                d = np.linalg.norm(p_est - dst[k])
                #print 'dist:', d
                error.append(d)
            error = np.array(error)
            avg = np.mean(error)
            std = np.std(error)
            averages[i][j] = avg
            stddevs[i][j] = std
            status_flags[i][j] = status
            print('pair:', i, j, 'avg:', avg, 'std:', std)
            bypair.append( [avg, std, i, j] )

bypair = sorted(bypair, key=lambda fields: fields[0], reverse=True)
mark_list = []
for line in bypair:
    print(line)
    i = line[2]
    j = line[3]
    i1 = proj.image_list[i]
    i2 = proj.image_list[j]
    status, key = matcher.showMatchOrient(i1, i2, pairs[i][j])
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
                for k, match in enumerate(matches_direct):
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
        cull.mark_using_list(mark_list, matches_direct)
        cull.delete_marked_matches(matches_direct)
 
        # write out the updated match dictionaries
        print("Writing direct matches...")
        pickle.dump(matches_direct, open(os.path.join(args.project, "matches_direct"), "wb"))

print('Hard coded exit in mid script...')
quit()

# run through the match list and compute the homography error (in image space)
error_list = []
for k, match in enumerate(matches_direct):
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
#    cull.draw_match(line[1], 0, matches_direct, proj.image_list)
    
mark_list = cull.show_outliers(error_list, matches_direct, proj.image_list)
mark_sum = len(mark_list)

# mark both direct and/or sba match lists as requested
cull.mark_using_list(mark_list, matches_direct)

if mark_sum > 0:
    print('Outliers removed from match lists:', mark_sum)
    result=input('Save these changes? (y/n):')
    if result == 'y' or result == 'Y':
        cull.delete_marked_matches(matches_direct)
        print(len(matches_direct))
        
        # write out the updated match dictionaries
        print("Writing direct matches...")
        pickle.dump(matches_direct, open(os.path.join(args.project, "matches_direct"), "wb"))


