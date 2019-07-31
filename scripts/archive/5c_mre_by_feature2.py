#!/usr/bin/python3

# For all the feature matches and camera poses, estimate a mean
# reprojection error

import argparse
import pickle
import cv2
import math
import numpy as np
import os

import sys
sys.path.append('../lib')
import Groups
import ProjectMgr

import match_culling as cull

print("Fixme: match the new MRE in the new Optimizer.py, otherwise this will be bunk, bunk I say!")

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--stddev', type=float, default=3, help='how many stddevs above the mean for auto discarding features')
parser.add_argument('--strong', action='store_true', help='remove entire match chain, not just the worst offending element.')
parser.add_argument('--interactive', action='store_true', help='interactively review reprojection errors from worst to best and select for deletion or keep.')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints(optimized=True)

source = 'matches_grouped'
print("Loading matches:", source)
matches_orig = pickle.load( open( os.path.join(args.project, source), "rb" ) )
print("Loading optimized matches: matches_sba")
matches_sba = pickle.load( open( os.path.join(args.project, "matches_sba"), "rb" ) )

# load the group connections within the image set
groups = Groups.load(args.project)
print('Main group size:', len(groups[0]))

# image mean reprojection error
def compute_feature_mre(image, kp, ned):
    uvh = image.M.dot( np.hstack((ned, 1.0)) )
    uvh /= uvh[0,2]
    uv = uvh[0, 0:2]
    dist = np.linalg.norm(np.array(kp) - uv)
    return dist

# group reprojection error for every used feature
def compute_reprojection_errors(image_list, matches, group, cam):
    print("Computing reprojection error for all match points...")
    
    camw, camh = proj.cam.get_image_params()
    scale = float(image_list[0].width) / float(camw)
    K = cam.get_K(scale, optimized=True)
    
    # compute PROJ and M=K*PROJ matrices
    for i, image in enumerate(image_list):
        if not i in group:
            continue
        rvec, tvec = image.get_proj_sba() # fitted pose
        R, jac = cv2.Rodrigues(rvec)
        image.PROJ = np.concatenate((R, tvec), axis=1)
        image.M = K.dot( image.PROJ )

    # iterate through the match dictionary and build a per image list of
    # obj_pts and img_pts
    result_list = []

    for i, match in enumerate(matches):
        ned = match[0]
        for j, p in enumerate(match[1:]):
            max_dist = 0
            max_index = 0
            if p[0] in group:
                image = image_list[ p[0] ]
                # kp = image.kp_list[p[1]].pt # distorted
                kp = image.uv_list[ p[1] ]  # undistorted uv point
                scale = float(image.width) / float(camw)
                dist = compute_feature_mre(image, kp, ned)
                if dist > max_dist:
                    max_dist = dist
                    max_index = j
        result_list.append( (max_dist, i, max_index) )

    # sort by error, worst is first
    result_list = sorted(result_list, key=lambda fields: fields[0],
                         reverse=True)
    return result_list

def mark_outliers(error_list, trim_stddev):
    print("Marking outliers...")
    sum = 0.0
    count = len(error_list)

    # numerically it is better to sum up a list of floatting point
    # numbers from smallest to biggest (error_list is sorted from
    # biggest to smallest)
    for line in reversed(error_list):
        sum += line[0]
        
    # stats on error values
    print(" computing stats...")
    mre = sum / count
    stddev_sum = 0.0
    for line in error_list:
        error = line[0]
        stddev_sum += (mre-error)*(mre-error)
    stddev = math.sqrt(stddev_sum / count)
    print("mre = %.4f stddev = %.4f" % (mre, stddev))

    # mark match items to delete
    print(" marking outliers...")
    mark_count = 0
    for line in error_list:
        # print "line:", line
        if line[0] > mre + stddev * trim_stddev:
            cull.mark_outlier(matches_sba, line[1], line[2], line[0])
            mark_count += 1
            
    return mark_count

# delete marked matches
def delete_marked_matches(matches):
    print(" deleting marked items...")
    for i in reversed(range(len(matches))):
        match = matches[i]
        has_bad_elem = False
        for j in reversed(range(1, len(match))):
            p = match[j]
            if p == [-1, -1]:
                has_bad_elem = True
                match.pop(j)
        if args.strong and has_bad_elem:
            print("deleting entire match that contains a bad element")
            matches.pop(i)
        elif len(match) < 3:
            print("deleting match that is now in less than 2 images:", match)
            matches.pop(i)

# fixme: probably should pass in the matches structure to this call
error_list = compute_reprojection_errors(proj.image_list, matches_sba, groups[0], proj.cam)

if args.interactive:
    # interactively pick outliers
    mark_list = cull.show_outliers(error_list, matches_sba, proj.image_list)

    # mark selection
    cull.mark_using_list(mark_list, matches_orig)
    cull.mark_using_list(mark_list, matches_sba)
    mark_sum = len(mark_list)
else:
    # trim outliers by some # of standard deviations high
    mark_sum = mark_outliers(error_list, args.stddev)

# after marking the bad matches, now count how many remaining features
# show up in each image
for i in proj.image_list:
    i.feature_count = 0
for i, match in enumerate(matches_sba):
    for j, p in enumerate(match[1:]):
        if p[1] != [-1, -1]:
            image = proj.image_list[ p[0] ]
            image.feature_count += 1

purge_weak_images = False
if purge_weak_images:
    # make a dict of all images with less than 25 feature matches
    weak_dict = {}
    for i, img in enumerate(proj.image_list):
        # print img.name, img.feature_count
        if img.feature_count > 0 and img.feature_count < 25:
            weak_dict[i] = True
    print('weak images:', weak_dict)

    # mark any features in the weak images list
    for i, match in enumerate(matches_orig):
        #print 'before:', match
        for j, p in enumerate(match[1:]):
            if p[0] in weak_dict:
                 match[j+1] = [-1, -1]
                 mark_sum += 1
    for i, match in enumerate(matches_sba):
        #print 'before:', match
        for j, p in enumerate(match[1:]):
            if p[0] in weak_dict:
                 match[j+1] = [-1, -1]
                 mark_sum += 0      # don't count these in the mark_sum
        #print 'after:', match

if mark_sum > 0:
    print('Outliers removed from match lists:', mark_sum)
    result=input('Save these changes? (y/n):')
    if result == 'y' or result == 'Y':
        delete_marked_matches(matches_orig)
        delete_marked_matches(matches_sba)
        # write out the updated match dictionaries
        print("Writing direct matches...")
        pickle.dump(matches_orig, open(os.path.join(args.project, source), "wb"))
        print("Writing optimized matches...")
        pickle.dump(matches_sba, open(os.path.join(args.project, "matches_sba"), "wb"))

