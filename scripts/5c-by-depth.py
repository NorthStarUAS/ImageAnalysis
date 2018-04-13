#!/usr/bin/python3

# When solving a bundle adjustment problem with pair-wise feature
# matches (i.e. no 3+ way matches), the bad matches will often find a
# zero error position when minimizing the mre, but they may be
# nonsensically far away from the other points.  We can't catch every
# bad match this way, but we can get some that don't show up in the
# mre test.

# For each image, estimate the depth of each feature (i.e. distance
# from the camera.)  Then compute an average depth and standard
# deviation.  Leverage this to find depth outliers.

# Notes: this filter will work best for mostly nadir shots
# (vs. oblique angle shots) where the feature depth is more consistant
# throughout the image.  However, for the use cases here, oblique
# shots tend to show up at the fringes of the data set due to turns
# and often have poor connectivity and aren't as useful anyway.

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

import argparse
import pickle
import cv2
#import json
import math
import numpy as np

sys.path.append('../lib')
import Groups
import ProjectMgr

import match_culling as cull

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--std', type=float, default=3, help='how many standard deviations above the mean for auto discarding features')
parser.add_argument('--interactive', action='store_true', help='interactively review reprojection errors from worst to best and select for deletion or keep.')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

print("Loading original (direct) matches ...")
matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )

print("Loading fitted (sba) matches...")
matches_sba = pickle.load( open( args.project + "/matches_sba", "rb" ) )

# compute the depth of each feature for each image
def compute_feature_depths(image_list, group, matches):
    print("Computing depths for all match points...")

    # init structures
    for image in image_list:
        image.z_list = []
        
    # make a list of distances for each feature of each image
    for match in matches:
        feat_ned = match[0]
        for p in match[1:]:
            if p[0] in group:
                image = image_list[p[0]]
                cam_ned, ypr, quat = image.get_camera_pose_sba()
                dist = np.linalg.norm(feat_ned - cam_ned)
                image.z_list.append(dist)

    # compute stats
    for image in image_list:
        if len(image.z_list):
            avg = np.mean(np.array(image.z_list))
            std = np.std(np.array(image.z_list))
        else:
            avg = None
            std = None
        image.z_avg = avg
        image.z_std = std
        print(image.name, 'features:', len(image.z_list), 'avg:', avg, 'std:', std)

    # make a list of relative depth errors corresponding to the
    # matches list
    error_list = []
    for i, match in enumerate(matches):
        feat_ned = match[0]
        error_sum = 0
        for p in match[1:]:
            if p[0] in group:
                image = image_list[p[0]]
                cam_ned, ypr, quat = image.get_camera_pose_sba()
                dist = np.linalg.norm(feat_ned - cam_ned)
                error_sum += abs(dist - image.z_avg)
        error = error_sum / len(match[1:])
        error_list.append( [error, i, 0] )

    # sort by error, worst is first
    error_list = sorted(error_list, key=lambda fields: fields[0],
                         reverse=True)
    return error_list

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
            cull.mark_outlier(matches_direct, line[1], line[2], line[0])
            cull.mark_outlier(matches_sba, line[1], line[2], line[0])
            mark_count += 1
            
    return mark_count

# load the group connections within the image set
groups = Groups.load(args.project)

error_list = compute_feature_depths(proj.image_list, groups[0], matches_sba)

if args.interactive:
    # interactively pick outliers
    mark_list = cull.show_outliers(error_list, matches_sba, proj.image_list)

    # mark both direct and/or sba match lists as requested
    cull.mark_using_list(mark_list, matches_direct)
    cull.mark_using_list(mark_list, matches_sba)
    mark_sum = len(mark_list)
else:
    # trim outliers by some # of standard deviations high
    mark_sum = mark_outliers(error_list, args.std)

# after marking the bad matches, now count how many remaining features
# show up in each image
for i in proj.image_list:
    i.feature_count = 0
for i, match in enumerate(matches_direct):
    for j, p in enumerate(match[1:]):
        if p[1] != [-1, -1]:
            image = proj.image_list[ p[0] ]
            image.feature_count += 1

# make a dict of all images with less than 25 feature matches
weak_dict = {}
for i, img in enumerate(proj.image_list):
    # print img.name, img.feature_count
    if img.feature_count > 0 and img.feature_count < 25:
        weak_dict[i] = True
print('weak images:', weak_dict)

# mark any features in the weak images list
for i, match in enumerate(matches_direct):
    #print 'before:', match
    for j, p in enumerate(match[1:]):
        if p[0] in weak_dict:
             match[j+1] = [-1, -1]
             mark_sum += 1
    #print 'after:', match

if mark_sum > 0:
    print('Outliers removed from match lists:', mark_sum)
    result=raw_input('Save these changes? (y/n):')
    if result == 'y' or result == 'Y':
        cull.delete_marked_matches(matches_direct)
        cull.delete_marked_matches(matches_sba)
        
        # write out the updated match dictionaries
        print("Writing direct matches...")
        pickle.dump(matches_direct, open(args.project+"/matches_direct", "wb"))

        print("Writing sba matches...")
        pickle.dump(matches_sba, open(args.project + "/matches_sba", "wb"))

