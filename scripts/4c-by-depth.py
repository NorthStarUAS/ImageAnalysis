#!/usr/bin/env python3

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

# at the moment my brain is not thinking too clearly, but this script
# is essentially computing a # of standard deviations from the mean
# error metric.  But then in non-interacative mode it is doing statics
# on the metric and so we are culling 'n' standard deviations from the
# mean of standard deviations which probably does something, but is
# weird and I don't want to think about it right now!

import argparse
import pickle
from math import sqrt
import numpy as np
import os
import sys

sys.path.append('../lib')
import groups
import project

import match_culling as cull

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('project', help='project directory')
parser.add_argument('--stddev', type=float, default=3, help='how many standard deviations above the mean for auto discarding features')
parser.add_argument('--interactive', action='store_true', help='interactively review reprojection errors from worst to best and select for deletion or keep.')

args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()
proj.load_features()
proj.undistort_keypoints()

source = 'matches_direct'
print("Loading matches:", source)
matches_orig = pickle.load( open( os.path.join(args.project, source), "rb" ) )
print('Number of original features:', len(matches_orig))
print("Loading optimized matches: matches_opt")
matches_opt = pickle.load( open( os.path.join(args.project, "matches_opt"), "rb" ) )
print('Number of optimized features:', len(matches_opt))

# load the group connections within the image set
group_list = groups.load(args.project)
print('Main group size:', len(group_list[0]))

# compute the depth of each feature for each image
def compute_feature_depths(image_list, group, matches):
    print("Computing depths for all match points...")

    # init structures
    for image in image_list:
        image.z_list = []

    # make a list of distances for each feature of each image
    for match in matches:
        feat_ned = match[0]
        count = 0
        for m in match[1:]:
            if m[0] in group:
                count += 1
        if count < 2:
            continue
        for m in match[1:]:
            if m[0] in group:
                image = image_list[m[0]]
                cam_ned, ypr, quat = image.get_camera_pose(opt=True)
                dist = np.linalg.norm(np.array(feat_ned) - np.array(cam_ned))
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
        metric_sum = 0
        count = 0
        for p in match[1:]:
            if p[0] in group:
                image = image_list[p[0]]
                count += 1
                cam_ned, ypr, quat = image.get_camera_pose(opt=True)
                dist = np.linalg.norm(np.array(feat_ned) - np.array(cam_ned))
                dist_error = abs(dist - image.z_avg)
                #dist_metric = dist_error / image.z_std
                dist_metric = dist_error
                metric_sum += dist_metric
        if count >= 2:
            metric_avg = metric_sum / count
            error_list.append( [metric_avg, i, 0] )

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
    mean = sum / count
    stddev_sum = 0.0
    for line in error_list:
        error = line[0]
        stddev_sum += (mean-error)*(mean-error)
    stddev = sqrt(stddev_sum / count)
    print("mean = %.4f stddev = %.4f" % (mean, stddev))

    # mark match items to delete
    print(" marking outliers...")
    mark_count = 0
    for line in error_list:
        # print "line:", line
        if line[0] > mean + stddev * trim_stddev:
            cull.mark_feature(matches_orig, line[1], line[2], line[0])
            cull.mark_feature(matches_opt, line[1], line[2], line[0])
            mark_count += 1

    return mark_count

error_list = compute_feature_depths(proj.image_list, group_list[0], matches_opt)

if args.interactive:
    # interactively pick outliers
    mark_list = cull.show_outliers(error_list, matches_opt, proj.image_list)

    # mark both direct and optimized match lists as requested
    cull.mark_using_list(mark_list, matches_orig)
    cull.mark_using_list(mark_list, matches_opt)
    mark_sum = len(mark_list)
else:
    # trim outliers by some # of standard deviations high
    mark_sum = mark_outliers(error_list, args.stddev)

# after marking the bad matches, now count how many remaining features
# show up in each image
for image in proj.image_list:
    image.feature_count = 0
for i, match in enumerate(matches_orig):
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
for i, match in enumerate(matches_orig):
    #print 'before:', match
    for j, p in enumerate(match[1:]):
        if p[0] in weak_dict:
             match[j+1] = [-1, -1]
             mark_sum += 1
    #print 'after:', match

if mark_sum > 0:
    print('Outliers removed from match lists:', mark_sum)
    result = input('Save these changes? (y/n):')
    if result == 'y' or result == 'Y':
        cull.delete_marked_features(matches_orig)
        cull.delete_marked_features(matches_opt)

        # write out the updated match dictionaries
        print("Writing original matches...")
        pickle.dump(matches_orig, open(os.path.join(args.project, source), "wb"))
        print("Writing optimized matches...")
        pickle.dump(matches_opt, open(os.path.join(args.project, "matches_opt"), "wb"))
