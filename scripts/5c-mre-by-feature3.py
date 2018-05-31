#!/usr/bin/python3

# For all the feature matches and camera poses, estimate a mean
# reprojection error

import argparse
import pickle
import math
import numpy as np
import os

import sys
sys.path.append('../lib')
import Groups
import Optimizer
import ProjectMgr

import match_culling as cull

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--stddev', type=float, default=5, help='how many stddevs above the mean for auto discarding features')
parser.add_argument('--strong', action='store_true', help='remove entire match chain, not just the worst offending element.')
parser.add_argument('--interactive', action='store_true', help='interactively review reprojection errors from worst to best and select for deletion or keep.')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()
#proj.load_features()

#source = 'matches_direct'
source = 'matches_grouped'
print("Loading matches:", source)
matches_orig = pickle.load( open( os.path.join(args.project, source), "rb" ) )
print('Number of original features:', len(matches_orig))
print("Loading optimized matches: matches_opt")
matches_opt = pickle.load( open( os.path.join(args.project, "matches_opt"), "rb" ) )
print('Number of optimized features:', len(matches_opt))

# load the group connections within the image set
groups = Groups.load(args.project)
print('Main group size:', len(groups[0]))

opt = Optimizer.Optimizer(args.project)
opt.setup( proj, groups[0], matches_opt, optimized=True )
x0 = np.hstack((opt.camera_params.ravel(), opt.points_3d.ravel(),
                opt.K[0,0], opt.K[0,2], opt.K[1,2],
                opt.distCoeffs))
error = opt.fun(x0, opt.n_cameras, opt.n_points, opt.by_camera_point_indices, opt.by_camera_points_2d)

print(len(error))
mre = np.mean(np.abs(error))
std = np.std(error)
max = np.amax(np.abs(error))
print('mre: %.3f std: %.3f max: %.2f' % (mre, std, max) )

results = []
count = 0
for i, cam in enumerate(opt.camera_params.reshape((opt.n_cameras, opt.ncp))):
    print(i, opt.camera_map_fwd[i])
    orig_cam_index = opt.camera_map_fwd[i]
    # print(count, opt.by_camera_point_indices[i])
    for j in opt.by_camera_point_indices[i]:
        match = matches_opt[opt.feat_map_rev[j]]
        match_index = 0
        #print(orig_cam_index, match)
        for k, p in enumerate(match[1:]):
            if p[0] == orig_cam_index:
                match_index = k
        # print(match[0], opt.points_3d[j*3:j*3+3])
        e = error[count*2:count*2+2]
        #print(count, e, np.linalg.norm(e))
        #if abs(e[0]) > 5*std or abs(e[1]) > 5*std:
        #    print("big")
        results.append( [np.linalg.norm(e), opt.feat_map_rev[j], match_index] )
        count += 1

error_list = sorted(results, key=lambda fields: fields[0], reverse=True)


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
            cull.mark_outlier(matches_orig, line[1], line[2], line[0])
            cull.mark_outlier(matches_opt, line[1], line[2], line[0])
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

if args.interactive:
    # interactively pick outliers
    mark_list = cull.show_outliers(error_list, matches_opt, proj.image_list)

    # mark selection
    cull.mark_using_list(mark_list, matches_orig)
    cull.mark_using_list(mark_list, matches_opt)
    mark_sum = len(mark_list)
else:
    # trim outliers by some # of standard deviations high
    mark_sum = mark_outliers(error_list, args.stddev)

# after marking the bad matches, now count how many remaining features
# show up in each image
for i in proj.image_list:
    i.feature_count = 0
for i, match in enumerate(matches_opt):
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
    for i, match in enumerate(matches_opt):
        #print 'before:', match
        for j, p in enumerate(match[1:]):
            if p[0] in weak_dict:
                 match[j+1] = [-1, -1]
                 mark_sum += 0      # don't count these in the mark_sum
        #print 'after:', match

if mark_sum > 0:
    print('Outliers removed from match lists:', mark_sum)
    result = input('Save these changes? (y/n):')
    if result == 'y' or result == 'Y':
        delete_marked_matches(matches_orig)
        delete_marked_matches(matches_opt)
        # write out the updated match dictionaries
        print("Writing direct matches...")
        pickle.dump(matches_orig, open(os.path.join(args.project, source), "wb"))
        print("Writing optimized matches...")
        pickle.dump(matches_opt, open(os.path.join(args.project, "matches_opt"), "wb"))

