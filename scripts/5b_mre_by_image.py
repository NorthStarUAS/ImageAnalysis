#!/usr/bin/python3

# For all the feature matches and camera poses, estimate a mean
# reprojection error

import argparse
import pickle
import math
import numpy as np
import os

from props import getNode

from lib import Groups
from lib import Optimizer
from lib import ProjectMgr
from lib import match_culling as cull

def mre(project_dir, mre_options):
    
    group_id = mre_options[0]
    stddev = mre_options[1]
    initial_pose = mre_options[2]
    strong = mre_options[3]
    interactive = mre_options[4]

    proj = ProjectMgr.ProjectMgr(project_dir)
    proj.load_images_info()

    # a value of 2 let's pairs exist which can be trouble ...
    matcher_node = getNode('/config/matcher', True)
    min_chain_len = matcher_node.getInt("min_chain_len")
    if min_chain_len == 0:
        min_chain_len = 3
    print("Notice: min_chain_len is:", min_chain_len)

    source = 'matches_grouped'
    print("Loading matches:", source)
    matches = pickle.load( open( os.path.join(proj.analysis_dir, source), "rb" ) )
    print('Number of original features:', len(matches))

    # load the group connections within the image set
    groups = Groups.load(proj.analysis_dir)
    print('Group sizes:', end=" ")
    for group in groups:
        print(len(group), end=" ")
    print()

    opt = Optimizer.Optimizer(project_dir)
    if initial_pose:
        opt.setup( proj, groups, group_id, matches, optimized=False )
    else:
        opt.setup( proj, groups, group_id, matches, optimized=True )
    x0 = np.hstack((opt.camera_params.ravel(), opt.points_3d.ravel(),
                    opt.K[0,0], opt.K[0,2], opt.K[1,2],
                    opt.distCoeffs))
    error = opt.fun(x0, opt.n_cameras, opt.n_points, opt.by_camera_point_indices, opt.by_camera_points_2d)

    print('cameras:', opt.n_cameras)

    print(len(error))
    mre = np.mean(np.abs(error))
    std = np.std(error)
    max = np.amax(np.abs(error))
    print('mre: %.3f std: %.3f max: %.2f' % (mre, std, max) )

    print('Tabulating results...')
    results = []
    results_by_cam = []
    count = 0
    for i, cam in enumerate(opt.camera_params.reshape((opt.n_cameras, opt.ncp))):
        # print(i, opt.camera_map_fwd[i])
        orig_cam_index = opt.camera_map_fwd[i]
        cam_errors = []
        # print(count, opt.by_camera_point_indices[i])
        for j in opt.by_camera_point_indices[i]:
            match = matches[opt.feat_map_rev[j]]
            match_index = 0
            #print(orig_cam_index, match)
            for k, p in enumerate(match[2:]):
                if p[0] == orig_cam_index:
                    match_index = k
            # print(match[0], opt.points_3d[j*3:j*3+3])
            e = error[count*2:count*2+2]
            #print(count, e, np.linalg.norm(e))
            #if abs(e[0]) > 5*std or abs(e[1]) > 5*std:
            #    print("big")
            cam_errors.append( np.linalg.norm(e) )
            results.append( [np.linalg.norm(e), opt.feat_map_rev[j], match_index] )
            count += 1
        if len(cam_errors):
            results_by_cam.append( [np.mean(np.abs(np.array(cam_errors))),
                                    np.amax(np.abs(np.array(cam_errors))),
                                    proj.image_list[orig_cam_index].name ] )
        else:
            results_by_cam.append( [9999.0, 9999.0,
                                    proj.image_list[orig_cam_index].name ] )
            
        #print(proj.image_list[orig_cam_index].name, ':',
        #      np.mean(np.abs(np.array(cam_errors))))

    print("Report of images that aren't fitting well:")
    results_by_cam = sorted(results_by_cam, key=lambda fields: fields[0], reverse=True)
    for line in results_by_cam:
        if line[0] > mre + 3*std:
            print("%s - mean: %.3f max: %.3f" % (line[2], line[0], line[1]))
    for line in results_by_cam:
        if line[0] > mre + 3*std:
            print(line[2], end=" ")
    print()
        
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
                cull.mark_feature(matches, line[1], line[2], line[0])
                mark_count += 1
                
        return mark_count

    if interactive:
        # interactively pick outliers
        mark_list = cull.show_outliers(error_list, matches, proj.image_list)

        # mark selection
        cull.mark_using_list(mark_list, matches)
        mark_sum = len(mark_list)
    else:
        # trim outliers by some # of standard deviations high
        mark_sum = mark_outliers(error_list, stddev)

    # after marking the bad matches, now count how many remaining features
    # show up in each image
    for i in proj.image_list:
        i.feature_count = 0
    for i, match in enumerate(matches):
        for j, p in enumerate(match[2:]):
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
        for i, match in enumerate(matches):
            #print 'before:', match
            for j, p in enumerate(match[2:]):
                if p[0] in weak_dict:
                    match[j+1] = [-1, -1]
                    mark_sum += 1

    if mark_sum > 0:
        print('Outliers removed from match lists:', mark_sum)
        result = input('Save these changes? (y/n):')
        if result == 'y' or result == 'Y':
            cull.delete_marked_features(matches, min_chain_len, strong=strong)
            # write out the updated match dictionaries
            print("Writing:", source)
            pickle.dump(matches, open(os.path.join(proj.analysis_dir, source), "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keypoint projection.')
    parser.add_argument('--project', required=True, help='project directory')
    parser.add_argument('--group', type=int, default=0, help='group number')
    parser.add_argument('--stddev', type=float, default=5, help='how many stddevs above the mean for auto discarding features')
    parser.add_argument('--initial-pose', action='store_true', help='work on initial pose, not optimized pose')
    parser.add_argument('--strong', action='store_true', help='remove entire match chain, not just the worst offending element.')
    parser.add_argument('--interactive', action='store_true', help='interactively review reprojection errors from worst to best and select for deletion or keep.')

    args = parser.parse_args()

    mre_options = [args.group, args.stddev, args.initial_pose, args.strong, args.interactive]

    mre(args.project, mre_options)