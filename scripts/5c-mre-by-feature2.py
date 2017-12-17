#!/usr/bin/python

# For all the feature matches and camera poses, estimate a mean
# reprojection error

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

import argparse
import cPickle as pickle
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
parser.add_argument('--stddev', type=float, default=3, help='how many stddevs above the mean for auto discarding features')
parser.add_argument('--direct', action='store_true', help='analyze direct matches (might help if initial sba fit fails.)')
parser.add_argument('--strong', action='store_true', help='remove entire match chain, not just the worst offending element.')
parser.add_argument('--interactive', action='store_true', help='interactively review reprojection errors from worst to best and select for deletion or keep.')

args = parser.parse_args()

if args.direct:
    print "NOTICE: analyzing direct matches list"
    
proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

print "Loading original (direct) matches ..."
matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )

if not args.direct:
    print "Loading fitted (sba) matches..."
    matches_sba = pickle.load( open( args.project + "/matches_sba", "rb" ) )

# image mean reprojection error
def compute_feature_mre(K, image, kp, ned):
    if image.PROJ is None:
        if args.direct:
            rvec, tvec = image.get_proj() # original direct pose
        else:
            rvec, tvec = image.get_proj_sba() # fitted pose
        R, jac = cv2.Rodrigues(rvec)
        image.PROJ = np.concatenate((R, tvec), axis=1)

    PROJ = image.PROJ
    uvh = K.dot( PROJ.dot( np.hstack((ned, 1.0)) ).T )
    #print uvh
    uvh /= uvh[2]
    #print uvh
    #print "%s -> %s" % ( image.img_pts[i], [ np.squeeze(uvh[0,0]), np.squeeze(uvh[1,0]) ] )
    uv = np.array( [ np.squeeze(uvh[0,0]), np.squeeze(uvh[1,0]) ] )
    dist = np.linalg.norm(np.array(kp) - uv)
    return dist

# group reprojection error for every used feature
def compute_reprojection_errors(image_list, group, cam):
    print "Computing reprojection error for all match points..."

    # start with a clean slate
    for image in image_list:
        image.PROJ = None

    camw, camh = proj.cam.get_image_params()

    # iterate through the match dictionary and build a per image list of
    # obj_pts and img_pts
    result_list = []

    if args.direct:
        matches_source = matches_direct
    else:
        matches_source = matches_sba
        
    for i, match in enumerate(matches_source):
        ned = match[0]
        
        # debug
        verbose = 0
        for p in match[1:]:
            if image_list[p[0]].name == 'DSC05841.JPG':
                verbose += 1
            if image_list[p[0]].name == 'DSC06299.JPG':
                verbose += 1
                
        for j, p in enumerate(match[1:]):
            max_dist = 0
            max_index = 0
            if p[0] in group:
                image = image_list[ p[0] ]
                # kp = image.kp_list[p[1]].pt # distorted
                kp = image.uv_list[ p[1] ]  # undistorted uv point
                scale = float(image.width) / float(camw)
                dist = compute_feature_mre(cam.get_K(scale), image, kp, ned)
                if dist > max_dist:
                    max_dist = dist
                    max_index = j
                if verbose >= 2:
                    print i, match, dist
        result_list.append( (max_dist, i, max_index) )

    # sort by error, worst is first
    result_list = sorted(result_list, key=lambda fields: fields[0],
                         reverse=True)
    return result_list

def mark_outliers(error_list, trim_stddev):
    print "Marking outliers..."
    sum = 0.0
    count = len(error_list)

    # numerically it is better to sum up a list of floatting point
    # numbers from smallest to biggest (error_list is sorted from
    # biggest to smallest)
    for line in reversed(error_list):
        sum += line[0]
        
    # stats on error values
    print " computing stats..."
    mre = sum / count
    stddev_sum = 0.0
    for line in error_list:
        error = line[0]
        stddev_sum += (mre-error)*(mre-error)
    stddev = math.sqrt(stddev_sum / count)
    print "mre = %.4f stddev = %.4f" % (mre, stddev)

    # mark match items to delete
    print " marking outliers..."
    mark_count = 0
    for line in error_list:
        # print "line:", line
        if line[0] > mre + stddev * trim_stddev:
            cull.mark_outlier(matches_direct, line[1], line[2], line[0])
            if not args.direct:
                cull.mark_outlier(matches_sba, line[1], line[2], line[0])
            mark_count += 1
            
    return mark_count

# delete marked matches
def delete_marked_matches():
    print " deleting marked items..."
    for i in reversed(range(len(matches_direct))):
        match_direct = matches_direct[i]
        if not args.direct:
            match_sba = matches_sba[i]
        has_bad_elem = False
        for j in reversed(range(1, len(match_direct))):
            p = match_direct[j]
            if p == [-1, -1]:
                has_bad_elem = True
                match_direct.pop(j)
                if not args.direct:
                    match_sba.pop(j)
        if args.strong and has_bad_elem:
            print "deleting entire match that contains a bad element"
            matches_direct.pop(i)
            if not args.direct:
                matches_sba.pop(i)
        elif len(match_direct) < 3:
            print "deleting match that is now in less than 2 images:", match_direct
            matches_direct.pop(i)
            if not args.direct:
                matches_sba.pop(i)
        elif False and len(match_direct) < 4:
            # this is seeming like less and less of a good idea (Jan 3, 2017)
            print "deleting match that is now in less than 3 images:", match_direct
            matches_direct.pop(i)
            if not args.direct:
                matches_sba.pop(i)

# load the group connections within the image set
groups = Groups.load(args.project)

if args.direct:
    matches=matches_direct
else:
    matches=matches_sba

# fixme: probably should pass in the matches structure to this call
error_list = compute_reprojection_errors(proj.image_list, groups[0], proj.cam)

if args.interactive:
    # interactively pick outliers
    mark_list = cull.show_outliers(error_list, matches, proj.image_list)

    # mark both direct and/or sba match lists as requested
    cull.mark_using_list(mark_list, matches_direct)
    if not args.direct:
        cull.mark_using_list(mark_list, matches_sba)
    mark_sum = len(mark_list)
else:
    # trim outliers by some # of standard deviations high
    mark_sum = mark_outliers(error_list, args.stddev)

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
print 'weak images:', weak_dict

# mark any features in the weak images list
for i, match in enumerate(matches_direct):
    #print 'before:', match
    for j, p in enumerate(match[1:]):
        if p[0] in weak_dict:
             match[j+1] = [-1, -1]
             mark_sum += 1
    #print 'after:', match

if mark_sum > 0:
    print 'Outliers removed from match lists:', mark_sum
    result=raw_input('Save these changes? (y/n):')
    if result == 'y' or result == 'Y':
        delete_marked_matches()
        # write out the updated match dictionaries
        print "Writing direct matches..."
        pickle.dump(matches_direct, open(args.project+"/matches_direct", "wb"))

        if not args.direct:
            print "Writing sba matches..."
            pickle.dump(matches_sba, open(args.project + "/matches_sba", "wb"))

#print "Mean reprojection error = %.4f" % (mre)

