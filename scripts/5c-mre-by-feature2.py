#!/usr/bin/python

# For all the feature matches and camera poses, estimate a mean
# reprojection error

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import cPickle as pickle
import cv2
#import json
import math
import numpy as np

sys.path.append('../lib')
import ProjectMgr

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--stddev', type=int, default=5, help='how many stddevs above the mean for auto discarding features')
parser.add_argument('--direct', action='store_true', help='analyze direct matches (might help if initial sba fit fails.)')

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
    if image.PROJ == None:
        # rvec, tvec = image.get_proj()   # original direct pose
        if args.direct:
            rvec, tvec = image.get_proj() # fitted pose
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
def compute_reprojection_errors(image_list, cam):
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
        for j, p in enumerate(match[1:]):
            image = image_list[ p[0] ]
            kp = image.uv_list[ p[1] ] # undistorted uv point
            scale = float(image.width) / float(camw)
            dist = compute_feature_mre(cam.get_K(scale), image, kp, ned)
            result_list.append( (dist, i, j) )

    # sort by worst max error first
    result_list = sorted(result_list, key=lambda fields: fields[0],
                         reverse=True)
    return result_list

def mark_outliers(result_list, trim_stddev):
    print "Marking outliers..."
    sum = 0.0
    count = len(result_list)

    # numerically it is better to sum up a list of numbers from
    # smallest to biggest (result_list is sorted from biggest to
    # smallest)
    for line in reversed(result_list):
        sum += line[0]
        
    # stats on error values
    print " computing stats..."
    mre = sum / count
    stddev_sum = 0.0
    for line in result_list:
        error = line[0]
        stddev_sum += (mre-error)*(mre-error)
    stddev = math.sqrt(stddev_sum / count)
    print "mre = %.4f stddev = %.4f" % (mre, stddev)

    # mark match items to delete
    print " marking outliers..."
    mark_count = 0
    for line in result_list:
        # print "line:", line
        if line[0] > mre + stddev * trim_stddev:
            print "  outlier index %d-%d err=%.2f" % (line[1], line[2],
                                                      line[0])
            match = matches_direct[line[1]]
            match[line[2]+1] = [-1, -1]
            if not args.direct:
                match = matches_sba[line[1]]
                match[line[2]+1] = [-1, -1]
            mark_count += 1

    # trim the result_list
    print " trimming results list..."
    for i in range(len(result_list)):
        line = result_list[i]
        if line[0] < mre + stddev * trim_stddev:
            if i > 0:
                # delete the remainder of the sorted list
                del result_list[0:i]
            # and break
            break
            
    return result_list, mark_count

# delete marked matches
def delete_marked_matches():
    print " deleting marked items..."
    for i in reversed(range(len(matches_direct))):
        match_direct = matches_direct[i]
        if not args.direct:
            match_sba = matches_sba[i]
        for j in reversed(range(1, len(match_direct))):
            p = match_direct[j]
            if p == [-1, -1]:
                match_direct.pop(j)
                if not args.direct:
                    match_sba.pop(j)
        if len(match_direct) < 3:
            # print "deleting:", match_direct
            matches_direct.pop(i)
            if not args.direct:
                matches_sba.pop(i)


result_list = compute_reprojection_errors(proj.image_list, proj.cam)

mark_sum = 0
result_list, mark_count = mark_outliers(result_list, args.stddev)
while mark_count > 0:
    mark_sum += mark_count
    result_list, mark_count = mark_outliers(result_list, args.stddev)

if mark_sum > 0:
    result=raw_input('Remove ' + str(mark_sum) + ' outliers from the original matches? (y/n):')
    if result == 'y' or result == 'Y':
        delete_marked_matches()
        # write out the updated match dictionaries
        print "Writing direct matches..."
        pickle.dump(matches_direct, open(args.project+"/matches_direct", "wb"))

        if not args.direct:
            print "Writing sba matches..."
            pickle.dump(matches_sba, open(args.project + "/matches_sba", "wb"))


#print "Mean reprojection error = %.4f" % (mre)

