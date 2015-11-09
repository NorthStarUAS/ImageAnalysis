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
parser.add_argument('--stddev', required=True, type=int, default=6, help='how many stddevs above the mean for auto discarding features')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

print "Loading original (direct) matches ..."
matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )

print "Loading fitted (sba) matches..."
matches_sba = pickle.load( open( args.project + "/matches_sba", "rb" ) )

# image mean reprojection error
def compute_feature_mre(K, image, kp, ned):
    if image.PROJ == None:
        # rvec, tvec = image.get_proj()   # original direct pose
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

# group mean reprojection error
def compute_group_mre(image_list, cam):
    # start with a clean slate
    for image in image_list:
        image.PROJ = None

    camw, camh = proj.cam.get_image_params()

    # iterate through the match dictionary and build a per image list of
    # obj_pts and img_pts
    sum = 0.0
    count = 0
    result_list = []
    
    for i, match in enumerate(matches_sba):
        ned = match[0]
        max_dist = 0.0
        for p in match[1:]:
            image = image_list[ p[0] ]
            kp = image.uv_list[ p[1] ] # undistorted uv point
            scale = float(image.width) / float(camw)
            dist = compute_feature_mre(cam.get_K(scale), image, kp, ned)
            if dist > max_dist:
                max_dist = dist
            sum += dist
            count += 1
            #print dist,
        result_list.append( (max_dist, i) )
        #print

    # sort by worst max error first
    result_list = sorted(result_list, key=lambda fields: fields[0],
                         reverse=True)
    # meta stats on error values
    mre = sum / count
    stddev_sum = 0.0
    for line in result_list:
        error = line[0]
        stddev_sum += (mre-error)*(mre-error)
    stddev = math.sqrt(stddev_sum / count)
    print "   mre = %.4f stddev = %.4f" % (mre, stddev)

    delete_list = []
    for line in result_list:
        if line[0] > mre + stddev * args.stddev:
            index = line[1]
            print "outlier index %d err=%.2f" % (index, line[0])
            delete_list.append(index)

    delete_list = sorted(delete_list, reverse=True)
    for index in delete_list:
        # print "deleting", index
        matches_direct.pop(index)
        matches_sba.pop(index)

    return len(delete_list)

deleted_sum = 0
result = compute_group_mre(proj.image_list, proj.cam)
while result > 0:
    deleted_sum += result
    result = compute_group_mre(proj.image_list, proj.cam)

if deleted_sum > 0:
    result=raw_input('Remove ' + str(deleted_sum) + ' outliers from the original matches? (y/n):')
    if result == 'y' or result == 'Y':
        # write out the updated match dictionaries
        print "Writing direct matches..."
        pickle.dump(matches_direct, open(args.project+"/matches_direct", "wb"))

        print "Writing sba matches..."
        pickle.dump(matches_sba, open(args.project + "/matches_sba", "wb"))


#print "Mean reprojection error = %.4f" % (mre)

