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
            #print dist,
            sum += dist
            count += 1
            result_list.append( (dist, i, j) )
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
    print "mre = %.4f stddev = %.4f" % (mre, stddev)

    # mark items to delete
    delete_count = 0
    for line in result_list:
        if line[0] > mre + stddev * args.stddev:
            print "  outlier index %d-%d err=%.2f" % (line[1], line[2],
                                                      line[0])
            match = matches_direct[line[1]]
            match[line[2]+1] = [-1, -1]
            if not args.direct:
                match = matches_sba[line[1]]
                match[line[2]+1] = [-1, -1]
            delete_count += 1

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

    return delete_count

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

        if not args.direct:
            print "Writing sba matches..."
            pickle.dump(matches_sba, open(args.project + "/matches_sba", "wb"))


#print "Mean reprojection error = %.4f" % (mre)

