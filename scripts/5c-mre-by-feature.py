#!/usr/bin/python

# For all the feature matches and camera poses, estimate a mean
# reprojection error

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import cv2
import json
import math
import numpy as np

sys.path.append('../lib')
import ProjectMgr

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--stddev', required=True, type=int, default=6, help='how many stddevs above the mean for auto discarding features')
parser.add_argument('--select', required=True, default='direct', choices=(['direct', 'sba']))

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

print "Loading original matches ..."
f = open(args.project + "/Matches.json", 'r')
matches_direct = json.load(f)
f.close()

print "Loading sba matches ..."
f = open(args.project + "/Matches-sba.json", 'r')
matches_sba = json.load(f)
f.close()

# image mean reprojection error
def compute_feature_mre(K, image, kp, ned, select):
    if image.PROJ == None:
        if select == 'direct': rvec, tvec = image.get_proj()
        if select == 'sba': rvec, tvec = image.get_proj_sba()
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
def compute_group_mre(image_list, cam, select='direct'):
    # start with a clean slate
    for image in image_list:
        image.img_pts = []
        image.obj_pts = []
        image.PROJ = None

    camw, camh = proj.cam.get_image_params()

    # iterate through the match dictionary and build a per image list of
    # obj_pts and img_pts
    sum = 0.0
    count = 0
    result_list = []
    
    if select == 'direct': matches = matches_direct
    elif select == 'sba': matches = matches_sba
        
    for key in matches:
        feature_dict = matches[key]
        points = feature_dict['pts']
        ned = matches[key]['ned']
        #print key,
        for p in points:
            image = image_list[ p[0] ]
            kp = image.uv_list[ p[1] ] # undistorted uv point
            scale = float(image.width) / float(camw)
            dist = compute_feature_mre(cam.get_K(scale), image, kp, ned, select)
            sum += dist
            count += 1
            #print dist,
            result_list.append( (dist, key) )
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

    for line in result_list:
        if line[0] > mre + stddev * args.stddev:
            key = line[1]
            print "deleting key %s err=%.2f" % (key, line[0])
            if key in matches_direct: del matches_direct[key]
            if key in matches_sba: del matches_sba[key]
    return mre

# group altitude filter
def compute_group_altitude(select='direct'):
    # iterate through the match dictionary and build a per image list of
    # obj_pts and img_pts
    sum = 0.0
    count = 0

    if select == 'direct': matches = matches_direct
    elif select == 'sba': matches = matches_sba

    for key in matches:
        feature_dict = matches[key]
        ned = matches[key]['ned']
        sum += ned[2]
        
    avg_alt = sum / len(matches)
    print "Average altitude = %.2f" % (avg_alt)
    
    # stats
    stddev_sum = 0.0
    for key in matches:
        feature_dict = matches[key]
        ned = matches[key]['ned']
        error = avg_alt - ned[2]
        stddev_sum += error**2
    stddev = math.sqrt(stddev_sum / len(matches))
    print "stddev = %.4f" % (stddev)

    # cull outliers
    bad_keys = []
    for i, key in enumerate(matches_sba):
        feature_dict = matches[key]
        ned = matches[key]['ned']
        error = avg_alt - ned[2]
        if abs(error) > stddev * args.stddev:
            print "deleting key %s err=%.2f" % (key, error)
            bad_keys.append(key)
    for key in bad_keys:
        if key in matches_direct: del matches_direct[key]
        if key in matches_sba: del matches_sba[key]
            
    return avg_alt

mre = compute_group_mre(proj.image_list, proj.cam, select=args.select)
print "Mean reprojection error = %.4f" % (mre)

alt = compute_group_altitude(select=args.select)

# write out the updated match_dict
print "Writing original matches..."
f = open(args.project + "/Matches.json", 'w')
json.dump(matches_direct, f, sort_keys=True)
f.close()
print "Writing sba matches..."
f = open(args.project + "/Matches-sba.json", 'w')
json.dump(matches_sba, f, sort_keys=True)
f.close()
