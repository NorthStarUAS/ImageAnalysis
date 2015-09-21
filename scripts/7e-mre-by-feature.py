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

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()

f = open(args.project + "/Matches.json", 'r')
matches_dict = json.load(f)
f.close()

# image mean reprojection error
def compute_feature_mre(K, image, kp, ned):
    if image.PROJ == None:
        rvec, tvec = image.get_proj()
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
        image.img_pts = []
        image.obj_pts = []
        image.PROJ = None

    # iterate through the match dictionary and build a per image list of
    # obj_pts and img_pts
    sum = 0.0
    count = 0
    result_list = []
    for key in matches_dict:
        feature_dict = matches_dict[key]
        points = feature_dict['pts']
        ned = matches_dict[key]['ned']
        #print key,
        for p in points:
            image = image_list[ p[0] ]
            kp = image.kp_list[ p[1] ]
            dist = compute_feature_mre( cam.get_K(), image, kp.pt, ned )
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
        if line[0] > mre + 6*stddev:
            key = line[1]
            print "deleting key %s err=%.2f" % (key, line[0])
            if key in matches_dict: del matches_dict[key]
    return mre

mre = compute_group_mre(proj.image_list, proj.cam)
print "Mean reprojection error = %.4f" % (mre)

# write out the updated match_dict
f = open(args.project + "/Matches.json", 'w')
json.dump(matches_dict, f, sort_keys=True)
f.close()
