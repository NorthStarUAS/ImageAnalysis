#!/usr/bin/python3

import sys
#sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import argparse
import pickle
import cv2
import fnmatch
import json
import math
import numpy as np
import os.path
from progress.bar import Bar

sys.path.append('../lib')
import Matcher
import Pose
import ProjectMgr
import SRTM

# working on matching features ...

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--matcher', default='FLANN',
                    choices=['FLANN', 'BF'])
parser.add_argument('--match-ratio', default=0.75, type=float,
                    help='match ratio')
parser.add_argument('--min-pairs', default=20, type=int,
                    help='minimum matches between image pairs to keep')
parser.add_argument('--filter', default='essential',
                    choices=['homography', 'fundamental', 'essential', 'none'])
parser.add_argument('--ground', type=float, help='ground elevation in meters')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

if args.ground:
    proj.fastProjectKeypointsToGround(args.ground)
else:
    # setup SRTM ground interpolator
    ref = proj.ned_reference_lla
    sss = SRTM.NEDGround( ref, 2000, 2000, 30 )

    slow_way = False
    if slow_way:
        camw, camh = proj.cam.get_image_params()
        bar = Bar('Projecting keypoints to vectors:',
                  max = len(proj.image_list))
        for image in proj.image_list:
            scale = float(image.width) / float(camw)
            K = proj.cam.get_K(scale)
            IK = np.linalg.inv(K)
            image.vec_list = proj.projectVectors(IK, image, image.uv_list)
            bar.next()
        bar.finish()

        # intersect keypoint vectors with srtm terrain
        bar = Bar('Vector/terrain intersecting:',
                  max = len(proj.image_list))
        for image in proj.image_list:
            image.coord_list = sss.interpolate_vectors(image.camera_pose,
                                                       image.vec_list)
            bar.next()
        bar.finish()
    else:
        # compute keypoint usage map
        proj.compute_kp_usage(all=True)

        # fast way:
        # 1. make a grid (i.e. 8x8) of uv coordinates covering the whole image
        # 2. undistort these uv coordinates
        # 3. project them into vectors
        # 4. intersect them with the srtm terrain to get ned coordinates
        # 5. use linearndinterpolator ... g = scipy.interpolate.LinearNDInterpolator([[0,0],[1,0],[0,1],[1,1]], [[0,4,8],[1,3,2],[2,2,-4],[4,1,0]])
        #    with origin uv vs. 3d location to build a table
        # 6. interpolate original uv coordinates to 3d locations
        proj.fastProjectKeypointsTo3d(sss)

proj.matcher_params = { 'matcher': args.matcher,
                        'match-ratio': args.match_ratio,
                        'filter': args.filter }
proj.save()

# determine scale value so we can get correct K matrix
image_width = proj.image_list[0].width
camw, camh = proj.cam.get_image_params()
scale = float(image_width) / float(camw)
print('scale:', scale)
# camera calibration
K = proj.cam.get_K(scale)
print("K:", K)

# fire up the matcher
m = Matcher.Matcher()
m.min_pairs = args.min_pairs
m.configure(proj.detector_params, proj.matcher_params)
m.robustGroupMatches(proj.image_list, K, filter=args.filter, review=False)

# The following code is deprecated ...
do_old_match_consolodation = False
if do_old_match_consolodation:
    # build a list of all 'unique' keypoints.  Include an index to each
    # containing image and feature.
    matches_dict = {}
    for i, i1 in enumerate(proj.image_list):
        for j, matches in enumerate(i1.match_list):
            if j > i:
                for pair in matches:
                    key = "%d-%d" % (i, pair[0])
                    m1 = [i, pair[0]]
                    m2 = [j, pair[1]]
                    if key in matches_dict:
                        feature_dict = matches_dict[key]
                        feature_dict['pts'].append(m2)
                    else:
                        feature_dict = {}
                        feature_dict['pts'] = [m1, m2]
                        matches_dict[key] = feature_dict
    #print match_dict
    count = 0.0
    sum = 0.0
    for key in matches_dict:
        sum += len(matches_dict[key]['pts'])
        count += 1
    if count > 0.1:
        print("total unique features in image set = %d" % count)
        print("kp average instances = %.4f" % (sum / count))

    # compute an initial guess at the 3d location of each unique feature
    # by averaging the locations of each projection
    for key in matches_dict:
        feature_dict = matches_dict[key]
        sum = np.array( [0.0, 0.0, 0.0] )
        for p in feature_dict['pts']:
            sum += proj.image_list[ p[0] ].coord_list[ p[1] ]
        ned = sum / len(feature_dict['pts'])
        feature_dict['ned'] = ned.tolist()

def update_match_location(match):
    sum = np.array( [0.0, 0.0, 0.0] )
    for p in match[1:]:
        # print proj.image_list[ p[0] ].coord_list[ p[1] ]
        sum += proj.image_list[ p[0] ].coord_list[ p[1] ]
        ned = sum / len(match[1:])
        # print "avg =", ned
        match[0] = ned.tolist()
    return match

print("Constructing unified match structure...")
matches_direct = []
for i, image in enumerate(proj.image_list):
    # print image.name
    for j, matches in enumerate(image.match_list):
        # print proj.image_list[j].name
        if j > i:
            for pair in matches:
                match = []
                # ned place holder
                match.append([0.0, 0.0, 0.0])
                match.append([i, pair[0]])
                match.append([j, pair[1]])
                update_match_location(match)
                matches_direct.append(match)
                # print pair, match
                
print("Writing match file ...")
pickle.dump(matches_direct, open(args.project + "/matches_direct", "wb"))
