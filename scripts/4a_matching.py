#!/usr/bin/python3

import argparse
import pickle
import numpy as np
import os

from props import getNode

from lib import Matcher
from lib import Pose
from lib import ProjectMgr
from lib import SRTM

# working on matching features ...
def match(project_dir, matching_options):

    
    matcher = matching_options[0]
    match_ratio = matching_options[1]
    min_pairs = matching_options[2]
    min_dist = matching_options[3]
    max_dist = matching_options[4]
    filters = matching_options[5]
    min_chain_length = matching_options[6]

    proj = ProjectMgr.ProjectMgr(project_dir)
    proj.load_images_info()
    proj.load_features(descriptors=False) # descriptors cached on the fly later
    proj.undistort_keypoints()
    proj.load_match_pairs()

    matcher_node = getNode('/config/matcher', True)
    matcher_node.setString('matcher', matcher)
    matcher_node.setFloat('match_ratio', match_ratio)
    matcher_node.setString('filter', filters)
    matcher_node.setInt('min_pairs', min_pairs)
    matcher_node.setFloat('min_dist', min_dist)
    matcher_node.setFloat('max_dist', max_dist)
    matcher_node.setInt('min_chain_len', min_chain_length)

    # save any config changes
    proj.save()

    # camera calibration
    K = proj.cam.get_K()
    print("K:", K)

    # fire up the matcher
    m = Matcher.Matcher()
    m.configure()
    m.robustGroupMatches(proj.image_list, K,
                        filter=filters, review=False)

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

    if False:
        print("Constructing unified match structure...")
        print("This probably will fail because we didn't do the ground intersection at the start...")
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
        pickle.dump(matches_direct, open(project_dir + "/matches_direct", "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keypoint projection.')
    parser.add_argument('--project', required=True, help='project directory')
    parser.add_argument('--matcher', default='FLANN',
                        choices=['FLANN', 'BF'])
    parser.add_argument('--match-ratio', default=0.75, type=float,
                        help='match ratio')
    parser.add_argument('--min-pairs', default=25, type=int,
                        help='minimum matches between image pairs to keep')
    parser.add_argument('--min-dist', default=0, type=float,
                        help='minimum 2d camera distance for pair comparison')
    parser.add_argument('--max-dist', default=75, type=float,
                        help='maximum 2d camera distance for pair comparison')
    parser.add_argument('--filter', default='gms',
                        choices=['gms', 'homography', 'fundamental', 'essential', 'none'])
    parser.add_argument('--min-chain-length', type=int, default=3, help='minimum match chain length (3 recommended)')
    #parser.add_argument('--ground', type=float, help='ground elevation in meters')

    args = parser.parse_args()

    matching_options = [args.matcher, args.match_ratio, args.min_pairs, args.min_distance,
            args.filter, args.min_chain_length]
            
    match(args.prject, matching_options)
