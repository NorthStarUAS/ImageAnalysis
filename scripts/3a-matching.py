#!/usr/bin/env python3

import argparse
import pickle
import numpy as np

from props import getNode

from lib import camera
from lib.logger import log
from lib import matcher
from lib import project
from lib import smart
from lib import srtm

# working on matching features ...

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('project', help='project directory')

parser.add_argument('--scale', type=float, default=0.4, help='scale images before detecting features, this acts much like a noise filter')
parser.add_argument('--detector', default='SIFT',
                    choices=['SIFT', 'SURF', 'ORB', 'Star'])
#parser.add_argument('--sift-max-features', default=30000,
#                    help='maximum SIFT features')
parser.add_argument('--surf-hessian-threshold', default=600,
                    help='hessian threshold for surf method')
parser.add_argument('--surf-noctaves', default=4,
                    help='use a bigger number to detect bigger features')
parser.add_argument('--orb-max-features', default=20000,
                    help='maximum ORB features')
parser.add_argument('--star-max-size', default=16,
                    help='4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128')
parser.add_argument('--star-response-threshold', default=30)
parser.add_argument('--star-line-threshold-projected', default=10)
parser.add_argument('--star-line-threshold-binarized', default=8)
parser.add_argument('--star-suppress-nonmax-size', default=5)
parser.add_argument('--reject-margin', default=0, help='reject features within this distance of the image outer edge margin')

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

proj = project.ProjectMgr(args.project)
proj.load_images_info()
#proj.load_features(descriptors=False) # descriptors cached on the fly later
#proj.undistort_keypoints()
proj.load_match_pairs()

# setup project detector params
detector_node = getNode('/config/detector', True)
detector_node.setString('detector', args.detector)
detector_node.setString('scale', args.scale)
if args.detector == 'SIFT':
    #detector_node.setInt('sift_max_features', args.sift_max_features)
    pass
elif args.detector == 'SURF':
    detector_node.setInt('surf_hessian_threshold', args.surf_hessian_threshold)
    detector_node.setInt('surf_noctaves', args.surf_noctaves)
elif args.detector == 'ORB':
    detector_node.setInt('orb_max_features', args.orb_max_features)
elif args.detector == 'Star':
    detector_node.setInt('star_max_size', args.star_max_size)
    detector_node.setInt('star_response_threshold',
                         args.star_response_threshold)
    detector_node.setInt('star_line_threshold_projected',
                         args.star_response_threshold)
    detector_node.setInt('star_line_threshold_binarized',
                         args.star_line_threshold_binarized)
    detector_node.setInt('star_suppress_nonmax_size',
                         args.star_suppress_nonmax_size)

matcher_node = getNode('/config/matcher', True)
matcher_node.setFloat('match_ratio', args.match_ratio)
matcher_node.setString('filter', args.filter)
matcher_node.setInt('min_pairs', args.min_pairs)
matcher_node.setFloat('min_dist', args.min_dist)
matcher_node.setFloat('max_dist', args.max_dist)
matcher_node.setInt('min_chain_len', args.min_chain_length)

# save any config changes
proj.save()

ref_node = getNode('/config/ned_reference', True)
ref = [ ref_node.getFloat('lat_deg'),
        ref_node.getFloat('lon_deg'),
        ref_node.getFloat('alt_m') ]
log("NED reference location:", ref)
# local surface approximation
srtm.initialize( ref, 6000, 6000, 30)
smart.load(proj.analysis_dir)
smart.update_srtm_elevations(proj)
smart.set_yaw_error_estimates(proj)
proj.save_images_info()

# camera calibration
K = camera.get_K()
print("K:", K)

# fire up the matcher
matcher.configure()
matcher.find_matches(proj.image_list, K, transform=args.filter, sort=True,
                     review=False)

feature_count = 0
image_count = 0
for image in proj.image_list:
    feature_count += image.num_features
    image_count += 1
log("Average # of features per image found = %.0f" % (feature_count / image_count))

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
    pickle.dump(matches_direct, open(args.project + "/matches_direct", "wb"))
