#!/usr/bin/python3

import argparse
import fnmatch
import numpy as np
import os.path

from props import getNode

from lib import ProjectMgr

# for all the images in the project image_dir, detect features using the
# specified method and parameters
#
# Suggests censure/star has good stability between images (highest
# likelihood of finding a match in the target features set:
# http://computer-vision-talks.com/articles/2011-01-04-comparison-of-the-opencv-feature-detection-algorithms/
#
# Suggests censure/star works better than sift in outdoor natural
# environments: http://www.ai.sri.com/~agrawal/isrr.pdf
#
# Basic description of censure/star algorithm: http://www.researchgate.net/publication/221304099_CenSurE_Center_Surround_Extremas_for_Realtime_Feature_Detection_and_Matching

parser = argparse.ArgumentParser(description='Detect features in the project images.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--scale', type=float, default=0.4, help='scale images before detecting features, this acts much like a noise filter')
parser.add_argument('--detector', default='SIFT',
                    choices=['SIFT', 'SURF', 'ORB', 'Star'])
#parser.add_argument('--sift-max-features', default=30000,
#                    help='maximum SIFT features')
parser.add_argument('--surf-hessian-threshold', default=600,
                    help='hessian threshold for surf method')
parser.add_argument('--surf-noctaves', default=4,
                    help='use a bigger number to detect bigger features')
parser.add_argument('--orb-max-features', default=2000,
                    help='maximum ORB features')
parser.add_argument('--grid-detect', default=1,
                    help='run detect on gridded squares for (maybe) better feature distribution, 4 is a good starting value, only affects ORB method')
parser.add_argument('--star-max-size', default=16,
                    help='4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128')
parser.add_argument('--star-response-threshold', default=30)
parser.add_argument('--star-line-threshold-projected', default=10)
parser.add_argument('--star-line-threshold-binarized', default=8)
parser.add_argument('--star-suppress-nonmax-size', default=5)
parser.add_argument('--reject-margin', default=0, help='reject features within this distance of the image outer edge margin')

parser.add_argument('--show', action='store_true',
                    help='show features as we detect them')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)

# load existing images info which could include things like camera pose
proj.load_images_info()

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
    detector_node.setInt('grid_detect', args.grid_detect)
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

# find features in the full image set
proj.detect_features(scale=args.scale, force=True, show=args.show)

feature_count = 0
image_count = 0
for image in proj.image_list:
    feature_count += len(image.kp_list)
    image_count += 1

print("Average # of features per image found = %.0f" % (feature_count / image_count))

print("Saving project configuration")
proj.save()
