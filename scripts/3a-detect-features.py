#!/usr/bin/python3

import sys
sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import argparse
import cv2
import fnmatch
import numpy as np
import os.path
from progress.bar import Bar

sys.path.append('../lib')
import ProjectMgr

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

parser = argparse.ArgumentParser(description='Load the project\'s images.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--detector', default='SIFT',
                    choices=['SIFT', 'SURF', 'ORB', 'Star'])
parser.add_argument('--sift-max-features', default=5000,
                    help='maximum SIFT features')
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
parser.add_argument('--reject-margin', default=5, help='reject features within this distance of the image margin')

parser.add_argument('--force', action='store_true',
                    help='force redection of features even if features already exist')
parser.add_argument('--show', action='store_true',
                    help='show features as we detect them')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()

detector_params = { 'detector': args.detector,
                    'sift-max-features': args.sift_max_features,
                    'surf-hessian-threshold': args.surf_hessian_threshold,
                    'surf-noctaves': args.surf_noctaves,
                    'orb-max-features': args.orb_max_features,
                    'grid-detect': args.grid_detect,
                    'star-max-size': args.star_max_size,
                    'star-response-threshold': args.star_response_threshold,
                    'star-line-threshold-projected': args.star_line_threshold_projected,
                    'star-line-threshold-binarized': args.star_line_threshold_binarized,
                    'star-suppress-nonmax-size': args.star_suppress_nonmax_size }
proj.set_detector_params(detector_params)
proj.save()

# find features in the full image set
proj.detect_features(force=args.force, show=args.show)

# compute the undistorted mapping of the keypoints (features)
proj.undistort_keypoints()

# if any undistorted keypoints extend beyond the image bounds, remove them!
margin = args.reject_margin
print("Features that fall out of the image bounds after undistortion.")
bar = Bar('Filtering:', max = len(proj.image_list))
for image in proj.image_list:
    # traverse the list in reverse so we can safely remove features if
    # needed
    dirty = False
    for i in reversed(range(len(image.uv_list))):
        uv = image.uv_list[i]
        if uv[0] < margin or uv[0] > image.width - margin \
           or uv[1] < margin or uv[1] > image.height - margin:
            dirty = True
            #print ' ', i, uv
            image.kp_list.pop(i)                             # python list
            image.des_list = np.delete(image.des_list, i, 0) # np array

    if dirty:
        image.save_features()
        image.save_descriptors()
    #print image.name, len(image.kp_list), image.des_list.size
    bar.next()
bar.finish()
    
feature_count = 0
image_count = 0
for image in proj.image_list:
    feature_count += len(image.kp_list)
    image_count += 1

print("Average # of features per image found = %.0f" % (feature_count / image_count))
