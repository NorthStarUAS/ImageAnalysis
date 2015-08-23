#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import os.path

sys.path.append('../lib')
import ProjectMgr

# for all the images in the project image_dir, detect features using the
# specified method and parameters

parser = argparse.ArgumentParser(description='Load the project\'s images.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--detector', default='SIFT',
                    choices=['SIFT', 'SURF', 'ORB'])
parser.add_argument('--sift-max-features', default=2000,
                    help='maximum SIFT features')
parser.add_argument('--surf-hessian-threshold', default=600,
                    help='hessian threshold for surf method')
parser.add_argument('--surf-noctaves', default=4,
                    help='use a bigger number to detect bigger features')
parser.add_argument('--orb-max-features', default=2000,
                    help='maximum ORB features')
parser.add_argument('--grid-detect', default=1,
                    help='run detect on gridded squares for (maybe) better feature distribution, 4 is a good starting value, only affects ORB method')
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
                    'grid-detect': args.grid_detect }
proj.set_detector_params(detector_params)
proj.save()

proj.detect_features(force=args.force, show=args.show)

feature_count = 0
image_count = 0
for image in proj.image_list:
    feature_count += len(image.kp_list)
    image_count += 1

print "Average # of features per image found = %.0f" % (feature_count / image_count)
