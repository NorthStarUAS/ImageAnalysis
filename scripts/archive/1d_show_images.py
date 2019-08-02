#!/usr/bin/python3

sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import argparse

import sys
sys.path.append('../lib')
import ProjectMgr

# for all the images in the project image_dir, detect features using the
# specified method and parameters

parser = argparse.ArgumentParser(description='Load the project\'s images.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()
#print args

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.show_features_images()
