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
parser.add_argument('--image', help='show specific image')

args = parser.parse_args()
#print args

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()

if args.image:
    image = proj.findImageByName(args.image)
    proj.show_features_image(image)
else:
    proj.show_features_images()
