#!/usr/bin/env python3

import sys

import argparse
import cv2
import fnmatch
import os.path

from lib import project

# for all the images in the project image_dir, detect features using the
# specified method and parameters

parser = argparse.ArgumentParser(description='Load the project\'s images.')
parser.add_argument('project', help='project directory')
parser.add_argument('--image', help='show specific image')
parser.add_argument('--index', type=int, help='show specific image by index')

args = parser.parse_args()
#print args

proj = project.ProjectMgr(args.project)
proj.load_images_info()
proj.load_features()

if args.image:
    image = proj.findImageByName(args.image)
    proj.show_features_image(image)
elif args.index:
    proj.show_features_image(proj.image_list[args.index])
else:
    proj.show_features_images()
