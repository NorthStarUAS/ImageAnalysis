#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import os.path
from visual import *

sys.path.append('../lib')
import Pose
import ProjectMgr

# for all the images in the project image_dir, compute the camera poses from
# the aircraft pose (and camera mounting transform)

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()

for image in proj.image_list:
    ned = image.camera_pose['ned']
    ball = sphere(pos=(ned[1],ned[0],-ned[2]), radius=5, color=color.cyan)
