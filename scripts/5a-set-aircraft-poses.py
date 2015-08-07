#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import os.path

sys.path.append('../lib')
import Pose
import ProjectMgr

# for all the images in the project image_dir, detect features using the
# specified method and parameters

parser = argparse.ArgumentParser(description='Set the aircraft poses from flight data.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--sentera', help='use the specified sentera image-metadata.txt file')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()

if args.sentera != None:
    Pose.setAircraftPoses(proj, args.sentera)

# compute the project's NED reference location (based on average of
# aircraft poses)
proj.compute_cart_reference_coord()
print "Cartesian reference location:", proj.cart_reference_coord

proj.save()
    
