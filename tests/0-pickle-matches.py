#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse

sys.path.append('../lib')
import Matcher
import Pose
import ProjectMgr
import SRTM

# Rest all match point locations to their original direct
# georeferenced locations based on estimated camera pose and
# projection onto DEM earth surface

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
print "Converting match_list save format to 'pickle' ..."
for image in proj.image_list:
    image.load_matches()
    image.save_matches()
