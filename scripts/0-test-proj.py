#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import os.path

sys.path.append('../lib')
import Image
import Pose
import ProjectMgr

# for all the images in the project image_dir, compute the camera poses from
# the aircraft pose (and camera mounting transform)

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
#proj.load_image_info()
#proj.load_features() # for height/width

#image = proj.image_list[1]
image = Image.Image()
image.set_camera_pose([0.0, 0.0, 0.0], 0.0, -90.0, 0.0)
image.width = 952
image.height = 689

print "camera pose =", image.camera_pose
print "projectPoint2 =", proj.projectPoint2(image, image.camera_pose['quat'], [476, 344.5], 272)
print "projectPoint3 =", proj.projectPoint3(image, image.camera_pose['quat'], [476, 344.5], 272)

print "K\n", proj.cam.K
print "inv(K)\n", proj.cam.IK
