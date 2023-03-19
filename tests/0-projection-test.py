#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse

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

# setup the camera with sentera 3 Mpx params
width_px = 3808
height_px = 2754
fx = fy = 4662.25 # [pixels] - where 1 pixel = 1.67 micrometer
horiz_mm = width_px * 1.67 * 0.001
vert_mm = height_px * 1.67 * 0.001
focal_len_mm = (fx * horiz_mm) / width_px
proj.cam.set_lens_params(horiz_mm, vert_mm, focal_len_mm)
proj.cam.set_calibration_params(fx, fy, width_px/2, height_px/2,
                                [0.0, 0.0, 0.0, 0.0, 0.0], 0.0)
proj.cam.set_calibration_std(0.0, 0.0, 0.0, 0.0,
                             [0.0, 0.0, 0.0, 0.0, 0.0], 0.0)
proj.cam.set_image_params(width_px, height_px)
proj.cam.set_mount_params(0.0, -90.0, 0.0)

#image = proj.image_list[1]
image = Image.Image()
image.set_camera_pose([0.0, 0.0, 0.0], [0.0, -90.0, 0.0])
image.width = 3808
image.height = 2754

px = [0, 0]
print "camera pose =", image.camera_pose
print "projectPoint2:\n", proj.projectPoint2(image, image.camera_pose['quat'], px, 272)
print "projectVector:\n", proj.projectVector(image, image.camera_pose['quat'], px)
