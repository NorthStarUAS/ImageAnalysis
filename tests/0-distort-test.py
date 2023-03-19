#!/usr/bin/python

# distortion formula here:
#
# http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html
#
# undistort via cv2 function

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import cv2
import numpy as np
import random

sys.path.append('../lib')
import Image
import Pose
import ProjectMgr

# generate a list of random pixels, undistort them, then distort them to
# test our un/distortion code.

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
image.set_camera_pose([0.0, 0.0, 0.0], 0.0, -90.0, 0.0)
image.width = width_px
image.height = height_px

# k1, k2, p1, p2, k3

# from example online:
# http://stackoverflow.com/questions/11017984/how-to-format-xy-points-for-undistortpoints-with-the-python-cv2-api
#dist_coeffs = np.array([-0.24, 0.095, -0.0004, 0.000089, 0.], dtype=np.float32)

# from laura's camera:
dist_coeffs = np.array([-0.12474347, 0.82940434, -0.01625672, -0.00958748, -1.20843989], dtype=np.float32)

# no distortion?
#dist_coeffs = None

num_points = 20

# generate num points random points
points_orig = np.zeros((num_points,1,2), dtype=np.float32)
for i in range(0, num_points):
    x = random.randrange(image.width)
    y = random.randrange(image.height)
    points_orig[i][0] = [y,x]
#print points_orig

# undistort the points
K = proj.cam.get_K()
points_undistort = cv2.undistortPoints(np.array(points_orig, dtype=np.float32),
                                       K, dist_coeffs, P=K)

def redistort(u, v, dist_coeffs, K):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    print [x, y]
    k1, k2, p1, p2, k3 = dist_coeffs

    # Compute radius^2
    r2 = x**2 + y**2
    r4, r6 = r2**2, r2**3

    # Compute tangential distortion
    dx = 2*p1*x*y + p2*(r2 + 2*x*x)
    dy = p1*(r2 + 2*y*y) + 2*p2*x*y

    # Compute radial factor
    Lr = 1.0 + k1*r2 + k2*r4 + k3*r6

    ud = Lr*x + dx
    vd = Lr*y + dy

    return ud * fx + cx, vd * fy + cy

for i in range(0, num_points):
    ud = points_undistort[i][0]
    rd = redistort(ud[0], ud[1], dist_coeffs, proj.cam.K)
    print "orig = %s  undist = %s redist = %s" % (points_orig[i], ud, rd)

