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

# set all the various camera configuration parameters

parser = argparse.ArgumentParser(description='Set camera configuration.')
parser.add_argument('--project', required=True, help='project directory')

parser.add_argument('--horiz-mm', type=float,
                    help='imager horizontal size (mm)')
parser.add_argument('--vert-mm', type=float,
                    help='imager vertical size (mm)')
parser.add_argument('--focal-len-mm', type=float,
                    help='lens focal length (mm)')

parser.add_argument('--fx', type=float, help='fx')
parser.add_argument('--fy', type=float, help='fy')
parser.add_argument('--cu', type=float, help='cu')
parser.add_argument('--cv', type=float, help='cv')
parser.add_argument('--kcoeffs', type=float, nargs=5,
                    help='distortion parameters k1 ... k5')
parser.add_argument('--skew', type=float, help='skew')

parser.add_argument('--fx-std', type=float, help='fx standard deviation')
parser.add_argument('--fy-std', type=float, help='fy standard deviation')
parser.add_argument('--cu-std', type=float, help='cu standard deviation')
parser.add_argument('--cv-std', type=float, help='cv standard deviation')
parser.add_argument('--kcoeffs-std', type=float, nargs=5,
                    help='distortion paramsstandard deviation k1_std ... k5_std')
parser.add_argument('--skew-std', type=float, help='skew standard deviation')

parser.add_argument('--width-px', type=int,
                    help='expected image width in pixels')
parser.add_argument('--height-px',
                    type=int, help='expected image height in pixels')

parser.add_argument('--yaw-deg', type=float,
                    help='camera yaw mounting offset from aircraft')
parser.add_argument('--pitch-deg', type=float,
                    help='camera pitch mounting offset from aircraft')
parser.add_argument('--roll-deg', type=float,
                    help='camera roll mounting offset from aircraft')

parser.add_argument('--sentera3M', action='store_true',
                    help='settings for Sentera 3.1 Mpx 3808x2754 camera')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)

if args.horiz_mm or args.vert_mm or args.focal_len_mm:
    if args.horiz_mm and args.vert_mm and args.focal_len_mm:
        proj.cam.set_lens_params(args.horiz_mm, args.vert_mm, args.focal_len_mm)
    else:
        print "Must set horiz-mm, vert-mm, and focal-len-mm together"

if args.fx or args.fy or args.cu or args.cv or args.kcoeffs or args.skew:
    if args.fx and args.fy and args.cu and args.cv and args.kcoeffs and args.skew:
        proj.cam.set_calibration_params(args.fx, args.fy, args.cu, args.cv, args.kcoeffs, args.skew)
    else:
         print "Must set fx, fy, cu, cv, kcoeffs, and skew together"
   
if args.fx_std or args.fy_std or args.cu_std or args.cv_std or args.kcoeffs_std or args.skew_std:
    if args.fx_std and args.fy_std and args.cu_std and args.cv_std and args.kcoeffs_std and args.skew_std:
        proj.cam.set_calibration_std(args.fx_std, args.fy_std, args.cu_std, args.cv_std, args.kcoeffs_std, args.skew_std)
    else:
         print "Must set fx-std, fy-std, cu-std, cv-std, kcoeffs-std, and skew-std together"
   
if args.width_px or args.height_px:
    if args.width_px and args.height_px:
        proj.cam.set_image_params(args.with_px, args.height_px)
    else:
         print "Must set width-px and height-px together"

if args.yaw_deg != None or args.pitch_deg != None or args.roll_deg != None:
    if args.yaw_deg != None and args.pitch_deg != None and args.roll_deg != None:
        proj.cam.set_mount_params(args.yaw_deg, args.pitch_deg, args.roll_deg)
    else:
         print "Must set yaw-deg, pitch-deg, and roll-deg together"

if args.sentera3M:
    # need confirmation on these numbers because they don't all exactly jive
    # 1 pixel = 1.67 micrometer
    # horiz-mm = 6.36 (?)
    # vert-mm = 4.6 (?)
    # focal-len-mm = 8 (?)
    # 3808 x 2754 (?)1
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

proj.save()
