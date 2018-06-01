#!/usr/bin/python3

import sys

import argparse
import cv2
import fnmatch
import os.path

sys.path.append('../lib')
import ProjectMgr

# from the aura-props package
from props import getNode, PropertyNode
import props_json

# set all the various camera configuration parameters

parser = argparse.ArgumentParser(description='Set camera configuration.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--camera', required=True, help='camera config file')

parser.add_argument('--yaw-deg', required=True, type=float,
                    help='camera yaw mounting offset from aircraft')
parser.add_argument('--pitch-deg', required=True, type=float,
                    help='camera pitch mounting offset from aircraft')
parser.add_argument('--roll-deg', required=True, type=float,
                    help='camera roll mounting offset from aircraft')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)

# copy/overlay/update the specified camera config into the existing
# project configuration
cam_node = getNode('/config/camera', True)
tmp_node = PropertyNode()
props_json.load(args.camera, tmp_node)
for child in tmp_node.getChildren(expand=False):
    if tmp_node.isEnum(child):
        # print(child, tmp_node.getLen(child))
        for i in range(tmp_node.getLen(child)):
            cam_node.setFloatEnum(child, i, tmp_node.getFloatEnum(child, i))
    else:
        # print(child, type(tmp_node.__dict__[child]))
        child_type = type(tmp_node.__dict__[child])
        if child_type is float:
            cam_node.setFloat(child, tmp_node.getFloat(child))
        elif child_type is int:
            cam_node.setInt(child, tmp_node.getInt(child))
        elif child_type is str:
            cam_node.setString(child, tmp_node.getString(child))
        else:
            print('Unknown child type:', child, child_type)

proj.cam.set_mount_params(args.yaw_deg, args.pitch_deg, args.roll_deg)

# note: dist_coeffs = array[5] = k1, k2, p1, p2, k3

# the following counts as cruft, but saving it for now until I can
# migrate these# over to official camera database files.

# if args.sentera_3M:
#     # need confirmation on these numbers because they don't all exactly jive
#     # 1 pixel = 1.67 micrometer
#     # horiz-mm = 6.36 (?)
#     # vert-mm = 4.6 (?)
#     # focal-len-mm = 8 (?)
#     # 3808 x 2754 (?)1
#     width_px = 3808
#     height_px = 2754
#     fx = fy = 4662.25 # [pixels] - where 1 pixel = 1.67 micrometer
#     ccd_width_mm = width_px * 1.67 * 0.001
#     ccd_height_mm = height_px * 1.67 * 0.001
#     focal_len_mm = (fx * ccd_width_mm) / width_px
#     dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
#     proj.cam.set_lens_params(ccd_width_mm, ccd_height_mm, focal_len_mm)
#     proj.cam.set_K(fx, fy, width_px/2, height_px/2)
#     proj.cam_set_dist_coeffs(dist_coeffs)
#     proj.cam.set_image_params(width_px, height_px)
#     proj.cam.set_mount_params(0.0, -90.0, 0.0)
# elif args.sentera_global:
#     width_px = 1248
#     height_px = 950
#     fx = fy = 1613.33 # [pixels] - where 1 pixel = 3.75 micrometer
#     ccd_width_mm = width_px * 3.75 * 0.001
#     ccd_height_mm = height_px * 3.75 * 0.001
#     focal_len_mm = (fx * ccd_width_mm) / width_px
#     # dist_coeffs = array[5] = k1, k2, p1, p2, k3
#     dist_coeffs = [-0.387486, 0.211065, 0.0, 0.0, 0.0]
#     proj.cam.set_lens_params(ccd_width_mm, ccd_height_mm, focal_len_mm)
#     proj.cam.set_K(fx, fy, width_px/2, height_px/2)
#     proj.cam.set_dist_coeffs(dist_coeffs)
#     proj.cam.set_image_params(width_px, height_px)
#     proj.cam.set_mount_params(0.0, -90.0, 0.0)
# elif args.sentera_global_aem:
#     width_px = 1248
#     height_px = 950
#     fx = 1612.26
#     fy = 1610.56
#     cu = 624
#     cv = 475
#     # [pixels] - where 1 pixel = 3.75 micrometer
#     ccd_width_mm = width_px * 3.75 * 0.001
#     ccd_height_mm = height_px * 3.75 * 0.001
#     focal_len_mm = (fx * ccd_width_mm) / width_px
#     # dist_coeffs = array[5] = k1, k2, p1, p2, k3
#     dist_coeffs = [-0.37158252, 0.4333338, 0.0, 0.0, -1.40601407]
#     proj.cam.set_lens_params(ccd_width_mm, ccd_height_mm, focal_len_mm)
#     proj.cam.set_K(fx, fy, width_px/2, height_px/2)
#     proj.cam.set_dist_coeffs(dist_coeffs)
#     proj.cam.set_image_params(width_px, height_px)
#     proj.cam.set_mount_params(0.0, -90.0, 0.0)

# ... and save
proj.save()
