#!/usr/bin/python3

import argparse
import fnmatch
import os.path

# from the aura-props package
from props import getNode, PropertyNode
import props_json

from lib import ProjectMgr

# set all the various camera configuration parameters

parser = argparse.ArgumentParser(description='Set camera configuration.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--camera', help='camera config file')
parser.add_argument('--yaw-deg', type=float, default=0.0,
                    help='camera yaw mounting offset from aircraft')
parser.add_argument('--pitch-deg', type=float, default=-90.0,
                    help='camera pitch mounting offset from aircraft')
parser.add_argument('--roll-deg', type=float, default=0.0,
                    help='camera roll mounting offset from aircraft')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)

if args.camera:
    # specified on command line
    camera_file = args.camera
else:
    # auto detect camera from image meta data
    camera, make, model, lens_model = proj.detect_camera()
    camera_file = os.path.join("..", "cameras", camera + ".json")
print("Camera:", camera_file)

# copy/overlay/update the specified camera config into the existing
# project configuration
cam_node = getNode('/config/camera', True)
tmp_node = PropertyNode()
if props_json.load(camera_file, tmp_node):
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

    # ... and save
    proj.save()
else:
    # failed to load camera config file
    if not args.camera:
        print("Camera autodetection failed.")
        print("Consider running the new camera script to create a camera config")
        print("and then try running this script again.")
    else:
        print("Provided camera config not found:", args.camera)
