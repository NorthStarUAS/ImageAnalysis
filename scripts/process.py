#!/usr/bin/python3

import argparse
import os

from lib import logger
from lib import Pose
from lib import ProjectMgr

# from the aura-props python package
from props import getNode, PropertyNode
import props_json

parser = argparse.ArgumentParser(description='Create an empty project.')

parser.add_argument('--project', required=True, help='Directory with a set of aerial images.')

parser.add_argument('--camera', help='camera config file')
parser.add_argument('--yaw-deg', type=float, default=0.0,
                    help='camera yaw mounting offset from aircraft')
parser.add_argument('--pitch-deg', type=float, default=-90.0,
                    help='camera pitch mounting offset from aircraft')
parser.add_argument('--roll-deg', type=float, default=0.0,
                    help='camera roll mounting offset from aircraft')

args = parser.parse_args()


############################################################################
# Step 1: setup the project
############################################################################

# 1a. initialize a new project workspace

# test if images directory exists
if not os.path.isdir(args.project):
    print("Images directory doesn't exist:", args.project)
    quit()

# create an empty project and save...
proj = ProjectMgr.ProjectMgr(args.project, create=True)
proj.save()

logger.log("Created project:", args.project)

# 1b. intialize camera

if args.camera:
    # specified on command line
    camera_file = args.camera
else:
    # auto detect camera from image meta data
    camera, make, model, lens_model = proj.detect_camera()
    camera_file = os.path.join("..", "cameras", camera + ".json")
    logger.log("Camera auto-detected:", camera, make, model, lens_model)
logger.log("Camera file:", camera_file)

# copy/overlay/update the specified camera config into the existing
# project configuration
cam_node = getNode('/config/camera', True)
tmp_node = PropertyNode()
if props_json.load(camera_file, tmp_node):
    props_json.overlay(cam_node, tmp_node)
    proj.cam.set_mount_params(args.yaw_deg, args.pitch_deg, args.roll_deg)
    # note: dist_coeffs = array[5] = k1, k2, p1, p2, k3
    # ... and save
    proj.save()
else:
    # failed to load camera config file
    if not args.camera:
        logger.log("Camera autodetection failed.  Consider running the new camera script to create a camera config and then try running this script again.")
    else:
        logger.log("Specified camera config not found:", args.camera)
    logger.log("Aborting due to camera detection failure.")
    quit()

# 1c. create pose file (if it doesn't already exist)

pix4d_file = os.path.join(args.project, 'pix4d.csv')
meta_file = os.path.join(args.project, 'image-metadata.txt')
if not os.path.exists(pix4d_file) and not os.path.exists(meta_file):
    Pose.make_pix4d(args.project)

