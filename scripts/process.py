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

# camera setup options
parser.add_argument('--camera', help='camera config file')
parser.add_argument('--yaw-deg', type=float, default=0.0,
                    help='camera yaw mounting offset from aircraft')
parser.add_argument('--pitch-deg', type=float, default=-90.0,
                    help='camera pitch mounting offset from aircraft')
parser.add_argument('--roll-deg', type=float, default=0.0,
                    help='camera roll mounting offset from aircraft')

# pose setup options
parser.add_argument('--max-angle', type=float, default=25.0, help='max pitch or roll angle for image inclusion')

# feature detection options
parser.add_argument('--scale', type=float, default=0.4, help='scale images before detecting features, this acts much like a noise filter')
parser.add_argument('--detector', default='SIFT',
                    choices=['SIFT', 'SURF', 'ORB', 'Star'])
parser.add_argument('--surf-hessian-threshold', default=600,
                    help='hessian threshold for surf method')
parser.add_argument('--surf-noctaves', default=4,
                    help='use a bigger number to detect bigger features')
parser.add_argument('--orb-max-features', default=2000,
                    help='maximum ORB features')
parser.add_argument('--grid-detect', default=1,
                    help='run detect on gridded squares for (maybe) better feature distribution, 4 is a good starting value, only affects ORB method')
parser.add_argument('--star-max-size', default=16,
                    help='4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128')
parser.add_argument('--star-response-threshold', default=30)
parser.add_argument('--star-line-threshold-projected', default=10)
parser.add_argument('--star-line-threshold-binarized', default=8)
parser.add_argument('--star-suppress-nonmax-size', default=5)
parser.add_argument('--reject-margin', default=0, help='reject features within this distance of the image outer edge margin')

args = parser.parse_args()


############################################################################
### Step 1: setup the project
############################################################################

### 1a. initialize a new project workspace

# test if images directory exists
if not os.path.isdir(args.project):
    print("Images directory doesn't exist:", args.project)
    quit()

# create an empty project and save...
proj = ProjectMgr.ProjectMgr(args.project, create=True)
proj.save()

logger.log("Created project:", args.project)

### 1b. intialize camera

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

### 1c. create pose file (if it doesn't already exist, for example
###     sentera cameras will generate the pix4d.csv file
###     automatically)

pix4d_file = os.path.join(args.project, 'pix4d.csv')
meta_file = os.path.join(args.project, 'image-metadata.txt')
if os.path.exists(pix4d_file):
    logger.log("Found a pose file:", pix4d_file)
elif os.path.exists(meta_file):
    logger.log("Found a pose file:", meta_file)
else:
    Pose.make_pix4d(args.project)


############################################################################
### Step 2: configure camera poses and per-image meta data files
############################################################################

logger.log("Configuring images")

# load existing image meta data in case this isn't a first run
proj.load_images_info()

pix4d_file = os.path.join(args.project, 'pix4d.csv')
meta_file = os.path.join(args.project, 'image-metadata.txt')
if os.path.exists(pix4d_file):
    Pose.setAircraftPoses(proj, pix4d_file, order='rpy',
                          max_angle=args.max_angle)
elif os.path.exists(meta_file):
    Pose.setAircraftPoses(proj, meta_file, order='ypr',
                          max_angle=args.max_angle)
else:
    logger.log("Error: no pose file found in image directory:", args.project)
    quit()

# compute the project's NED reference location (based on average of
# aircraft poses)
proj.compute_ned_reference_lla()
ned_node = getNode('/config/ned_reference', True)
logger.log("NED reference location:", ned_node.getFloat('lat_deg'),
           ned_node.getFloat('lon_deg'), ned_node.getFloat('alt_m'))

# set the camera poses (fixed offset from aircraft pose) Camera pose
# location is specfied in ned, so do this after computing the ned
# reference point for this project.
Pose.compute_camera_poses(proj)

# save the poses
proj.save_images_info()

# save change to ned reference
proj.save()


############################################################################
### Step 3: detect features and compute descriptors
############################################################################

# setup project detector parameters
detector_node = getNode('/config/detector', True)
detector_node.setString('detector', args.detector)
detector_node.setString('scale', args.scale)
if args.detector == 'SIFT':
    pass
elif args.detector == 'SURF':
    detector_node.setInt('surf_hessian_threshold', args.surf_hessian_threshold)
    detector_node.setInt('surf_noctaves', args.surf_noctaves)
elif args.detector == 'ORB':
    detector_node.setInt('grid_detect', args.grid_detect)
    detector_node.setInt('orb_max_features', args.orb_max_features)
elif args.detector == 'Star':
    detector_node.setInt('star_max_size', args.star_max_size)
    detector_node.setInt('star_response_threshold',
                         args.star_response_threshold)
    detector_node.setInt('star_line_threshold_projected',
                         args.star_response_threshold)
    detector_node.setInt('star_line_threshold_binarized',
                         args.star_line_threshold_binarized)
    detector_node.setInt('star_suppress_nonmax_size',
                         args.star_suppress_nonmax_size)

logger.log("Detecting features.")
logger.log("detector:", args.detector)
logger.log("image scale for detection:", args.scale)

# find features in the full image set
proj.detect_features(scale=args.scale, force=True)

feature_count = 0
image_count = 0
for image in proj.image_list:
    feature_count += len(image.kp_list)
    image_count += 1

logger.log("Average # of features per image found = %.0f" % (feature_count / image_count))

# save project configuration
proj.save()
