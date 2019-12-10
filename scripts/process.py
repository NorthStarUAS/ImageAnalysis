#!/usr/bin/python3

# This is the master ImageAnalysis processing script.  For DJI and
# Sentera cameras it should typically be able to run through with
# default settings and produce a good result with no further input.
#
# If something goes wrong, there are usually specific sub-scripts that
# can be run to fix the problem and then this script can be re-run to
# continue.
#
# If your camera isn't yet supported, you can run a script that mostly
# automates the process of adding a new camera (possibly with a small
# amount of extra info that you can usually research by googling.
#
# If you run into an unsolvable glitch and are willing to share your
# data set, I may be able to look at the issue and make some sort of
# determination or fix.


import argparse
import numpy as np
import os
import pickle
import time

from lib import camera
from lib import groups
from lib.logger import log
from lib import matcher
from lib import match_cleanup
from lib import optimizer
from lib import pose
from lib import project
from lib import render_panda3d
from lib import state

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
parser.add_argument('--orb-max-features', default=20000,
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

# feature matching arguments
parser.add_argument('--match-ratio', default=0.75, type=float,
                    help='match ratio')
parser.add_argument('--min-pairs', default=25, type=int,
                    help='minimum matches between image pairs to keep')
parser.add_argument('--min-dist', default=0, type=float,
                    help='minimum 2d camera distance for pair comparison')
parser.add_argument('--max-dist', default=75, type=float,
                    help='maximum 2d camera distance for pair comparison')
parser.add_argument('--filter', default='gms',
                    choices=['gms', 'homography', 'fundamental', 'essential', 'none'])
parser.add_argument('--min-chain-length', type=int, default=3, help='minimum match chain length (3 recommended)')

# optimizer arguments
parser.add_argument('--group', type=int, default=0, help='group number')
parser.add_argument('--cam-calibration', action='store_true', help='include camera calibration in the optimization.')
parser.add_argument('--refine', action='store_true', help='refine a previous optimization.')

args = parser.parse_args()


############################################################################
log("Step 1: setup the project", fancy=True)
############################################################################

### 1a. initialize a new project workspace

# test if images directory exists
if not os.path.isdir(args.project):
    print("Images directory doesn't exist:", args.project)
    quit()

# create an empty project and save...
proj = project.ProjectMgr(args.project, create=True)
proj.save()

log("Created project:", args.project)

### 1b. intialize camera

if args.camera:
    # specified on command line
    camera_file = args.camera
else:
    # auto detect camera from image meta data
    camera_name, make, model, lens_model = proj.detect_camera()
    camera_file = os.path.join("..", "cameras", camera_name + ".json")
    log("Camera auto-detected:", camera_name, make, model, lens_model)
log("Camera file:", camera_file)

# copy/overlay/update the specified camera config into the existing
# project configuration
cam_node = getNode('/config/camera', True)
tmp_node = PropertyNode()
if props_json.load(camera_file, tmp_node):
    props_json.overlay(cam_node, tmp_node)
    camera.set_mount_params(args.yaw_deg, args.pitch_deg, args.roll_deg)
    # note: dist_coeffs = array[5] = k1, k2, p1, p2, k3
    # ... and save
    proj.save()
else:
    # failed to load camera config file
    if not args.camera:
        log("Camera autodetection failed.  Consider running the new camera script to create a camera config and then try running this script again.")
    else:
        log("Specified camera config not found:", args.camera)
    log("Aborting due to camera detection failure.")
    quit()

state.update("STEP1")


############################################################################
log("Step 2: configure camera poses and per-image meta data files", fancy=True)
############################################################################

log("Configuring images")

# create pose file (if it doesn't already exist, for example sentera
# cameras will generate the pix4d.csv file automatically, dji does not)
pix4d_file = os.path.join(args.project, 'pix4d.csv')
meta_file = os.path.join(args.project, 'image-metadata.txt')
if os.path.exists(pix4d_file):
    log("Found a pose file:", pix4d_file)
elif os.path.exists(meta_file):
    log("Found a pose file:", meta_file)
else:
    pose.make_pix4d(args.project)
    
# load existing image meta data in case this isn't a first run
proj.load_images_info()

pix4d_file = os.path.join(args.project, 'pix4d.csv')
meta_file = os.path.join(args.project, 'image-metadata.txt')
if os.path.exists(pix4d_file):
    pose.setAircraftPoses(proj, pix4d_file, order='rpy',
                          max_angle=args.max_angle)
elif os.path.exists(meta_file):
    pose.setAircraftPoses(proj, meta_file, order='ypr',
                          max_angle=args.max_angle)
else:
    log("Error: no pose file found in image directory:", args.project)
    quit()

# compute the project's NED reference location (based on average of
# aircraft poses)
proj.compute_ned_reference_lla()
ned_node = getNode('/config/ned_reference', True)
log("NED reference location:", ned_node.getFloat('lat_deg'),
           ned_node.getFloat('lon_deg'), ned_node.getFloat('alt_m'))

# set the camera poses (fixed offset from aircraft pose) Camera pose
# location is specfied in ned, so do this after computing the ned
# reference point for this project.
pose.compute_camera_poses(proj)

# save the poses
proj.save_images_info()

# save change to ned reference
proj.save()

state.update("STEP2")


############################################################################
log("Step 3: feature matching", fancy=True)
############################################################################

if not state.check("STEP3a"):
    proj.load_images_info()
    proj.load_match_pairs()

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

    log("detector:", args.detector)
    log("image scale for fearture detection/matching:", args.scale)

    matcher_node = getNode('/config/matcher', True)
    matcher_node.setFloat('match_ratio', args.match_ratio)
    matcher_node.setString('filter', args.filter)
    matcher_node.setInt('min_pairs', args.min_pairs)
    matcher_node.setFloat('min_dist', args.min_dist)
    matcher_node.setFloat('max_dist', args.max_dist)
    matcher_node.setInt('min_chain_len', args.min_chain_length)

    # save any config changes
    proj.save()

    # camera calibration
    K = camera.get_K()
    # print("K:", K)

    log("Matching features")
    
    # fire up the matcher
    matcher.configure()
    matcher.find_matches(proj.image_list, K, transform=args.filter,
                         review=False)

    feature_count = 0
    image_count = 0
    for image in proj.image_list:
        feature_count += image.num_features
        image_count += 1
    log("Average # of features per image found = %.0f" % (feature_count / image_count))

    state.update("STEP3a")

matches_name = os.path.join(proj.analysis_dir, "matches_grouped")

if not state.check("STEP3b"):
    proj.load_images_info()
    proj.load_features(descriptors=False)
    proj.load_match_pairs()
    
    match_cleanup.merge_duplicates(proj)
    match_cleanup.check_for_pair_dups(proj)
    match_cleanup.check_for_1vn_dups(proj)
    matches_direct = match_cleanup.make_match_structure(proj)
    matches_grouped = match_cleanup.link_matches(proj, matches_direct)

    log("Writing full group chain file:", matches_name)
    pickle.dump(matches_grouped, open(matches_name, "wb"))

    state.update("STEP3b")

if not state.check("STEP3c"):
    proj.load_images_info()
    
    K = camera.get_K(optimized=False)
    IK = np.linalg.inv(K)

    log("Loading source matches:", matches_name)
    matches_grouped = pickle.load( open(matches_name, 'rb') )
    match_cleanup.triangulate_srtm(proj, matches_grouped)
    log("Writing triangulated group file:", matches_name)
    pickle.dump(matches_grouped, open(matches_name, "wb"))

    state.update("STEP3c")

if not state.check("STEP3d"):
    proj.load_images_info()

    log("Loading source matches:", matches_name)
    matches = pickle.load( open( matches_name, 'rb' ) )
    log("matched features:", len(matches))

    # compute the group connections within the image set.
    group_list = groups.compute(proj.image_list, matches)
    groups.save(proj.analysis_dir, group_list)

    log("Total images:", len(proj.image_list))
    line = "Group sizes:"
    for g in group_list:
        line += " " + str(len(g))
    log(line)

    log("Counting allocated features...")
    count = 0
    for i, match in enumerate(matches):
        if match[1] >= 0:
            count += 1

    print("Features: %d/%d" % (count, len(matches)))
    
    log("Writing grouped tagged matches:", matches_name)
    pickle.dump(matches, open(matches_name, "wb"))

    state.update("STEP3d")


############################################################################
log("Step 5: Optimization (fit)", fancy=True)
############################################################################

if not state.check("STEP5"):
    proj.load_images_info()

    log("Loading source matches:", matches_name)
    matches = pickle.load( open( matches_name, 'rb' ) )
    log("matched features:", len(matches))

    # load the group connections within the image set
    group_list = groups.load(proj.analysis_dir)

    opt = optimizer.Optimizer(args.project)

    # setup the data structures
    opt.setup( proj, group_list, args.group, matches, optimized=args.refine,
               cam_calib=args.cam_calibration)

    # run the optimization (fit)
    cameras, features, cam_index_map, feat_index_map, \
        fx_opt, fy_opt, cu_opt, cv_opt, distCoeffs_opt \
        = opt.run()

    # update camera poses
    opt.update_camera_poses(proj)

    # update and save the optimized camera calibration
    camera.set_K(fx_opt, fy_opt, cu_opt, cv_opt, optimized=True)
    camera.set_dist_coeffs(distCoeffs_opt.tolist(), optimized=True)
    proj.save()

    # reposition the optimized data set to best fit the original gps
    # locations of the camera poses.
    opt.refit(proj, matches, group_list, args.group)

    # write out the updated match_dict
    log("Writing optimized (fitted) matches:", matches_name)
    pickle.dump(matches, open(matches_name, 'wb'))

    state.update("STEP5")


############################################################################
log("Step 6: Create the map", fancy=True)
############################################################################

if not state.check("STEP6"):
    # load the group connections within the image set
    group_list = groups.load(proj.analysis_dir)

    render_panda3d.build_map(proj, group_list, args.group)
    
    #state.update("STEP6")
