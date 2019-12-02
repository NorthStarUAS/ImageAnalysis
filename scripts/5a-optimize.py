#!/usr/bin/python3

# Run the optimization step to place all the features and camera poses
# in a way that minimizes the mean reprojection error for the
# collective data set.

import argparse
import pickle
import cv2
import math
import numpy as np
import os

from lib import groups
from lib import Optimizer
from lib import ProjectMgr
from lib import transformations

d2r = math.pi / 180.0
r2d = 180.0 / math.pi

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--group', type=int, default=0, help='group number')
parser.add_argument('--refine', action='store_true', help='refine a previous optimization.')
parser.add_argument('--cam-calibration', action='store_true', help='include camera calibration in the optimization.')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()

source_file = os.path.join(proj.analysis_dir, 'matches_grouped' )
print('Match file:', source_file)
matches = pickle.load( open(source_file, "rb") )
print('Match features:', len(matches))

# load the group connections within the image set
group_list = groups.load(proj.analysis_dir)
# sort from smallest to largest: groups.sort(key=len)

opt = Optimizer.Optimizer(args.project)

opt.setup( proj, group_list, args.group, matches, optimized=args.refine,
           cam_calib=args.cam_calibration)

cameras, features, cam_index_map, feat_index_map, fx_opt, fy_opt, cu_opt, cv_opt, distCoeffs_opt = opt.run()

opt.update_camera_poses(proj)

# update and save the optimized camera calibration
proj.cam.set_K(fx_opt, fy_opt, cu_opt, cv_opt, optimized=True)
proj.cam.set_dist_coeffs(distCoeffs_opt.tolist(), optimized=True)
proj.save()

# compare original camera locations with optimized camera locations and
# derive a transform matrix to 'best fit' the new camera locations
# over the original ... trusting the original group gps solution as
# our best absolute truth for positioning the system in world
# coordinates.
#
# each optimized group needs a separate/unique fit

matches_opt = list(matches) # shallow copy
refit_group_orientations = True
if refit_group_orientations:
    opt.refit(proj, matches, group_list, args.group)
else:
    # not refitting group orientations, just copy over optimized
    # coordinates
    for i, feat in enumerate(features):
        match_index = feat_index_map[i]
        match = matches_opt[match_index]
        match[0] = feat

# write out the updated match_dict
print('Updating matches file:', len(matches_opt), 'features')
pickle.dump(matches_opt, open(source_file, 'wb'))

#proj.cam.set_K(fx_opt/scale[0], fy_opt/scale[0], cu_opt/scale[0], cv_opt/scale[0], optimized=True)
#proj.save()

# temp write out just the points so we can plot them with gnuplot
f = open(os.path.join(proj.analysis_dir, 'opt-plot.txt'), 'w')
for m in matches_opt:
    f.write('%.2f %.2f %.2f\n' % (m[0][0], m[0][1], m[0][2]))
f.close()

# temp write out direct and optimized camera positions
f1 = open(os.path.join(proj.analysis_dir, 'cams-direct.txt'), 'w')
f2 = open(os.path.join(proj.analysis_dir, 'cams-opt.txt'), 'w')
for name in group_list[args.group]:
    image = proj.findImageByName(name)
    ned1, ypr1, quat1 = image.get_camera_pose()
    ned2, ypr2, quat2 = image.get_camera_pose(opt=True)
    f1.write('%.2f %.2f %.2f\n' % (ned1[1], ned1[0], -ned1[2]))
    f2.write('%.2f %.2f %.2f\n' % (ned2[1], ned2[0], -ned2[2]))
f1.close()
f2.close()
