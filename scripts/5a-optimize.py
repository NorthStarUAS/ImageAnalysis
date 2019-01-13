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
import sys

sys.path.append('../lib')
import Groups
import Optimizer
import ProjectMgr
import transformations

d2r = math.pi / 180.0
r2d = 180.0 / math.pi

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--area', required=True, help='sub area directory')
parser.add_argument('--refine', action='store_true', help='refine a previous optimization.')

args = parser.parse_args()

# return a 3d affine tranformation between current camera locations
# and original camera locations.
def get_recenter_affine(src_list, dst_list):
    print('get_recenter_affine():')
    src = [[], [], [], []]      # current camera locations
    dst = [[], [], [], []]      # original camera locations
    for i in range(len(src_list)):
        src_ned = src_list[i]
        src[0].append(src_ned[0])
        src[1].append(src_ned[1])
        src[2].append(src_ned[2])
        src[3].append(1.0)
        dst_ned = dst_list[i]
        dst[0].append(dst_ned[0])
        dst[1].append(dst_ned[1])
        dst[2].append(dst_ned[2])
        dst[3].append(1.0)
        # print("{} <-- {}".format(dst_ned, src_ned))
    A = transformations.superimposition_matrix(src, dst, scale=True)
    print("A:\n", A)
    return A

# transform a point list given an affine transform matrix
def transform_points( A, pts_list ):
    src = [[], [], [], []]
    for p in pts_list:
        src[0].append(p[0])
        src[1].append(p[1])
        src[2].append(p[2])
        src[3].append(1.0)
    dst = A.dot( np.array(src) )
    result = []
    for i in range(len(pts_list)):
        result.append( [ float(dst[0][i]),
                         float(dst[1][i]),
                         float(dst[2][i]) ] )
    return result

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_area_info(args.area)

area_dir = os.path.join(args.project, args.area)

source_file = os.path.join(area_dir, 'matches_grouped' )
#source_file = os.path.join(area_dir, 'matches_direct' )
opt_file = os.path.join(area_dir, 'matches_opt')
if args.refine and os.path.isfile( opt_file ):
    print('Match file:', opt_file)
    matches = pickle.load( open(opt_file, "rb") )
elif os.path.isfile( source_file ):
    print('Match file:', source_file)
    matches = pickle.load( open(source_file, "rb") )
else:
    print("Cannot find a matches file to load... aborting")
    quit()
    
print('Match features:', len(matches))

# load the group connections within the image set
groups = Groups.load(area_dir)
# sort from smallest to largest
groups.sort(key=len)

# make a single flat list of all images named in any group
flat_group = []
for group in groups:
    for name in group:
        index = proj.findIndexByName(name)
        flat_group.append(index)
        
opt = Optimizer.Optimizer(args.project)
opt.setup( proj, flat_group, matches, optimized=args.refine )
cameras, features, cam_index_map, feat_index_map, fx_opt, fy_opt, cu_opt, cv_opt, distCoeffs_opt = opt.run()

# mark all the optimized poses as invalid
for image in proj.image_list:
    opt_cam_node = image.node.getChild('camera_pose_opt', True)
    opt_cam_node.setBool('valid', False)

for i, cam in enumerate(cameras):
    image_index = cam_index_map[i]
    image = proj.image_list[image_index]
    ned_orig, ypr_orig, quat_orig = image.get_camera_pose()
    print('optimized cam:', cam)
    rvec = cam[0:3]
    tvec = cam[3:6]
    Rned2cam, jac = cv2.Rodrigues(rvec)
    cam2body = image.get_cam2body()
    Rned2body = cam2body.dot(Rned2cam)
    Rbody2ned = np.matrix(Rned2body).T
    (yaw, pitch, roll) = transformations.euler_from_matrix(Rbody2ned, 'rzyx')
    #print "orig ypr =", image.camera_pose['ypr']
    #print "new ypr =", [yaw/d2r, pitch/d2r, roll/d2r]
    pos = -np.matrix(Rned2cam).T * np.matrix(tvec).T
    newned = pos.T[0].tolist()[0]
    print(image.name, ned_orig, '->', newned, 'dist:', np.linalg.norm(np.array(ned_orig) - np.array(newned)))
    image.set_camera_pose( newned, yaw*r2d, pitch*r2d, roll*r2d, opt=True )
    image.placed = True
proj.save_images_info()
print('Updated the optimized camera poses.')

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
    for group in groups:
        if len(group) < 10:
            # skip group sizes < 10
            continue
        print('refitting group size:', len(group))
        src_list = []
        dst_list = []
        # only consider images that are in the current   group
        for name in group:
            image = proj.findImageByName(name)
            ned, ypr, quat = image.get_camera_pose(opt=True)
            src_list.append(ned)
            ned, ypr, quat = image.get_camera_pose()
            dst_list.append(ned)
        A = get_recenter_affine(src_list, dst_list)

        # extract the rotation matrix (R) from the affine transform
        scale, shear, angles, trans, persp = transformations.decompose_matrix(A)
        print('  scale:', scale)
        print('  shear:', shear)
        print('  angles:', angles)
        print('  translate:', trans)
        print('  perspective:', persp)
        R = transformations.euler_matrix(*angles)
        print("R:\n{}".format(R))

        # fixme (just group):
        
        # update the optimized camera locations based on best fit
        camera_list = []
        # load optimized poses
        for image in proj.image_list:
            if image.name in group:
                ned, ypr, quat = image.get_camera_pose(opt=True)
            else:
                # this is just fodder to match size/index of the lists
                ned, ypr, quat = image.get_camera_pose()
            camera_list.append( ned )

        # refit
        new_cams = transform_points(A, camera_list)

        # update position
        for i, image in enumerate(proj.image_list):
            if not image.name in group:
                continue
            ned, [y, p, r], quat = image.get_camera_pose(opt=True)
            image.set_camera_pose(new_cams[i], y, p, r, opt=True)
        proj.save_images_info()

        if True:
            # update optimized pose orientation.
            dist_report = []
            for i, image in enumerate(proj.image_list):
                if not image.name in group:
                    continue
                ned_orig, ypr_orig, quat_orig = image.get_camera_pose()
                ned, ypr, quat = image.get_camera_pose(opt=True)
                Rbody2ned = image.get_body2ned(opt=True)
                # update the orientation with the same transform to keep
                # everything in proper consistent alignment

                newRbody2ned = R[:3,:3].dot(Rbody2ned)
                (yaw, pitch, roll) = transformations.euler_from_matrix(newRbody2ned, 'rzyx')
                image.set_camera_pose(new_cams[i], yaw*r2d, pitch*r2d, roll*r2d,
                                      opt=True)
                dist = np.linalg.norm( np.array(ned_orig) - np.array(new_cams[i]))
                print('image: {}'.format(image.name))
                print('  orig pos: {}'.format(ned_orig))
                print('  fit pos: {}'.format(new_cams[i]))
                print('  dist moved: {}'.format(dist))
                dist_report.append( (dist, image.name) )
            proj.save_images_info()

            dist_report = sorted(dist_report,
                                 key=lambda fields: fields[0],
                                 reverse=False)
            print('Image movement sorted lowest to highest:')
            for report in dist_report:
                print('{} dist: {}'.format(report[1], report[0]))

        # tranform the optimized point locations using the same best
        # fit transform for the camera locations.
        new_feats = transform_points(A, features)
        
        # update any of the transformed feature locations that have
        # membership in the currently processing group back to the
        # master match structure.  Note we process groups in order of
        # little to big so if a match is in more than one group it
        # follows the larger group.
        for i, feat in enumerate(new_feats):
            match_index = feat_index_map[i]
            match = matches_opt[match_index]
            in_group = False
            for m in match[1:]:
                if proj.image_list[m[0]].name in group:
                    in_group = True
                    break
            if in_group:
                #print(' before:', match)
                match[0] = feat
                #print(' after:', match)
else:
    # not refitting group orientations, just copy over optimized
    # coordinates
    for i, feat in enumerate(features):
        match_index = feat_index_map[i]
        match = matches_opt[match_index]
        match[0] = feat

# write out the updated match_dict
print('Writing matches_opt file:', len(matches_opt), 'features')
pickle.dump(matches_opt, open(os.path.join(area_dir, 'matches_opt'), 'wb'))

#proj.cam.set_K(fx_opt/scale[0], fy_opt/scale[0], cu_opt/scale[0], cv_opt/scale[0], optimized=True)
#proj.save()

# temp write out just the points so we can plot them with gnuplot
f = open(os.path.join(area_dir, 'opt-plot.txt'), 'w')
for m in matches_opt:
    f.write('%.2f %.2f %.2f\n' % (m[0][0], m[0][1], m[0][2]))
f.close()

# temp write out direct and optimized camera positions
f1 = open(os.path.join(area_dir, 'cams-direct.txt'), 'w')
f2 = open(os.path.join(area_dir, 'cams-opt.txt'), 'w')
for name in groups[0]:
    image = proj.findImageByName(name)
    ned1, ypr1, quat1 = image.get_camera_pose()
    ned2, ypr2, quat2 = image.get_camera_pose(opt=True)
    f1.write('%.2f %.2f %.2f\n' % (ned1[0], ned1[1], ned1[2]))
    f2.write('%.2f %.2f %.2f\n' % (ned2[0], ned2[1], ned2[2]))
f1.close()
f2.close()
