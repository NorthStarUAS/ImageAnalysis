#!/usr/bin/python3

# write out the data in a form useful to pass to the sba (demo) program

# it appears camera poses are basically given as [ R | t ] where R is
# the same R we use throughout and t is the 'tvec'

# todo, run sba and automatically parse output ...

import sys
#sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import argparse
import pickle
import cv2
import math
import numpy as np
import os
import random

sys.path.append('../lib')
import Groups
import Optimizer
import ProjectMgr
import transformations

d2r = math.pi / 180.0       # a helpful constant

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

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
        print("{} <-- {}".format(dst_ned, src_ned))
    A = transformations.superimposition_matrix(src, dst, scale=True)
    print("A:\n{}".formata(A))
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
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()
# proj.load_match_pairs()

grouped_file = os.path.join(args.project, 'matches_grouped' )
direct_file = os.path.join(args.project, 'matches_direct' )
if os.path.isfile( grouped_file ):
    print('Match file:', grouped_file)
    matches = pickle.load( open(grouped_file, "rb") )
elif os.path.isfile( direct_file ):
    print('Match file:', direct_file)
    matches = pickle.load( open(direct_file, "rb") )
else:
    print("Cannot find a matches file to load... aborting")
    quit()
    
print('Match features:', len(matches))

# load the group connections within the image set
groups = Groups.load(args.project)
print(groups)

image_width = proj.image_list[0].width
camw, camh = proj.cam.get_image_params()
scale = float(image_width) / float(camw)
print('scale: {}'.format(scale))

opt = Optimizer.Optimizer(args.project)
cameras, features, cam_index_map, feat_index_map = opt.run( proj.image_list, groups[0], matches, proj.cam.get_K(scale), use_sba=False )

# wipe the sba pose for all images
for image in proj.image_list:
    image.camera_pose_sba = None
    image.placed = False
proj.save_images_meta()

for i, cam in enumerate(cameras):
    image_index = cam_index_map[i]
    image = proj.image_list[image_index]
    orig = image.camera_pose
    print('sba cam: {}'.format(cam))
    if len(cam) == 6:
        Rned2cam, jac = cv2.Rodrigues(cam[0:3])
        tvec = cam[3:6]
    elif len(cam) == 7:
        newq = np.array( [ cam[0], cam[1], cam[2], cam[3] ] )
        Rned2cam = transformations.quaternion_matrix(newq)[:3,:3]
        tvec = np.array( [ cam[4], cam[5], cam[6] ] )
    elif len(cam) == 12:
        newq = np.array( cam[5:9] )
        Rned2cam = transformations.quaternion_matrix(newq)[:3,:3]
        tvec = np.array( cam[9:12] )
        k = np.array( [ [cam[0], 0, cam[1]],
                        [0, cam[3]*cam[0], cam[2]] ] )
        print('est K:\n{}'.format(k))
    elif len(cam) == 17:
        newq = np.array( cam[10:14] )
        tvec = np.array( cam[14:17] )
        Rned2cam = transformations.quaternion_matrix(newq)[:3,:3]
    cam2body = image.get_cam2body()
    Rned2body = cam2body.dot(Rned2cam)
    Rbody2ned = np.matrix(Rned2body).T
    (yaw, pitch, roll) = transformations.euler_from_matrix(Rbody2ned, 'rzyx')
    #print "orig ypr =", image.camera_pose['ypr']
    #print "new ypr =", [yaw/d2r, pitch/d2r, roll/d2r]
    pos = -np.matrix(Rned2cam).T * np.matrix(tvec).T
    newned = pos.T[0].tolist()[0]
    #print "orig ned =", image.camera_pose['ned']
    #print "new ned =", newned
    image.set_camera_pose_sba( ned=newned, ypr=[yaw/d2r, pitch/d2r, roll/d2r] )
    image.placed = True
proj.save_images_meta()
print('Updated the sba poses for all the cameras')

# compare original camera locations with sba camera locations and
# derive a transform matrix to 'best fit' the new camera locations
# over the original ... trusting the original group gps solution as
# our best absolute truth for positioning the system in world
# coordinates.
#
# this can't be done globally, but can only be done for optimized
# group within the set

refit_group_orientations = False
if refit_group_orientations:
    src_list = []
    dst_list = []
    for image in proj.image_list:
        if image.placed:
            # only consider images that are in the fitted set
            ned, ypr, quat = image.get_camera_pose_sba()
            src_list.append(ned)
            ned, ypr, quat = image.get_camera_pose()
            dst_list.append(ned)
    A = get_recenter_affine(src_list, dst_list)

    # extract the rotation matrix (R) from the affine transform
    scale, shear, angles, trans, persp = transformations.decompose_matrix(A)
    print(transformations.decompose_matrix(A))
    R = transformations.euler_matrix(*angles)
    print("R:\n{}".format(R))

    # update the sba camera locations based on best fit
    camera_list = []
    # load current sba poses
    for image in proj.image_list:
        ned, ypr, quat = image.get_camera_pose_sba()
        camera_list.append( ned )
    # refit
    new_cams = transform_points(A, camera_list)

    # update sba poses. FIXME: do we need to update orientation here as
    # well?  Somewhere we worked out the code, but it may not matter all
    # that much ... except for later manually computing mean projection
    # error.
    dist_report = []
    for i, image in enumerate(proj.image_list):
        if not image.placed:
            continue
        ned_orig, ypr_orig, quat_orig = image.get_camera_pose()
        ned, ypr, quat = image.get_camera_pose_sba()
        Rbody2ned = image.get_body2ned_sba()
        # update the orientation with the same transform to keep
        # everything in proper consistent alignment

        newRbody2ned = R[:3,:3].dot(Rbody2ned)
        (yaw, pitch, roll) = transformations.euler_from_matrix(newRbody2ned, 'rzyx')
        image.set_camera_pose_sba(ned=new_cams[i],
                                  ypr=[yaw/d2r, pitch/d2r, roll/d2r])
        dist = np.linalg.norm( np.array(ned_orig) - np.array(new_cams[i]))
        print('image: {}'.format(image.name))
        print('  orig pos: {}'.format(ned_orig))
        print('  fit pos: {}'.format(new_cams[i]))
        print('  dist moved: {}'.format(dist))
        dist_report.append( (dist, image.name) )

        # fixme: are we doing the correct thing here with attitude, or
        # should we transform the point set and then solvepnp() all the
        # camera locations (for now comment out save_meta()
        image.save_meta()

    dist_report = sorted(dist_report,
                         key=lambda fields: fields[0],
                         reverse=False)
    print('Image movement sorted lowest to highest:')
    for report in dist_report:
        print('{} dist: {}'.format(report[1], report[0]))

    # update the sba point locations based on same best fit transform
    # derived from the cameras (remember that 'features' is the point
    # features structure spit out by the SBA process)
    feature_list = []
    for f in features:
        feature_list.append( f.tolist() )
    new_feats = transform_points(A, feature_list)

    # create the matches_sba list (copy) and update the ned coordinate
    matches_sba = []
    for i, feat in enumerate(new_feats):
        match_index = feat_index_map[i]
        match = list(matches[match_index])
        match[0] = feat
        matches_sba.append( match )
else:
    # not refitting group orientations create a matches_sba
    matches_sba = list(matches) # shallow copy
    for i, feat in enumerate(features):
        match_index = feat_index_map[i]
        match = matches_sba[match_index]
        match[0] = feat

# write out the updated match_dict
print("Writing match_sba file ... {} features".format(len(matches_sba)))
pickle.dump(matches_sba, open(os.path.join(args.project, 'matches_sba'), 'wb'))

# temp write out just the points so we can plot them with gnuplot
f = open(os.path.join(args.project, 'sba-plot.txt'), 'w')
for m in matches_sba:
    f.write('%.2f %.2f %.2f\n' % (m[0][0], m[0][1], m[0][2]))
f.close()

# temp write out direct and sba camera positions
f1 = open(os.path.join(args.project, 'cams-direct.txt'), 'w')
f2 = open(os.path.join(args.project, 'cams-sba.txt'), 'w')
for i in groups[0]:
    image = proj.image_list[i]
    ned1, ypr1, quat1 = image.get_camera_pose()
    ned2, ypr2, quat2 = image.get_camera_pose_sba()
    f1.write('%.2f %.2f %.2f\n' % (ned1[0], ned1[1], ned1[2]))
    f2.write('%.2f %.2f %.2f\n' % (ned2[0], ned2[1], ned2[2]))
f1.close()
f2.close()
