#!/usr/bin/python

# write out the data in a form useful to pass to the sba (demo) program

# it appears camera poses are basically given as [ R | t ] where R is
# the same R we use throughout and t is the 'tvec'

# todo, run sba and automatically parse output ...

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

import argparse
import cPickle as pickle
import math
import numpy as np

sys.path.append('../lib')
import Matcher
import ProjectMgr
import SBA1
import transformations

d2r = math.pi / 180.0       # a helpful constant

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

# return a 3d affine tranformation between current camera locations
# and original camera locations.
def get_recenter_affine(src_list, dst_list):
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
        print "%s <-- %s" % (dst_ned, src_ned)
    A = transformations.superimposition_matrix(src, dst, scale=True)
    print "A:\n", A
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

#m = Matcher.Matcher()

matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )
print "unique features:", len(matches_direct)

image_width = proj.image_list[0].width
camw, camh = proj.cam.get_image_params()
scale = float(image_width) / float(camw)
print 'scale:', scale

sba = SBA1.SBA1(args.project)
sba.prepair_data( proj.image_list, matches_direct, proj.cam.get_K(scale) )
cameras, features = sba.run_live()

for i, image in enumerate(proj.image_list):
    orig = image.camera_pose
    new = cameras[i]
    newq = np.array( [ new[0], new[1], new[2], new[3] ] )
    tvec = np.array( [ new[4], new[5], new[6] ] )
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

# now count how many features show up in each image
for i in proj.image_list:
    i.feature_count = 0
for i, match in enumerate(matches_direct):
    for j, p in enumerate(match[1:]):
        image = proj.image_list[ p[0] ]
        image.feature_count += 1

# compare original camera locations with sba camera locations and
# derive a transform matrix to 'best fit' the new camera locations
# over the original ... trusting the original group gps solution as
# our best absolute truth for positioning the system in world
# coordinates.
src_list = []
dst_list = []
for image in proj.image_list:
    if image.feature_count >= 25:
        # only consider images that are in the fitted set
        ned, ypr, quat = image.get_camera_pose_sba()
        src_list.append(ned)
        ned, ypr, quat = image.get_camera_pose()
        dst_list.append(ned)
A = get_recenter_affine(src_list, dst_list)

# extract the rotation matrix (R) from the affine transform
scale, shear, angles, trans, persp = transformations.decompose_matrix(A)
R = transformations.euler_matrix(*angles)
print "R:\n", R

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
for i, image in enumerate(proj.image_list):
    ned_orig, ypr_orig, quat_orig = image.get_camera_pose()
    ned, ypr, quat = image.get_camera_pose_sba()
    Rbody2ned = image.get_body2ned_sba()
    # update the orientation with the same transform to keep
    # everything in proper consistent alignment
    newRbody2ned = R[:3,:3].dot(Rbody2ned)
    (yaw, pitch, roll) = transformations.euler_from_matrix(newRbody2ned, 'rzyx')
    image.set_camera_pose_sba(ned=new_cams[i],
                              ypr=[yaw/d2r, pitch/d2r, roll/d2r])
    print 'image:', image.name
    print '  orig pos:', ned_orig
    print '  fit pos:', new_cams[i]
    print '  dist moved:', np.linalg.norm( np.array(ned_orig) - np.array(new_cams[i]))
    image.save_meta()
    
# update the sba point locations based on same best fit transform
# derived from the cameras (remember that 'features' is the point
# features structure spit out by the SBA process)
feature_list = []
for f in features:
    feature_list.append( f.tolist() )
new_feats = transform_points(A, feature_list)

# create the matches_sba list (copy) and update the ned coordinate
matches_sba = list(matches_direct)
for i, match in enumerate(matches_sba):
    #print type(new_feats[i])
    matches_sba[i][0] = new_feats[i]

# write out the updated match_dict
print "Writing match_sba file ...", len(matches_sba), 'features'
pickle.dump(matches_sba, open(args.project + "/matches_sba", "wb"))

# collect/group match chains that refer to the same keypoint

matches_tmp = list(matches_sba)

count = 0
done = False
while not done:
    print "Iteration:", count
    count += 1
    matches_new = []
    matches_lookup = {}
    for i, match in enumerate(matches_tmp):
        # scan if any of these match points have been previously seen
        # and record the match index
        index = -1
        for p in match[1:]:
            key = "%d-%d" % (p[0], p[1])
            if key in matches_lookup:
                index = matches_lookup[key]
                break
        if index < 0:
            # not found, append to the new list
            for p in match[1:]:
                key = "%d-%d" % (p[0], p[1])
                matches_lookup[key] = len(matches_new)
            matches_new.append(match)
        else:
            # found a previous reference, append these match items
            existing = matches_new[index]
            # only append items that don't already exist in the early
            # match, and only one match per image (!)
            for p in match[1:]:
                key = "%d-%d" % (p[0], p[1])
                found = False
                for e in existing[1:]:
                    if p[0] == e[0]:
                        found = True
                        break
                if not found:
                    # add
                    existing.append(p)
                    matches_lookup[key] = index
            # print "new:", existing
            # print 
    if len(matches_new) == len(matches_tmp):
        done = True
    else:
        matches_tmp = matches_new

matches_group = matches_tmp

# write out the updated match_dict
print "Writing match_group file ...", len(matches_group), 'features'
pickle.dump(matches_group, open(args.project + "/matches_group", "wb"))
