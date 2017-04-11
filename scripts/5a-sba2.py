#!/usr/bin/python

# write out the data in a form useful to pass to the sba (demo) program

# it appears camera poses are basically given as [ R | t ] where R is
# the same R we use throughout and t is the 'tvec'

# todo, run sba and automatically parse output ...

import sys
sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import argparse
import cPickle as pickle
import cv2
import math
import numpy as np
import random

sys.path.append('../lib')
import Matcher
import ProjectMgr
import SBA
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

# experimental, draw a visual of a match point in all it's images
def draw_match(i, index):
    green = (0, 255, 0)
    red = (0, 0, 255)

    match = matches_direct[i]
    print 'match:', match, 'index:', index
    for j, m in enumerate(match[1:]):
        print ' ', m, proj.image_list[m[0]]
        img = proj.image_list[m[0]]
        # kp = img.kp_list[m[1]].pt # distorted
        kp = img.uv_list[m[1]]  # undistored
        print ' ', kp
        rgb = img.load_rgb()
        h, w = rgb.shape[:2]
        crop = True
        range = 300
        if crop:
            cx = int(round(kp[0]))
            cy = int(round(kp[1]))
            if cx < range:
                xshift = range - cx
                cx = range
            elif cx > (w - range):
                xshift = (w - range) - cx
                cx = w - range
            else:
                xshift = 0
            if cy < range:
                yshift = range - cy
                cy = range
            elif cy > (h - range):
                yshift = (h - range) - cy
                cy = h - range
            else:
                yshift = 0
            print 'size:', w, h, 'shift:', xshift, yshift
            rgb1 = rgb[cy-range:cy+range, cx-range:cx+range]
            if ( j == index ):
                color = red
            else:
                color = green
            cv2.circle(rgb1, (range-xshift,range-yshift), 2, color, thickness=2)
        else:
            scale = 790.0/float(w)
            rgb1 = cv2.resize(rgb, (0,0), fx=scale, fy=scale)
            cv2.circle(rgb1, (int(round(kp[0]*scale)), int(round(kp[1]*scale))), 2, green, thickness=2)
        cv2.imshow(img.name, rgb1)
    print 'waiting for keyboard input...'
    key = cv2.waitKey() & 0xff
    cv2.destroyAllWindows()


proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()
proj.load_match_pairs()

#m = Matcher.Matcher()

matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )
print "direct features:", len(matches_direct)
#matches_sba = pickle.load( open( args.project + "/matches_sba", "rb" ) )
#print "sba features:", len(matches_direct)

# collect/group match chains that refer to the same keypoint

# collect/group match chains that refer to the same keypoint
matches_group = list(matches_direct) # shallow copy
count = 0
done = False
while not done:
    print "Iteration:", count
    count += 1
    matches_new = []
    matches_lookup = {}
    for i, match in enumerate(matches_group):
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
            matches_new.append(list(match)) # shallow copy
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
                    existing.append(list(p)) # shallow copy
                    matches_lookup[key] = index
            # print "new:", existing
            # print 
    if len(matches_new) == len(matches_group):
        done = True
    else:
        matches_group = list(matches_new) # shallow copy
print "unique features (after grouping):", len(matches_group)

# count the match groups that are longer than just pairs
group_count = 0
for m in matches_group:
    if len(m) > 3:
        group_count += 1

print "Number of groupings:", group_count

# add some grouped matches to the original matches_direct
# count = 0
# matches_direct = []
# while True:
#     index = random.randrange(len(matches_tmp))
#     print "index:", index
#     match = matches_tmp[index]
#     if count > 100:
#         break
#     if len(match) > 3:
#         # append whole match: matches_direct.append(match)
        
#         # append pair combinations
#         ned = match[0]
#         for i in range(1, len(match)-1):
#             for j in range(i+1, len(match)):
#                 print i, j
#                 matches_direct.append( [ ned, match[i], match[j] ] )
#         #draw_match(len(matches_direct)-1, -1)
#         count += 1
        
# add all the grouped matches pair-wise
# matches_direct = []
# for match in matches_tmp:
#     # append pair combinations
#     ned = match[0]
#     for i in range(1, len(match)-1):
#         for j in range(i+1, len(match)):
#             print i, j
#             matches_direct.append( [ ned, match[i], match[j] ] )
#     #draw_match(len(matches_direct)-1, -1)

# now forget all that and just add the matches referencing more than 2 views
# matches_direct = []
# for m in matches_tmp:
#     if len(m) > 3:
#         matches_direct.append(m)

print "Original match connector"
Matcher.groupByConnections(proj.image_list)

groups = Matcher.simpleGrouping(proj.image_list, matches_group)

image_width = proj.image_list[0].width
camw, camh = proj.cam.get_image_params()
scale = float(image_width) / float(camw)
print 'scale:', scale

sba = SBA.SBA(args.project)
sba.prepair_data( proj.image_list, groups[0], matches_direct, proj.cam.get_K(scale) )
cameras, features, cam_index_map, feat_index_map = sba.run_live(mode='')

# wipe the sba pose for all images
for image in proj.image_list:
    image.camera_pose_sba = None
    image.placed = False
proj.save_images_meta()

for i, cam in enumerate(cameras):
    image_index = cam_index_map[i]
    image = proj.image_list[image_index]
    orig = image.camera_pose
    print cam
    if len(cam) == 7:
        newq = np.array( [ cam[0], cam[1], cam[2], cam[3] ] )
        tvec = np.array( [ cam[4], cam[5], cam[6] ] )
    elif len(cam) == 12:
        newq = np.array( cam[5:9] )
        tvec = np.array( cam[9:12] )
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

# compare original camera locations with sba camera locations and
# derive a transform matrix to 'best fit' the new camera locations
# over the original ... trusting the original group gps solution as
# our best absolute truth for positioning the system in world
# coordinates.
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
print transformations.decompose_matrix(A)
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
max_moved_image = None
max_moved_dist = 0
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
    print 'image:', image.name
    print '  orig pos:', ned_orig
    print '  fit pos:', new_cams[i]
    print '  dist moved:', dist
    if dist > max_moved_dist:
        max_moved_dist = dist
        max_moved_image = image
        
    # fixme: are we doing the correct thing here with attitude, or
    # should we transform the point set and then solvepnp() all the
    # camera locations (for now comment out save_meta()
    image.save_meta()

print 'Image that moved the most:', max_moved_image.name
print 'Move distance (m):', max_moved_dist

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
    match = list(matches_direct[match_index])
    match[0] = feat
    matches_sba.append( match )
    
# write out the updated match_dict
print "Writing match_sba file ...", len(matches_sba), 'features'
pickle.dump(matches_sba, open(args.project + "/matches_sba", "wb"))

# temp write out just the points so we can plot them with gnuplot
f = open('sba-plot.txt', 'w')
for m in matches_sba:
    f.write('%.2f %.2f %.2f\n' % (m[0][0], m[0][1], m[0][2]))
f.close()
