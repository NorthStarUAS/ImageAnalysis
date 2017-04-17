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

# constants
d2r = math.pi / 180.0
r2d = 180 / math.pi

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--ground', required=True, type=float, help='ground elevation (estimate) in meters')

args = parser.parse_args()

# return a 3d affine tranformation between current camera locations
# and original camera locations.
def get_recenter_affine(src_list, dst_list):
    if len(src_list) < 3:
        T = transformations.translation_matrix([0.0, 0.0, 0.0])
        R = np.identity(4)
        S = transformations.scale_matrix(1.0)
        A = transformations.concatenate_matrices(T, R, S)
    else:
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

# for any match that references a placed image, but hasn't been
# triangulated (meaning it only shows up in one placed image so far)
# use the 'projected' feature location
def update_match_coordinates(matches, placed_images):
    # null out all match locations
    for match in matches:
        match[0] = None
        
    # average of projection placement for placed images
    for match in matches:
        if match[0] == None:
            # only consider not-already-triangulated features
            sum = np.zeros(3)
            count = 0;
            for m in match[1:]:
                if m[0] in placed_images:
                    sum += np.array(proj.image_list[m[0]].coord_list[m[1]])
                    # print 'sum:', sum
                    count += 1
            if count > 0:
                coord = sum / count
                match[0] = coord.tolist()
            
# iterate through all the matches and triangulate the 3d location for
# any feature that shows up in 3 or more placed images (uses the
# current 'sba' camera poses.)
import LineSolver
matches_group_counter = []
def my_triangulate(matches, placed_images, min_vectors=3):
    global matches_group_counter
    if len(matches_group_counter) != len(matches):
        matches_group_counter = [0] * len(matches)
        
    IK = np.linalg.inv( proj.cam.get_K() )
    for i, match in enumerate(matches):
        #print match
        points = []
        vectors = []
        for m in match[1:]:
            if m[0] in placed_images:
                image = proj.image_list[m[0]]
                (ned, ypr, quat) = image.get_camera_pose_sba()
                points.append(ned)
                vectors.append(image.vec_list[m[1]] )
                #print ' ', image.name
                #print ' ', uv_list
                #print '  ', vec_list
        if len(vectors) >= min_vectors and len(vectors) != matches_group_counter[i]:
            # only solve if we have new information and enough vectors
            p = LineSolver.ls_lines_intersection(points, vectors, transpose=True).tolist()
            #print p, p[0]
            match[0] = [ p[0][0], p[1][0], p[2][0] ]
            matches_group_counter[i] = len(vectors)

# null the 3d location of any features not referenced by a placed image
def null_unplaced_features(matches, placed_images):
    for match in matches:
        if match[0] != None:
            # only non-null match coordinates
            count = 0;
            for m in match[1:]:
                if m[0] in placed_images:
                    count += 1
            if count == 0:
                match[0] = None

def update_pose(matches, new_index):
    new_image = proj.image_list[new_index]
    
    # Build a list of existing 3d ned vs. 2d uv coordinates for the
    # new image so we can run solvepnp() and derive an initial pose
    # estimate relative to the already placed group.
    new_ned_list = []
    new_uv_list = []
    for i, match in enumerate(matches):
        # only proceed with 'located' features
        if match[0] != None:
            # check if this match refers to the new image
            for m in match[1:]:
                if m[0] == new_index:
                    new_ned_list.append(match[0])
                    new_uv_list.append(new_image.uv_list[m[1]])
                    break
    print "Number of solvepnp coordinates:", len(new_ned_list)

    # debug
    # f = open('ned.txt', 'wb')
    # for ned in new_ned_list:
    #     f.write("%.2f %.2f %.2f\n" % (ned[0], ned[1], ned[2]))

    # f = open('uv.txt', 'wb')
    # for uv in new_uv_list:
    #     f.write("%.1f %.1f\n" % (uv[0], uv[1]))

    # pose new image here:
    rvec, tvec = new_image.get_proj()
    #print 'new_ned_list', new_ned_list
    #print 'new_uv_list', new_uv_list
    (result, rvec, tvec) \
        = cv2.solvePnPRansac(np.float32(new_ned_list), np.float32(new_uv_list),
                             proj.cam.get_K(scale), None,
                             rvec, tvec, useExtrinsicGuess=True)
    Rned2cam, jac = cv2.Rodrigues(rvec)
    pos = -np.matrix(Rned2cam[:3,:3]).T * np.matrix(tvec)
    newned = pos.T[0].tolist()[0]

    # Our Rcam matrix (in our ned coordinate system) is body2cam * Rned,
    # so solvePnP returns this combination.  We can extract Rned by
    # premultiplying by cam2body aka inv(body2cam).
    cam2body = new_image.get_cam2body()
    Rned2body = cam2body.dot(Rned2cam)
    Rbody2ned = np.matrix(Rned2body).T
    (yaw, pitch, roll) = transformations.euler_from_matrix(Rbody2ned, 'rzyx')

    print "original pose:", new_image.get_camera_pose()
    #print "original pose:", proj.image_list[30].get_camera_pose()
    new_image.set_camera_pose_sba(ned=newned,
                                  ypr=[yaw*r2d, pitch*r2d, roll*r2d])
    new_image.save_meta()
    print "solvepnp() pose:", new_image.get_camera_pose_sba()

    
proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()
proj.load_match_pairs()

matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )
print "unique features (before grouping):", len(matches_direct)

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

print "Number of extended groupings:", group_count

# determine scale value so we can get correct K matrix
image_width = proj.image_list[0].width
camw, camh = proj.cam.get_image_params()
scale = float(image_width) / float(camw)
print 'scale:', scale

# pose new image here:
K = proj.cam.get_K(scale)
print "K:", K
IK = np.linalg.inv(K)

# initialize sba camera pose to direct pose
for image in proj.image_list:
    (ned, ypr, q) = image.get_camera_pose()
    image.set_camera_pose_sba(ned, ypr)
    image.save_meta()

# null out all the image.coord_lists and connection order
for image in proj.image_list:
    image.vec_list = None
    image.coord_list = None
    image.connection_order = -1
proj.save_images_meta()
    
# null all the match locations
for match in matches_group:
    match[0] = None

bootstrap = True
done = False
while not done:
    print "Start of top level placing algorithm..."
    
    # start with no placed images
    placed_images = set()

    # wipe vec_list and coord_list and connection order for all images
    for image in proj.image_list:
        image.vec_list = None
        image.coord_list = None
        image.connection_order = -1
    proj.save_images_meta()
        
    # find the image with the most connections to other images
    max_connections = 0
    max_index = -1
    for i, image in enumerate(proj.image_list):
        count = 0
        for m in image.match_list:
            if len(m):
                count += 1
        if count > max_connections:
            max_connections = count
            max_index = i
    max_image = proj.image_list[max_index]
    max_image.connection_order = 0
    max_image.save_meta()
    print "Image with max connections:", max_image.name
    print "Number of connected images:", max_connections
    placed_images.add(max_index)
    if not bootstrap:
        update_pose(matches_group, max_index)

    while True:
        # for each placed image compute feature vectors and point
        # projection for pose
        for index in placed_images:
            image = proj.image_list[index]
            if image.coord_list == None or image.vec_list == None:
                (ned, ypr, quat) = image.get_camera_pose_sba()
                image.vec_list = proj.projectVectorsImageSBA(IK, image)
                pt_list = proj.intersectVectorsWithGroundPlane(ned, args.ground,
                                                               image.vec_list)
                image.coord_list = pt_list

        # update the match coordinate based on the feature projections for
        # any features only found in a single image so far
        if len(placed_images) == 1:
            if bootstrap:
                # only on the first iterate of the first iteration,
                # place features on estimated ground plane
                update_match_coordinates(matches_group, placed_images)
                bootstrap = False
            else:
                null_unplaced_features(matches_group, placed_images)
                
        # triangulate the match coordinate using the feature projections
        # from the placed images.  This will only update features with 2
        # or more placed images
        my_triangulate(matches_group, placed_images, min_vectors=4)
        
        # find the unplaced image with the most connections into the placed set

        # per image counter
        image_counter = [0] * len(proj.image_list)

        # count up the placed feature references to unplaced images
        for i, match in enumerate(matches_group):
            # only proceed if this feature has been placed (i.e. it
            # connects to two or more placed images)
            if match[0] != None:
                for m in match[1:]:
                    if not m[0] in placed_images:
                        image_counter[m[0]] += 1
        print 'connected image count:', image_counter
        new_index = -1
        max_connections = -1
        for i in range(len(image_counter)):
            if image_counter[i] > max_connections:
                new_index = i
                max_connections = image_counter[i]
        if max_connections > 4:
            print "New image with max connections:", proj.image_list[new_index].name
            print "Number of connected features:", max_connections
            placed_images.add(new_index)
        else:
            break

        print "Total number of placed images so far:", len(placed_images)

        new_image = proj.image_list[new_index]
        new_image.connection_order = len(placed_images) - 1
        update_pose(matches_group, new_index)
        print 'Image placed:', new_image.name

    # final triangulation of all match coordinates
    # print 'Performing final complete group triangulation'
    my_triangulate(matches_group, placed_images, min_vectors=2)
    
    # write out the updated matches_group file as matches_sba
    print "Writing match_sba file ...", len(matches_group), 'features'
    pickle.dump(matches_group, open(args.project + "/matches_sba", "wb"))

    try:
        print 'All image positions updated...'
        input("press enter to continue:")
    except:
        pass
    done = True

print placed_images
sba = SBA.SBA(args.project)
sba.prepair_data( proj.image_list, placed_images, matches_group,
                  proj.cam.get_K(scale) )
cameras, features = sba.run_live()

if len(cameras) != len(proj.image_list):
    print "The solver barfed, sorry about that. :-("
    quit()

for i, image in enumerate(proj.image_list):
    orig = image.camera_pose
    new = cameras[i]
    if len(new) == 7:
        newq = np.array( new[0:4] )
        tvec = np.array( new[4:7] )
    elif len(new) == 12:
        newq = np.array( new[5:9] )
        tvec = np.array( new[9:12] )
    elif len(new) == 17:
        newq = np.array( new[10:14] )
        tvec = np.array( new[14:17] )
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

# compare original camera locations with sba camera locations and
# derive a transform matrix to 'best fit' the new camera locations
# over the original ... trusting the original group gps solution as
# our best absolute truth for positioning the system in world
# coordinates.
src_list = []
dst_list = []
for i, image in enumerate(proj.image_list):
    if i in placed_images:
        # only consider images that are in the placed set
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
    if not i in placed_images:
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

# update the point locations in original matches_direct
for i, f in enumerate(new_feats):
    matches_direct[index_partial[i]][0] = new_feats[i]

# create the matches_sba list (copy) and update the ned coordinate
matches_sba = list(matches_partial)
for i, match in enumerate(matches_sba):
    #print type(new_feats[i])
    matches_sba[i][0] = new_feats[i]

# write out the updated match_dict
print "Writing match_sba file ...", len(matches_sba), 'features'
pickle.dump(matches_sba, open(args.project + "/matches_sba", "wb"))

