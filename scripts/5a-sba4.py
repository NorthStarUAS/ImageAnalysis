#!/usr/bin/python

# write out the data in a form useful to pass to the sba (demo) program

# it appears that SBA camera poses are basically given as [ R | t ]
# where R is the same R we use throughout and t is the 'tvec'

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
def my_triangulate(matches, placed_images, min_vectors=3):
    IK = np.linalg.inv( proj.cam.get_K() )
    for match in matches:
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
        if len(vectors) >= min_vectors:
            p = LineSolver.ls_lines_intersection(points, vectors, transpose=True).tolist()
            #print p, p[0]
            match[0] = [ p[0][0], p[1][0], p[2][0] ]

# returns result in radians
def quaternion_angle(a, b):
    a_inv = transformations.quaternion_inverse(a)
    res = transformations.quaternion_multiply(b, a_inv)
    res = res / np.linalg.norm(res)
    angle = math.acos(res[0]) * 2
    if angle < math.pi: angle += 2*math.pi
    if angle > math.pi: angle -= 2*math.pi
    return angle

# dump all the orientation quaternion components into a big
# vector/list for the optimizer.
def initialGuess():
    initial = []
    for i, image in enumerate(proj.image_list):
        (ned, ypr, quat) = image.get_camera_pose()
        initial.extend(quat.tolist())
    return initial        

# dump all the orientation quaternion components into a big
# vector/list for the optimizer.
def saveOrientation(xk):
    for i, image in enumerate(proj.image_list):
        (ned_orig, ypr_orig, quat_orig) = image.get_camera_pose_sba()
        quat = xk[i*4:i*4+4]
        # convert q to ypr
        ypr = transformations.euler_from_quaternion(quat, 'rzyx')
        # q = transformations.quaternion_from_euler(ypr[0], ypr[1], ypr[2], 'rzyx')
        ypr_deg = [ ypr[0]*r2d, ypr[1]*r2d, ypr[2]*r2d ]
        image.set_camera_pose_sba(ned=ned_orig, ypr=ypr_deg)
    proj.save_images_meta()

        

# For each matching pair of images we can compute an 'essential'
# matrix E.  Decomposing E gives us the relative rotation between the
# two camera poses as well as the relative direction of the two
# cameras.  This function compares the difference between the actual
# current pair angle offset with the 'ideal' offset and generates a
# metric based on that.
def errorFunc(xk):
    # extract quats and stash in a temporary name
    for i, i1 in enumerate(proj.image_list):
        q = xk[i*4:i*4+4]
        i1.opt_quat = q
        
    image_sum = 0
    image_count = 0
    for i, i1 in enumerate(proj.image_list):
        pair_sum = 0
        pair_count = 0
        for j, i2 in enumerate(proj.image_list):
            if i == j:
                continue
            #(ned1, ypr1, quat1_sba) = i1.get_camera_pose_sba()
            #(ned2, ypr2, quat2_sba) = i2.get_camera_pose_sba()
            quat1 = i1.opt_quat
            quat2 = i2.opt_quat
            # print quat1, quat2
            R = i1.R_list[j]
            if R is None:
                # no match
                continue
            #print 'R:', R
            #Rh = np.concatenate((R, np.zeros((3,1))), axis=1)
            #print 'Rh:', Rh
            #Rh = np.concatenate((Rh, np.zeros((1,4))), axis=0)
            #Rh[3,3] = 1
            #print Rh              
            #q = transformations.quaternion_from_matrix(R)
            q_inv = i1.q_inv_list[j]
            q1_maybe = transformations.quaternion_multiply(q_inv, quat2)
            #q2_maybe = transformations.quaternion_multiply(q, quat1)
            angle1 = quaternion_angle(quat1, q1_maybe)
            #angle2 = quaternion_angle(quat2, q2_maybe)
            # print i, j
            # print ' q1: ', quat1
            # print ' q2: ', quat2
            # print ' q:  ', q
            # print ' q1?:', q1_maybe
            # print ' q2?:', q2_maybe
            # print ' ang1:', quaternion_angle(quat1, q1_maybe)
            # print ' ang2:', quaternion_angle(quat2, q2_maybe)
            pair_sum += angle1 * angle1 * i1.weight_list[j]
            pair_count += i1.weight_list[j]
        if pair_count > 0:
            image_err = math.sqrt(pair_sum / pair_count)
            # print 'image error:', image_err
            image_sum += image_err * image_err
            image_count += 1
    if image_count > 0:
        total_err = math.sqrt(image_sum / image_count)
        # print 'total error:', total_err
    return total_err

def printStatus(xk):
    print 'Current value:', errorFunc(xk), 'saving as (sba)'
    saveOrientation(xk)
    # try:
    #     print 'All image positions updated...'
    #     input("press enter to continue:")
    # except:
    #     pass

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()
proj.load_match_pairs()

matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )
print "unique features (before grouping):", len(matches_direct)

# determine scale value so we can get correct K matrix
image_width = proj.image_list[0].width
image_height = proj.image_list[0].height
max_area = image_width * image_height
camw, camh = proj.cam.get_image_params()
scale = float(image_width) / float(camw)
print 'scale:', scale

# camera calibration
K = proj.cam.get_K(scale)
print "K:", K
IK = np.linalg.inv(K)

# compute the essential matrix and relative poses for all match pairs
#method = cv2.RANSAC
method = cv2.LMEDS
for i, i1 in enumerate(proj.image_list):
    i1.E_list = [None] * len(proj.image_list) # essential matrix for pair
    i1.R_list = [None] * len(proj.image_list) # relative rotation matrix
    i1.q_inv_list = [None] * len(proj.image_list) # precompute to save time
    i1.weight_list = [None] * len(proj.image_list) # precompute a weight metric
    i1.tvec_list = [None] * len(proj.image_list) # relative translation vector
    i1.d_list = [None] * len(proj.image_list) # gps distance between pair
    for j, i2 in enumerate(proj.image_list):
        matches = i1.match_list[j]
        if i == j or len(matches) < 5:
            # essential matrix needs at least 5 matches
            continue
        uv1 = []
        uv2 = []
        for k, pair in enumerate(matches):
            uv1.append( i1.uv_list[pair[0]] )
            uv2.append( i2.uv_list[pair[1]] )
        uv1 = np.float32(uv1)
        uv2 = np.float32(uv2)
        E, mask = cv2.findEssentialMat(points1=uv1, points2=uv2,
                                       cameraMatrix=K,
                                       method=method)
        print i1.name, 'vs', i2.name
        print E
        print
        (n, R, tvec, mask) = cv2.recoverPose(E=E,
                                          points1=uv1, points2=uv2,
                                          cameraMatrix=K)
        print '  inliers:', n, 'of', len(uv1)
        print '  R:', R
        print '  tvec:', tvec

        # convert R to homogeonous
        #Rh = np.concatenate((R, np.zeros((3,1))), axis=1)
        #Rh = np.concatenate((Rh, np.zeros((1,4))), axis=0)
        #Rh[3,3] = 1
        # extract the equivalent quaternion, and invert
        q = transformations.quaternion_from_matrix(R)
        q_inv = transformations.quaternion_inverse(q)

        # generate a rough estimate of uv area of covered by common
        # features (used as a weighting factor)
        minu = maxu = uv1[0][0]
        minv = maxv = uv1[0][1]
        for uv in uv1:
            #print uv
            if uv[0] < minu: minu = uv[0]
            if uv[0] > maxu: maxu = uv[0]
            if uv[1] < minv: minv = uv[1]
            if uv[1] > maxv: maxv = uv[1]
        area = (maxu-minu)*(maxv-minv)
        # print 'u:', minu, maxu
        # print 'v:', minv, maxv
        # print 'area:', area

        # compute a weight metric, credit more matches between a pair,
        # and credict a bigger coverage area
        weight = area / max_area
        #weight = (area / max_area) * len(uv1)
        
        (ned1, ypr1, quat1) = i1.get_camera_pose()
        (ned2, ypr2, quat2) = i2.get_camera_pose()
        diff = np.array(ned2) - np.array(ned1)
        dist = np.linalg.norm( diff )
        dir = diff / dist
        print 'dist:', dist, 'ned dir:', dir[0], dir[1], dir[2]

        Rbody2ned = i1.get_body2ned()
        cam2body = i1.get_cam2body()
        body2cam = i1.get_body2cam()
        est_dir = Rbody2ned.dot(cam2body).dot(R).dot(tvec)
        est_dir = est_dir / np.linalg.norm(est_dir) # normalize
        print 'est dir:', est_dir.tolist()
        
        i1.E_list[j] = E
        i1.R_list[j] = R
        i1.q_inv_list[j] = q_inv
        i1.tvec_list[j] = tvec
        i1.d_list[j] = dist
        i1.weight_list[j] = weight*weight

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

# initialize sba camera pose to direct pose
for image in proj.image_list:
    (ned, ypr, q) = image.get_camera_pose()
    image.set_camera_pose_sba(ned, ypr)

# null out all the image.coord_lists and connection order
for image in proj.image_list:
    image.coord_list = None
    image.connection_order = -1
proj.save_images_meta()
    
# null all the match locations
for match in matches_group:
    match[0] = None

from scipy.optimize import minimize
initial = initialGuess()
print 'Optimizing %d values.' % (len(initial))
print 'Starting value:', errorFunc(initial)
res = minimize(errorFunc,
               initial,
               options={'disp': True},
               callback=printStatus)
print res

saveOrientation(res['x'])
quit()

while True:
    print "Start of top level placing algorithm..."
    
    # start with no placed images
    placed_images = set()

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

        # triangulate the match coordinate using the feature projections
        # from the placed images.  This will only update features with 2
        # or more placed images
        my_triangulate(matches_group, placed_images, min_vectors=2)
        
        # update the match coordinate based on the feature projections for
        # any features only found in a single image so far
        if len(placed_images) == 1:
            # only at the very start to boot strap the process
            update_match_coordinates(matches_group, placed_images)

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

        # Build a list of existing 3d ned vs. 2d uv coordinates for the
        # new image so we can run solvepnp() and derive an initial pose
        # estimate relative to the already placed group.
        new_ned_list = []
        new_uv_list = []
        for i, match in enumerate(matches_group):
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
            = cv2.solvePnP(np.float32(new_ned_list), np.float32(new_uv_list),
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

        print 'Image placed:', new_image.name

    # final triangulation of all match coordinates
    # print 'Performing final complete group triangulation'
    # my_triangulate(matches_group, placed_images, min_vectors=2)
    
    # write out the updated matches_group file as matches_sba
    print "Writing match_sba file ...", len(matches_group), 'features'
    pickle.dump(matches_group, open(args.project + "/matches_sba", "wb"))

    errorFunc()

    try:
        print 'All image positions updated...'
        input("press enter to continue:")
    except:
        pass


sba = SBA.SBA(args.project)
sba.prepair_data( proj.image_list, matches_group, proj.cam.get_K(scale) )
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

