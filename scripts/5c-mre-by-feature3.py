#!/usr/bin/python

# For all the feature matches and camera poses, estimate a mean
# reprojection error

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

import argparse
import cPickle as pickle
import cv2
#import json
import math
import numpy as np

sys.path.append('../lib')
import ProjectMgr

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--stddev', type=float, default=3, help='how many stddevs above the mean for auto discarding features')
parser.add_argument('--direct', action='store_true', help='analyze direct matches (might help if initial sba fit fails.)')
parser.add_argument('--group', action='store_true', help='analyze max grouped matches.')
parser.add_argument('--strong', action='store_true', help='remove entire match chain, not just the worst offending element.')
parser.add_argument('--show', action='store_true', help='show most extreme reprojection errors with matches.')

args = parser.parse_args()

if args.direct:
    print "NOTICE: analyzing direct matches list"
    
proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.undistort_keypoints()

print "Loading original (direct) matches ..."
matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )

print "Loading maximally grouped matches ..."
matches_group = pickle.load( open( args.project + "/matches_group", "rb" ) )

print "Loading fitted (sba) matches..."
matches_sba = pickle.load( open( args.project + "/matches_sba", "rb" ) )

# image mean reprojection error
def compute_feature_mre(K, image, kp, ned):
    if image.PROJ == None:
        if args.direct:
            rvec, tvec = image.get_proj() # original direct pose
        else:
            rvec, tvec = image.get_proj_sba() # fitted pose
        R, jac = cv2.Rodrigues(rvec)
        image.PROJ = np.concatenate((R, tvec), axis=1)

    PROJ = image.PROJ
    uvh = K.dot( PROJ.dot( np.hstack((ned, 1.0)) ).T )
    #print uvh
    uvh /= uvh[2]
    #print uvh
    #print "%s -> %s" % ( image.img_pts[i], [ np.squeeze(uvh[0,0]), np.squeeze(uvh[1,0]) ] )
    uv = np.array( [ np.squeeze(uvh[0,0]), np.squeeze(uvh[1,0]) ] )
    dist = np.linalg.norm(np.array(kp) - uv)
    return dist

# group reprojection error for every used feature
def compute_reprojection_errors(image_list, cam):
    print "Computing reprojection error for all match points..."
    
    # start with a clean slate
    for image in image_list:
        image.PROJ = None

    camw, camh = proj.cam.get_image_params()

    # iterate through the match dictionary and build a per image list of
    # obj_pts and img_pts
    result_list = []

    if args.direct:
        print "Using matches_direct"
        matches_source = matches_direct
    elif args.group:
        print "Using matches_group"
        matches_source = matches_group
    else:
        print "Using matches_sba"
        matches_source = matches_sba
        
    for i, match in enumerate(matches_source):
        ned = match[0]
        for j, p in enumerate(match[1:]):
            image = image_list[ p[0] ]
            # kp = image.kp_list[p[1]].pt # distorted
            kp = image.uv_list[ p[1] ]  # undistorted uv point
            scale = float(image.width) / float(camw)
            dist = compute_feature_mre(cam.get_K(scale), image, kp, ned)
            if i == 67364 or i == 67469:
                print i, 'dist:', dist, 'ned:', ned
            result_list.append( (dist, i, j) )

    # sort by worst max error first
    result_list = sorted(result_list, key=lambda fields: fields[0],
                         reverse=True)
    return result_list

def show_outliers(result_list, trim_stddev):
    print "Show outliers..."
    sum = 0.0
    count = len(result_list)

    # numerically it is better to sum up a list of floating point
    # numbers from smallest to biggest (result_list is sorted from
    # biggest to smallest)
    for line in reversed(result_list):
        sum += line[0]
        
    # stats on error values
    print " computing stats..."
    mre = sum / count
    stddev_sum = 0.0
    for line in result_list:
        error = line[0]
        stddev_sum += (mre-error)*(mre-error)
    stddev = math.sqrt(stddev_sum / count)
    print "mre = %.4f stddev = %.4f" % (mre, stddev)

    for line in result_list:
        # print "line:", line
        if line[0] > mre + stddev * trim_stddev:
            print "  outlier index %d-%d err=%.2f" % (line[1], line[2],
                                                      line[0])
            draw_match(line[1], line[2])
            
def mark_outliers(result_list, trim_stddev):
    print "Marking outliers..."
    sum = 0.0
    count = len(result_list)

    # numerically it is better to sum up a list of floatting point
    # numbers from smallest to biggest (result_list is sorted from
    # biggest to smallest)
    for line in reversed(result_list):
        sum += line[0]
        
    # stats on error values
    print " computing stats..."
    mre = sum / count
    stddev_sum = 0.0
    for line in result_list:
        error = line[0]
        stddev_sum += (mre-error)*(mre-error)
    stddev = math.sqrt(stddev_sum / count)
    print "mre = %.4f stddev = %.4f" % (mre, stddev)

    # mark match items to delete
    print " marking outliers..."
    mark_count = 0
    for line in result_list:
        # print "line:", line
        if line[0] > mre + stddev * trim_stddev:
            print "  outlier index %d-%d err=%.2f" % (line[1], line[2],
                                                      line[0])
            match = matches_direct[line[1]]
            match[line[2]+1] = [-1, -1]
            if not args.direct:
                match = matches_sba[line[1]]
                match[line[2]+1] = [-1, -1]
            mark_count += 1
            
    # trim the result_list
    print " trimming results list..."
    for i in range(len(result_list)):
        line = result_list[i]
        if line[0] < mre + stddev * trim_stddev:
            if i > 0:
                # remove the marked items from the sorted list
                del result_list[0:i]
            # and break
            break
            
    return result_list, mark_count

# delete marked matches
def delete_marked_matches():
    print " deleting marked items..."
    for i in reversed(range(len(matches_direct))):
        match_direct = matches_direct[i]
        if not args.direct:
            match_sba = matches_sba[i]
        has_bad_elem = False
        for j in reversed(range(1, len(match_direct))):
            p = match_direct[j]
            if p == [-1, -1]:
                has_bad_elem = True
                match_direct.pop(j)
                if not args.direct:
                    match_sba.pop(j)
        if args.strong and has_bad_elem:
            print "deleting entire match that contains a bad element"
            matches_direct.pop(i)
            if not args.direct:
                matches_sba.pop(i)
        elif len(match_direct) < 3:
            print "deleting match that is now in less than 2 images:", match_direct
            matches_direct.pop(i)
            if not args.direct:
                matches_sba.pop(i)
        elif False and len(match_direct) < 4:
            # this is seeming like less and less of a good idea (Jan 3, 2017)
            print "deleting match that is now in less than 3 images:", match_direct
            matches_direct.pop(i)
            if not args.direct:
                matches_sba.pop(i)

# experimental, draw a visual of a match point in all it's images
def draw_match(i, index):
    green = (0, 255, 0)
    red = (0, 0, 255)

    if not args.direct:
        match = matches_sba[i]
    else:
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

result_list = compute_reprojection_errors(proj.image_list, proj.cam)

if args.show:
    show_outliers(result_list, args.stddev)
    
result_list, mark_sum = mark_outliers(result_list, args.stddev)

# now count how many features show up in each image
for i in proj.image_list:
    i.feature_count = 0
for i, match in enumerate(matches_direct):
    for j, p in enumerate(match[1:]):
        if p[1] != [-1, -1]:
            image = proj.image_list[ p[0] ]
            image.feature_count += 1

# make a dict of all images with less than 25 feature matches
weak_dict = {}
for i, img in enumerate(proj.image_list):
    print img.name, img.feature_count
    if img.feature_count > 0 and img.feature_count < 25:
        weak_dict[i] = True
print 'weak images:', weak_dict

# mark any features in the weak images list
for i, match in enumerate(matches_direct):
    #print 'before:', match
    for j, p in enumerate(match[1:]):
        if p[0] in weak_dict:
             match[j+1] = [-1, -1]
             mark_sum += 1
    #print 'after:', match

if mark_sum > 0:
    result=raw_input('Remove ' + str(mark_sum) + ' outliers from the original matches? (y/n):')
    if result == 'y' or result == 'Y':
        delete_marked_matches()
        # write out the updated match dictionaries
        print "Writing direct matches..."
        pickle.dump(matches_direct, open(args.project+"/matches_direct", "wb"))

        if not args.direct:
            print "Writing sba matches..."
            pickle.dump(matches_sba, open(args.project + "/matches_sba", "wb"))

#print "Mean reprojection error = %.4f" % (mre)

