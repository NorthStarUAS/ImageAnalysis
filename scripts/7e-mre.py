#!/usr/bin/python

# For all the feature matches and camera poses, estimate a mean
# reprojection error

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import cv2
import json
import math
import numpy as np

sys.path.append('../lib')
import ProjectMgr

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()

f = open(args.project + "/Matches.json", 'r')
matches_dict = json.load(f)
f.close()

# image mean reprojection error
def compute_image_mre(image, cam):
    image_sum = 0.0
    image_max = 0.0
    #scale = float(image.width) / float(camw)
    #K = cam.get_K(scale)
    K = cam.get_K()
    rvec, tvec = image.get_proj()
    R, jac = cv2.Rodrigues(rvec)
    PROJ = np.concatenate((R, tvec), axis=1)
    result_list = []
    for i, pt in enumerate(image.obj_pts):
        uvh = K.dot( PROJ.dot( np.hstack((pt, 1.0)) ).T )
        #print uvh
        uvh /= uvh[2]
        #print uvh
        #print "%s -> %s" % ( image.img_pts[i], [ np.squeeze(uvh[0,0]), np.squeeze(uvh[1,0]) ] )
        uv = np.array( [ np.squeeze(uvh[0,0]), np.squeeze(uvh[1,0]) ] )
        dist = np.linalg.norm(np.array(image.img_pts[i]) - uv)
        #print dist
        image_sum += dist
        if dist > image_max:
            image_max = dist
        result_list.append( (dist, i) )
    # sort by worst max error first
    result_list = sorted(result_list, key=lambda fields: fields[0],
                         reverse=True)

    # meta stats on error values
    image_avg = image_sum / len(image.obj_pts)
    stddev_sum = 0.0
    for line in result_list:
        error = line[0]
        stddev_sum += (image_avg-error)*(image_avg-error)
    stddev = math.sqrt(stddev_sum / len(image.obj_pts))
    print "   error avg = %.2f stddev = %.2f" % (image_avg, stddev)

    for line in result_list:
        if line[0] > image_avg + 3*stddev:
            print "feat %d err=%.2f" % (line[1], line[0])
    return image_avg, image_max

# group mean reprojection error
def compute_group_mre(image_list, cam):
    # start with a clean slate
    for image in image_list:
        image.img_pts = []
        image.obj_pts = []

    # iterate through the match dictionary and build a per image list of
    # obj_pts and img_pts
    for key in matches_dict:
        feature_dict = matches_dict[key]
        points = feature_dict['pts']
        ned = matches_dict[key]['ned']
        for p in points:
            image = image_list[ p[0] ]
            kp = image.kp_list[ p[1] ]
            image.img_pts.append( kp.pt )
            image.obj_pts.append( ned )

    camw, camh = cam.get_image_params()
    result_list = []
    for i, image in enumerate(image_list):
        print image.name
        if len(image.img_pts) < 4:
            continue
        image_avg, image_max = compute_image_mre(image, cam)
        result_list.append( (image_avg, image_max, i) )
    
    # sort by worst max error first
    result_list = sorted(result_list, key=lambda fields: fields[0],
                         reverse=True)
    sum = 0.0
    for line in result_list:
        sum += line[0]
        print "%s mre avg=%.2f max=%.2f" % (image_list[line[2]].name,
                                            line[0], line[1])
    print "Total mean reprojection error = %.4f" % (sum / len(result_list))
    
    # meta stats on error values
    error_avg = math.sqrt(dist2_sum / len(match))
    stddev_sum = 0.0
    for line in report_list:
        error = line[0]
        stddev_sum += (error_avg-error)*(error_avg-error)
    stddev = math.sqrt(stddev_sum / len(match))
    print "   error avg = %.2f stddev = %.2f" % (error_avg, stddev)

    print "%s mre = %.2f max = %.2f" % (image.name,
                                        image_sum / len(image.obj_pts),
                                        image_max)
        
compute_group_mre(proj.image_list, proj.cam)
