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

def diff_stats(pts1, pts2):
    if len(pts1) != len(pts2):
        print "Error: point lists not the same dimension"
        return
    
    sum = 0.0
    count = len(pts1)
    for i in range(count):
        p1 = pts1[i]
        p2 = pts2[i]
        dist = np.linalg.norm(p2 - p1)
        sum += dist
    average = sum / count
    print "mean error = %.2f" % (average)

    sum = 0.0
    for i in range(count):
        p1 = pts1[i]
        p2 = pts2[i]
        dist = np.linalg.norm(p2 - p1)
        diff = average - dist
        sum += diff**2
    stddev = math.sqrt(sum / count)

    print "standard deviation = %.2f" % (stddev)
    return average, stddev


parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--stddev', required=True, type=int, default=6, help='how many stddevs above the mean to consider')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()

print "Loading original matches ..."
f = open(args.project + "/Matches.json", 'r')
matches_direct = json.load(f)
f.close()

print "Loading sba matches ..."
f = open(args.project + "/Matches-sba.json", 'r')
matches_sba = json.load(f)
f.close()

print "Evaluating camera pose updates..."
pts1 = []
pts2 = []
for image in proj.image_list:
    ned, ypr, quat = image.get_camera_pose()
    pts1.append(np.array(ned))
    ned, ypr, quat = image.get_camera_pose_sba()
    pts2.append(np.array(ned))
average, stddev = diff_stats(pts1, pts2)

for image in proj.image_list:
    ned_direct, ypr, quat = image.get_camera_pose()
    ned_sba, ypr, quat = image.get_camera_pose_sba()
    dist = np.linalg.norm(np.array(ned_direct) - np.array(ned_sba))
    if abs(dist) > average + 3*stddev:
        print "Possible outlier = %s, dist = %.2f" % (image.name, dist)
    
print "original matches = %d, sba matches = %d" % (len(matches_direct),
                                                   len(matches_sba))

pts1 = []
pts2 = []
for key in matches_direct:
    ned_direct = np.array(matches_direct[key]['ned'])
    if key in matches_sba:
        ned_sba = np.array(matches_sba[key]['ned'])
        pts1.append(ned_direct)
        pts2.append(ned_sba)
    else:
        print "Not in sba =", key
average, stddev = diff_stats(pts1, pts2)

delete_list = []
for key in matches_direct:
    ned_direct = np.array(matches_direct[key]['ned'])
    if key in matches_sba:
        ned_sba = np.array(matches_sba[key]['ned'])
        dist = np.linalg.norm(ned_direct - ned_sba)
        if abs(dist) > average + stddev * args.stddev:
            print "Possible outlier = %s, dist = %.2f" % (key, dist),
            for i in matches_direct[key]['pts']:
                print "%s " % proj.image_list[i[0]].name,
            print
            delete_list.append(key)

result = raw_input('Remove these outliers from the original matches? (y/n):')
if result == 'y' or result == 'Y':
    for key in delete_list:
        ned_direct = np.array(matches_direct[key]['ned'])
        ned_sba = np.array(matches_sba[key]['ned'])
        dist = np.linalg.norm(ned_direct - ned_sba)
        if abs(dist) > average + stddev * args.stddev:
            print "Removing outlier = %s, dist = %.2f" % (key, dist),
            for i in matches_direct[key]['pts']:
                print "%s " % proj.image_list[i[0]].name,
            print
            del matches_direct[key]
            del matches_sba[key]

    # write out the updated match dictionaries
    print "Writing original matches..."
    f = open(args.project + "/Matches.json", 'w')
    json.dump(matches_direct, f, sort_keys=True)
    f.close()
    print "Writing sba matches..."
    f = open(args.project + "/Matches-sba.json", 'w')
    json.dump(matches_sba, f, sort_keys=True)
    f.close()

