#!/usr/bin/python3

import sys
#sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import math

import argparse
import cv2
import os.path

sys.path.append('../lib')
import ProjectMgr

# this is a one-off script for debugging.  The intention is to strip
# down a larger data set to something small for testing/debugging.  It
# computes the camera distance from the reference point and can delete
# anything further than some threshold.  Essentially leaving just a
# smaller number of images clustered around the center of the data
# set.  This isn't generally useful for anything other than debugging
# and it's not very general for that purpose either.

parser = argparse.ArgumentParser(description='Load the project\'s images.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--delete-further-than', type=float, help='delete images furhter than this distance from center')
args = parser.parse_args()
# print args

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()

def robust_delete(path):
    try:
        os.remove(path)
    except:
        print('cannot remove:', path)

dist_list = []
for image in proj.image_list:
    print(image.name, image.camera_pose)
    if not image.camera_pose == None:
        ned = image.camera_pose['ned']
        dist = math.sqrt(ned[0]*ned[0] + ned[1]*ned[1])
    else:
        dist = 99999999
    dist_list.append( [ dist, image.name ] )

by_dist = sorted( dist_list, key=lambda fields: fields[0])
if not args.delete_further_than:
    # just print the list and exit
    for line in by_dist:
        print(line)
else:
    count_save = 0
    count_delete = 0
    for line in by_dist:
        if line[0] < args.delete_further_than:
            count_save += 1
        else:
            count_delete += 1
    print('save:', count_save, 'delete:', count_delete)
    result=input('Permanently delete ' + str(count_delete) + ' images from project: ' + args.project + '? (y/n):')
    if result == 'y' or result == 'Y':
        print('ok, will delete them')
    else:
        print('quitting without deleting any images')
        quit()

    for line in by_dist:
        if line[0] >= args.delete_further_than:
            dist = line[0]
            name = line[1]
            base, ext = os.path.splitext(name)
            print(base, ext)
            path = os.path.join(args.project, 'Images')
            robust_delete(os.path.join(path, base + '.desc.npy'))
            robust_delete(os.path.join(path, base + '.feat'))
            robust_delete(os.path.join(path, base + '.info'))
            robust_delete(os.path.join(path, base + '.match'))
            robust_delete(os.path.join(path, name))
   
