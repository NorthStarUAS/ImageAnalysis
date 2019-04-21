#!/usr/bin/python3

import argparse
import math
import numpy as np
import os.path
from progress.bar import Bar
import pickle

from props import getNode

from lib import Groups
from lib import ProjectMgr
from lib import match_culling as cull

r2d = 180.0 / math.pi

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--group', type=int, default=0, help='group index')
parser.add_argument('--min-angle', type=float, default=1.0, help='max feature angle')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()

# a value of 2 let's pairs exist which can be trouble ...
matcher_node = getNode('/config/matcher', True)
min_chain_len = matcher_node.getInt("min_chain_len")
print("Notice: min_chain_len is:", min_chain_len)

#source = 'matches_direct'
source = 'matches_grouped'
print("Loading matches:", source)
matches = pickle.load( open( os.path.join(proj.analysis_dir, source), "rb" ) )
print('Number of original features:', len(matches))

# load the group connections within the image set
groups = Groups.load(proj.analysis_dir)
print('Group sizes:', end=" ")
for group in groups:
    print(len(group), end=" ")
print()

def compute_angle(ned1, ned2, ned3):
    vec1 = np.array(ned3) - np.array(ned1)
    vec2 = np.array(ned3) - np.array(ned2)
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    denom = n1*n2
    if abs(denom - 0.000001) > 0:
        try:
            tmp = np.dot(vec1, vec2) / denom
            if tmp > 1.0: tmp = 1.0
            return math.acos(tmp)
        except:
            print('vec1:', vec1, 'vec2', vec2, 'dot:', np.dot(vec1, vec2))
            print('denom:', denom)
            return 0
    else:
        return 0

bar = Bar('Scanning match pair angles:', max=100)
step = int(len(matches) / 100)
#print("Scanning match pair angles...")
mark_list = []
for k, match in enumerate(matches):
    if match[1] == args.group:  # used by current group
        for i, m1 in enumerate(match[2:]):
            for j, m2 in enumerate(match[2:]):
                if i < j:
                    i1 = proj.image_list[m1[0]]
                    i2 = proj.image_list[m2[0]]
                    if i1.name in groups[args.group] and i2.name in groups[args.group]:
                        ned1, ypr1, q1 = i1.get_camera_pose(opt=True)
                        ned2, ypr2, q2 = i2.get_camera_pose(opt=True)
                        quick_approx = False
                        if quick_approx:
                            # quick hack angle approximation
                            avg = (np.array(ned1) + np.array(ned2)) * 0.5
                            y = np.linalg.norm(np.array(ned2) - np.array(ned1))
                            x = np.linalg.norm(avg - np.array(match[0]))
                            angle_deg = math.atan2(y, x) * r2d
                        else:
                            angle_deg = compute_angle(ned1, ned2, match[0]) * r2d
                        if angle_deg < args.min_angle:
                            mark_list.append( [k, i] )
    if (k+1) % step == 0:
        bar.next()
bar.finish()

# Pairs with very small average angles between each feature and camera
# location indicate closely located camera poses and these cause
# problems because very small changes in camera pose lead to very
# large changes in feature location.

# mark selection
cull.mark_using_list(mark_list, matches)
mark_sum = len(mark_list)

mark_sum = len(mark_list)
if mark_sum > 0:
    print('Outliers to remove from match lists:', mark_sum)
    result = input('Save these changes? (y/n):')
    if result == 'y' or result == 'Y':
        cull.delete_marked_features(matches, min_chain_len)
        # write out the updated match dictionaries
        print("Writing original matches:", source)
        pickle.dump(matches, open(os.path.join(proj.analysis_dir, source), "wb"))

