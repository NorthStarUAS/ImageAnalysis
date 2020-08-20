#!/usr/bin/env python3

import argparse
import math
import numpy as np
import os.path
import pickle
import sys

sys.path.append('../lib')
import groups
import project

import match_culling as cull

r2d = 180.0 / math.pi

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('project', help='project directory')
args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()

#source = 'matches_direct'
source = 'matches_grouped'
print("Loading matches:", source)
matches_orig = pickle.load( open( os.path.join(args.project, source), "rb" ) )
print('Number of original features:', len(matches_orig))
print("Loading optimized matches: matches_opt")
matches_opt = pickle.load( open( os.path.join(args.project, "matches_opt"), "rb" ) )
print('Number of optimized features:', len(matches_opt))

# load the group connections within the image set
group_list = groups.load(args.project)
print('Main group size:', len(group_list[0]))

pair_angles = []
for i in range(len(proj.image_list)):
    p = [ [] for j in range(len(proj.image_list)) ]
    pair_angles.append(p)

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
    
print("Computing match pair angles...")
for match in matches_opt:
    for m1 in match[1:]:
        for m2 in match[1:]:
            if m1[0] >= m2[0]:
                continue
            if m1[0] in group_list[0] and m2[0] in group_list[0]:
                i1 = proj.image_list[m1[0]]
                i2 = proj.image_list[m2[0]]
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
                pair_angles[m1[0]][m2[0]].append(angle_deg)

print("Computing per-image statistics...")
by_pair = []
for i in range(len(proj.image_list)):
    for j in range(len(proj.image_list)):
        if len(pair_angles[i][j]):
            angles = np.array(pair_angles[i][j])
            avg = np.mean(angles)
            std = np.std(angles)
            min = np.amin(angles)
            #print(i, j, 'avg:', avg, 'std:', std, 'min:', min)
            by_pair.append( [i, j, avg, std, min] )

# (Average angle) pairs with very small average angles between each feature
# and camera location indicate closely located camera poses and these
# cause problems because very small changes in camera pose lead to
# very large changes in feature location.

# (Standard Deviaion) I haven't wrapped my head around this but pairs
# with a high standard deviation of feature to camera pose angles also
# seem to be tied to optimizer/solution degeneracy.

# (Minimum angle) Pairs with a few small minimum angles in the set suggest
# one image is above the other and again run into the problem where
# small pose changes can lead to large feature location changes (but
# just for a few features, not all the features in the pairing.)

avg_cutoff_deg = 2
min_cutoff_deg = 0.5
std_cutoff_deg = 10
print("Marking small angle image pairs for deletion...")
mark_list = []
by_pair = sorted(by_pair, key=lambda fields: fields[4], reverse=False) # by min
for line in by_pair:
    print(line[0], line[1], 'avg: %.2f' % line[2], 'std: %.2f' % line[3], 'min: %.2f' % line[4])
    if line[2] < avg_cutoff_deg or line[3] > std_cutoff_deg or line[4] < min_cutoff_deg:   # cutoff angles (deg)
        print('  (remove)')
        for k, match in enumerate(matches_opt):
            size = len(match[1:])
            for i, m1 in enumerate(match[1:]):
                for j, m2 in enumerate(match[1:]):
                    # this will probably create double markings, but
                    # that's ok, I want to be sure.
                    if m1[0] == line[0] and m2[0] == line[1]:
                        mark_list.append( [k, i] )
                        mark_list.append( [k, j] )
                    if m1[0] == line[1] and m2[0] == line[0]:
                        mark_list.append( [k, i] )
                        mark_list.append( [k, j] )
result = input('Press enter to continue:')

# mark selection
cull.mark_using_list(mark_list, matches_orig)
cull.mark_using_list(mark_list, matches_opt)
mark_sum = len(mark_list)

def delete_marked_features(matches):
    print(" deleting marked items...")
    for i in reversed(range(len(matches))):
        match = matches[i]
        has_bad_elem = False
        for j in reversed(range(1, len(match))):
            p = match[j]
            if p == [-1, -1]:
                has_bad_elem = True
                match.pop(j)
        if len(match) < 3:
            print("deleting match that is now in less than 2 images:", match)
            matches.pop(i)

mark_sum = len(mark_list)
if mark_sum > 0:
    print('Outliers to remove from match lists:', mark_sum)
    result = input('Save these changes? (y/n):')
    if result == 'y' or result == 'Y':
        delete_marked_features(matches_orig)
        delete_marked_features(matches_opt)
        # write out the updated match dictionaries
        print("Writing original matches...")
        pickle.dump(matches_orig, open(os.path.join(args.project, source), "wb"))
        print("Writing optimized matches...")
        pickle.dump(matches_opt, open(os.path.join(args.project, "matches_opt"), "wb"))

