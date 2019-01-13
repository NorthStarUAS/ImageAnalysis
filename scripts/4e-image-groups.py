#!/usr/bin/python3

# determine the connected groups of images.  Images without
# connections to each other cannot be correctly placed.

import argparse
import pickle
import os.path
import sys

sys.path.append('../lib')
import Groups
import ProjectMgr

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--area', required=True, help='sub area directory')
parser.add_argument('--original-pairs', action='store_true', help='use original pair-rwise matches')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_area_info(args.area)

area_dir = os.path.join(args.project, args.area)
source = 'matches_grouped'
print("Loading source matches:", source)
matches = pickle.load( open( os.path.join(area_dir, source), 'rb' ) )

print("features:", len(matches))

if not args.original_pairs:
    # recreate the pair-wise match structure (so we can honor any
    # features/pairs removed by outlier rejection strategies
    for i1 in proj.image_list:
        i1.match_list = {}
    for match in matches:
        for p1 in match[1:]:
            image1 = proj.image_list[p1[0]]
            for p2 in match[1:]:
                if p1 == p2:
                    continue
                image2 = proj.image_list[p2[0]]
                if image2.name in image1.match_list:
                    image1.match_list[image2.name].append( [p1[1], p2[1]] )
                else:
                    image1.match_list[image2.name] = [ [p1[1], p2[1]] ]
else:
    proj.load_match_pairs(extra_verbose=False)

# compute the group connections within the image set.

#groups = Groups.groupByFeatureConnections(proj.image_list, matches)
#groups = Groups.groupByConnectedArea(proj.image_list, matches)
groups = Groups.groupByImageConnections(proj)

groups.sort(key=len, reverse=True)

Groups.save(area_dir, groups)

print('Group sizes:', end=" ")
for g in groups:
    print(len(g), end=" ")
print()

# this is extra (and I'll put it here for now for lack of a better
# place), but for visualization's sake, create a gnuplot data file
# that will show all the match connectivity in the set.
file = os.path.join(area_dir, 'connections.gnuplot')
f = open(file, 'w')
pair_dict = {}
for match in matches:
    for m1 in match[1:]:
        for m2 in match[1:]:
            if m1[0] == m2[0]:
                # skip selfies
                continue
            key = '%d %d' % (m1[0], m2[0])
            pair_dict[key] = [m1[0], m2[0]]
for pair in pair_dict:
    #print 'pair:', pair, pair_dict[pair]
    image1 = proj.image_list[pair_dict[pair][0]]
    image2 = proj.image_list[pair_dict[pair][1]]
    (ned1, ypr1, quat1) = image1.get_camera_pose()
    (ned2, ypr2, quat2) = image2.get_camera_pose()
    f.write("%.2f %.2f\n" % (ned1[1], ned1[0]))
    f.write("%.2f %.2f\n" % (ned2[1], ned2[0]))
    f.write("\n")
f.close()       
    
    
