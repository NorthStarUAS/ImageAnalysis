#!/usr/bin/python3

# determine the connected groups of images.  Images without
# connections to each other cannot be correctly placed.

import argparse
import pickle
import os.path

from lib import Groups
from lib import ProjectMgr

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--area', default='area-00', help='sub area directory')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_area_info(args.area)

area_dir = os.path.join(args.project, args.area)
source = 'matches_grouped'
print("Loading source matches:", source)
matches = pickle.load( open( os.path.join(area_dir, source), 'rb' ) )

print("features:", len(matches))

# compute the group connections within the image set.

groups = Groups.groupByFeatureConnections(proj.image_list, matches)
# groups = Groups.groupByConnectedArea(proj.image_list, matches)
# groups = Groups.groupByImageConnections(proj)

groups.sort(key=len, reverse=True)
Groups.save(area_dir, groups)

print('Total images:', len(proj.image_list))
print('Group sizes:', end=" ")
for g in groups:
    print(len(g), end=" ")
print()

# debug
print("Counting allocated features...")
count = 0
for i, match in enumerate(matches):
    if match[1] >= 0:
        count += 1

print("Writing:", source, "...")
print("Features: %d/%d" % (count, len(matches)))
pickle.dump(matches, open(os.path.join(area_dir, source), "wb"))

# this is extra (and I'll put it here for now for lack of a better
# place), but for visualization's sake, create a gnuplot data file
# that will show all the match connectivity in the set.
# file = os.path.join(area_dir, 'connections.gnuplot')
# f = open(file, 'w')
# pair_dict = {}
# for match in matches:
#     for m1 in match[1:]:
#         for m2 in match[1:]:
#             if m1[0] == m2[0]:
#                 # skip selfies
#                 continue
#             key = '%d %d' % (m1[0], m2[0])
#             pair_dict[key] = [m1[0], m2[0]]
# for pair in pair_dict:
#     #print 'pair:', pair, pair_dict[pair]
#     image1 = proj.image_list[pair_dict[pair][0]]
#     image2 = proj.image_list[pair_dict[pair][1]]
#     (ned1, ypr1, quat1) = image1.get_camera_pose()
#     (ned2, ypr2, quat2) = image2.get_camera_pose()
#     f.write("%.2f %.2f\n" % (ned1[1], ned1[0]))
#     f.write("%.2f %.2f\n" % (ned2[1], ned2[0]))
#     f.write("\n")
# f.close()       
    
    
