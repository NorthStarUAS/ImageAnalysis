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
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()
proj.load_features()
proj.undistort_keypoints()

# no! (maybe?)
print("Loading direct matches...")
matches = pickle.load( open( os.path.join(args.project, 'matches_direct'), 'rb' ) )

#print("Loading grouped matches...")
#matches = pickle.load( open( os.path.join(args.project, 'matches_grouped'), 'rb' ) )
print("features:", len(matches))

# compute the group connections within the image set (not used
# currently in the bundle adjustment process, but here's how it's
# done...)

groups = Groups.groupByFeatureConnections(proj.image_list, matches)

#groups = Groups.groupByConnectedArea(proj.image_list, matches)

#proj.load_match_pairs(extra_verbose=False)
#groups = Groups.groupByImageConnections(proj.image_list)

Groups.save(args.project, groups)

print('Main group size:', len(groups[0]))

# this is extra (and I'll put it here for now for lack of a better
# place), but for visualization's sake, create a gnuplot data file
# that will show all the match connectivity in the set.
file = os.path.join(args.project, 'connections.gnuplot')
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
    
    
