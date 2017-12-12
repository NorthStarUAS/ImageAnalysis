#!/usr/bin/python

# determine the connected groups of images.  Images without
# connections to each other cannot be correctly placed.

import argparse
import cPickle as pickle
import os
import sys

sys.path.append('../lib')
import Groups
import ProjectMgr

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()

print "Loading direct matches..."
matches_direct = pickle.load( open( os.path.join(args.project, 'matches_direct'), 'rb' ) )
print "features:", len(matches_direct)

# compute the group connections within the image set (not used
# currently in the bundle adjustment process, but here's how it's
# done...)
groups = Groups.simpleGrouping(proj.image_list, matches_direct)
Groups.save(args.project, groups)
