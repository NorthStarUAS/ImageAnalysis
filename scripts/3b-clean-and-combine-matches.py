#!/usr/bin/env python3

import argparse
import pickle
import os.path

from lib import matcher
from lib import match_cleanup
from lib import project

# Reset all match point locations to their original direct
# georeferenced locations based on estimated camera pose and
# projection onto DEM earth surface

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('project', help='project directory')
args = parser.parse_args()

m = matcher.Matcher()

proj = project.ProjectMgr(args.project)
proj.load_images_info()
proj.load_features(descriptors=False)
#proj.undistort_keypoints()
proj.load_match_pairs()

match_cleanup.merge_duplicates(proj)

# enable the following code to visualize the matches after collapsing
# identical uv coordinates
if False:
    for i, i1 in enumerate(proj.image_list):
        for j, i2 in enumerate(proj.image_list):
            if i >= j:
                # don't repeat reciprocal matches
                continue
            if len(i1.match_list[j]):
                print("Showing %s vs %s" % (i1.name, i2.name))
                status = m.showMatchOrient(i1, i2, i1.match_list[j])

match_cleanup.check_for_pair_dups(proj)

# enable the following code to visualize the matches after eliminating
# duplicates (duplicates can happen after collapsing uv coordinates.)
if False:
    for i, i1 in enumerate(proj.image_list):
        for j, i2 in enumerate(proj.image_list):
            if i >= j:
                # don't repeat reciprocal matches
                continue
            if len(i1.match_list[j]):
                print("Showing %s vs %s" % (i1.name, i2.name))
                status = m.showMatchOrient(i1, i2, i1.match_list[j])

match_cleanup.check_for_1vn_dups(proj)

matches_direct = match_cleanup.make_match_structure(proj)

# Note to self: I don't think we need the matches_direct file any more
# (except for debugging possibly in the future.)
#
#print("Writing matches_direct file ...")
#direct_file = os.path.join(proj.analysis_dir, "matches_direct")
#pickle.dump(matches_direct, open(direct_file, "wb"))

matches_grouped = match_cleanup.link_matches(proj, matches_direct)
print("Writing full group chain matches_grouped file ...")
pickle.dump(matches_grouped, open(os.path.join(proj.analysis_dir, "matches_grouped"), "wb"))
