#!/usr/bin/python3

import argparse
import pickle
import numpy as np
import os.path
from progress.bar import Bar
import sys

sys.path.append('../lib')
import ProjectMgr

# import match_culling as cull

# Maximally group all match chains.  If we squeeze out redundancy, the
# sba solution should be better.

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--area', required=True, help='sub area directory')

args = parser.parse_args()

#m = Matcher.Matcher()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_area_info(args.area)
proj.load_features(descriptors=False)

print("Loading match points (direct)...")
area_dir = os.path.join(args.project, args.area)
matches_direct = pickle.load( open( os.path.join(area_dir, "matches_direct"), "rb" ) )

# collect/group match chains that refer to the same keypoint

# warning 1: bad matches will lead to errors in the optimized fit.

# warning 2: if there are bad matches, this can lead to linking chains
# together that shouldn't be linked. This is hard to undo (I don't
# have tools to do this written.)  The alternative is nuking the
# entire match.

# notes: with only pairwise matching, the optimizer should have a good
# opportunity to generate a nice fit.  However, subsections within the
# solution could wander away from each other leading to some weird
# projection results at the intersection of these divergent groups.
# Ultimately we depend on a network of 3+ way feature match chains
# to link all the images and features together.

count = 0
done = False
while not done:
    print("Iteration:", count)
    count += 1
    matches_new = []
    matches_lookup = {}
    for i, match in enumerate(matches_direct):
        # scan if any of these match points have been previously seen
        # and record the match index
        index = -1
        for p in match[2:]:
            key = "%d-%d" % (p[0], p[1])
            if key in matches_lookup:
                index = matches_lookup[key]
                break
        if index < 0:
            # not found, append to the new list
            for p in match[2:]:
                key = "%d-%d" % (p[0], p[1])
                matches_lookup[key] = len(matches_new)
            matches_new.append(list(match)) # shallow copy
        else:
            # found a previous reference, append these match items
            existing = matches_new[index]
            for p in match[2:]:
                key = "%d-%d" % (p[0], p[1])
                found = False
                for e in existing[2:]:
                    if p[0] == e[0]:
                        found = True
                        break
                if not found:
                    # add
                    existing.append(list(p)) # shallow copy
                    matches_lookup[key] = index
            # no 3d location estimation yet
            # # attempt to combine location equitably
            # size1 = len(match[2:])
            # size2 = len(existing[2:])
            # ned1 = np.array(match[0])
            # ned2 = np.array(existing[0])
            # avg = (ned1 * size1 + ned2 * size2) / (size1 + size2)
            # existing[0] = avg.tolist()
            # # print(ned1, ned2, existing[0])
            # # print "new:", existing
            # # print
    if len(matches_new) == len(matches_direct):
        done = True
    else:
        matches_direct = list(matches_new) # shallow copy

# replace the keypoint index in the matches file with the actual kp
# values.  This will save time later and avoid needing to load the
# full original feature files which are quite large.  This also will
# reduce the in-memory footprint for many steps.
print('Replace keypoint indices with uv coordinates.')
for match in matches_direct:
    for m in match[2:]:
        kp = proj.image_list[m[0]].kp_list[m[1]].pt
        m[1] = list(kp)
    # print(match)

sum = 0.0
max = 0
max_index = 0
for i, match in enumerate(matches_direct):
    refs = len(match[2:])
    sum += refs
    if refs > max:
        max = refs
        max_index = i
        
if count >= 1:
    print("Total unique features in image set:", len(matches_direct))
    print("Keypoint average instances:", "%.2f" % (sum / len(matches_direct)))
    print("Max chain length =", max, ' @ index =', max_index)

print("Writing full group chain match file ...")
pickle.dump(matches_direct, open(os.path.join(area_dir, "matches_grouped"), "wb"))

#print "temp: writing ascii version..."
#for match in matches_direct:
#    print match
