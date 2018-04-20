#!/usr/bin/python3

import argparse
import pickle
import numpy as np
import os.path
from progress.bar import Bar
import sys

from props import getNode

sys.path.append('../lib')
import Matcher
#import Pose
import ProjectMgr
import SRTM

# Reset all match point locations to their original direct
# georeferenced locations based on estimated camera pose and
# projection onto DEM earth surface

# extends 4b-reset-matches-ned2 by only joining chains with similar 3d
# locations ... this should be what we want to do, but I believe there
# is a bug somewhere in linking or joinging or referencing that leads
# to wrong associations and then messes up the downstream solver.  So
# someday would like to come back to this and figure out what is going
# wrong because ultimately if we squeeze out redundancy, the optimized
# solution should be better.

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--full-grouping', action='store_true', help='maximal feature grouping (caution: can blow up the sba process for a not-yet-known reason)') # fixme: full grouping is probably safe again, but we have a separate script now that does this.
parser.add_argument('--ground', type=float, help='ground elevation in meters')

args = parser.parse_args()

m = Matcher.Matcher()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()
proj.load_features()
proj.undistort_keypoints()
proj.load_match_pairs(extra_verbose=False)

# compute keypoint usage map
proj.compute_kp_usage()

if args.ground:
    proj.fastProjectKeypointsToGround(args.ground)
else:
    # lookup ned reference
    ref_node = getNode("/config/ned_reference", True)
    ref = [ ref_node.getFloat('lat_deg'),
            ref_node.getFloat('lon_deg'),
            ref_node.getFloat('alt_m') ]
    
    # setup SRTM ground interpolator
    sss = SRTM.NEDGround( ref, 2000, 2000, 30 )

    # fast way:
    # 1. make a grid (i.e. 8x8) of uv coordinates covering the whole image
    # 2. undistort these uv coordinates
    # 3. project them into vectors
    # 4. intersect them with the srtm terrain to get ned coordinates
    # 5. use linearndinterpolator ... g = scipy.interpolate.LinearNDInterpolator([[0,0],[1,0],[0,1],[1,1]], [[0,4,8],[1,3,2],[2,2,-4],[4,1,0]])
    #    with origin uv vs. 3d location to build a table
    # 6. interpolate original uv coordinates to 3d locations
    proj.fastProjectKeypointsTo3d(sss)

# For some features detection algorithms we expect duplicated feature
# uv coordinates.  These duplicates may have different scaling or
# other attributes important during feature matching, yet ultimately
# resolve to the same uv coordinate in an image.
print("Indexing features by unique uv coordinates...")
for image in proj.image_list:
    print(image.name)
    # pass one, build a tmp structure of unique keypoints (by uv) and
    # the index of the first instance.
    image.kp_remap = {}
    used = 0
    for i, kp in enumerate(image.kp_list):
        if image.kp_used[i]:
            used += 1
            key = "%.2f-%.2f" % (kp.pt[0], kp.pt[1])
            if not key in image.kp_remap:
                image.kp_remap[key] = i
            else:
                pass
                print("%d -> %d" % (i, image.kp_remap[key]))
                print(" ", image.coord_list[i], image.coord_list[image.kp_remap[key]])
    print(" features used:", used)
    print(" unique by uv and used:", len(image.kp_remap))

# after feature matching we don't care about other attributes, just
# the uv coordinate.
#
# notes: we do a first pass duplicate removal during the actual
# matching process.  This removes 1->many relationships, or duplicate
# matches at different scales within a match pair.  However, different
# pairs could reference the same keypoint at different scales, so
# duplicates could still exist.  This finds all the duplicates within
# the entire match set and collapses them down to eliminate any
# redundancy.
print("Collapsing keypoints with duplicate uv coordinates...")
for i, i1 in enumerate(proj.image_list):
    for j, matches in enumerate(i1.match_list):
        count = 0
        i2 = proj.image_list[j]
        for k, pair in enumerate(matches):
            # print pair
            idx1 = pair[0]
            idx2 = pair[1]
            kp1 = i1.kp_list[idx1]
            kp2 = i2.kp_list[idx2]
            key1 = "%.2f-%.2f" % (kp1.pt[0], kp1.pt[1])
            key2 = "%.2f-%.2f" % (kp2.pt[0], kp2.pt[1])
            # print key1, key2
            new_idx1 = i1.kp_remap[key1]
            new_idx2 = i2.kp_remap[key2]
            # count the number of match rewrites
            if idx1 != new_idx1 or idx2 != new_idx2:
                count += 1
            if idx1 != new_idx1:
                # sanity check
                uv1 = list(i1.kp_list[idx1].pt)
                new_uv1 = list(i1.kp_list[new_idx1].pt)
                if not np.allclose(uv1, new_uv1):
                    print("OOPS!!!")
                    print("  index 1: %d -> %d" % (idx1, new_idx1))
                    print("  [%.2f, %.2f] -> [%.2f, %.2f]" % (uv1[0], uv1[1],
                                                              new_uv1[0],
                                                              new_uv1[1]))
            if idx2 != new_idx2:
                # sanity check
                uv2 = list(i2.kp_list[idx2].pt)
                new_uv2 = list(i2.kp_list[new_idx2].pt)
                if not np.allclose(uv2, new_uv2):
                    print("OOPS!")
                    print("  index 2: %d -> %d" % (idx2, new_idx2))
                    print("  [%.2f, %.2f] -> [%.2f, %.2f]" % (uv2[0], uv2[1],
                                                              new_uv2[0],
                                                              new_uv2[1]))
            # rewrite matches
            matches[k] = [new_idx1, new_idx2]
        if count > 0:
            print('Match:', i, 'vs', j, 'matches:', len(matches), 'rewrites:', count)

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
      
# after collapsing by uv coordinate, we could be left with duplicate
# matches (matched at different scales or other attributes, but same
# exact point.)
#
# notes: this really shouldn't (!) (by my best current understanding)
# be able to find any dups.  These should all get caught in the
# original pair matching step.  But some dupes seem to survive?
# Hmmm...
print("Eliminating pair duplicates...")
for i, i1 in enumerate(proj.image_list):
    for j, matches in enumerate(i1.match_list):
        i2 = proj.image_list[j]
        count = 0
        pair_dict = {}
        new_matches = []
        for k, pair in enumerate(matches):
            key = "%d-%d" % (pair[0], pair[1])
            if not key in pair_dict:
                pair_dict[key] = True
                new_matches.append(pair)
            else:
                count += 1
        if count > 0:
            print('Match:', i, 'vs', j, 'matches:', len(matches), 'dups:', count)
      
        i1.match_list[j] = new_matches
        
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

# Do we have a keypoint in i1 matching multiple keypoints in i2?
#
# Notes: again these shouldn't exist here, but let's check anyway.  If
# we start finding these here, I should hunt for the reason earlier in
# the code that lets some through, or try to understand what larger
# logic principle allows somne of these to still exist here.
print("Testing for 1 vs. n keypoint duplicates...")
for i, i1 in enumerate(proj.image_list):
    for j, matches in enumerate(i1.match_list):
        i2 = proj.image_list[j]
        count = 0
        kp_dict = {}
        for k, pair in enumerate(matches):
            if not pair[0] in kp_dict:
                kp_dict[pair[0]] = pair[1]
            else:
                print("Warning keypoint idx", pair[0], "already used in another match.")
                uv2a = list(i2.kp_list[ kp_dict[pair[0]] ].pt)
                uv2b = list(i2.kp_list[ pair[1] ].pt)
                if not np.allclose(uv2, new_uv2):
                    print("  [%.2f, %.2f] -> [%.2f, %.2f]" % (uv2a[0], uv2a[1],
                                                              uv2b[0], uv2b[1]))
                count += 1
        if count > 0:
            print('Match:', i, 'vs', j, 'matches:', len(matches), 'dups:', count)

def update_match_location(match):
    sum = np.array( [0.0, 0.0, 0.0] )
    for p in match[1:]:
        # print proj.image_list[ p[0] ].coord_list[ p[1] ]
        sum += proj.image_list[ p[0] ].coord_list[ p[1] ]
        ned = sum / len(match[1:])
        # print "avg =", ned
        match[0] = ned.tolist()
    return match
    
print("Constructing unified match structure...")
# create an initial pair-wise match list
matches_direct = []
for i, img in enumerate(proj.image_list):
    # print img.name
    for j, matches in enumerate(img.match_list):
        # print proj.image_list[j].name
        if j > i:
            for pair in matches:
                match = []
                # ned place holder
                match.append([0.0, 0.0, 0.0])
                match.append([i, pair[0]])
                match.append([j, pair[1]])
                update_match_location(match)
                matches_direct.append(match)
                # print pair, match

# compute the 3d distance between two points
def dist3d(p0, p1):
    dist = np.linalg.norm(np.array(p0) - np.array(p1))
    return dist

# collect/group match chains that refer to the same keypoint

# warning 1: if there are bad matches this can over-constrain the
# problem or tie the pieces together too tightly/incorrectly and lead
# to the optimizer exploding.)

# warning 2: if there are bad matches, this can lead to linking chains
# together that shouldn't be linked which really fouls things up.

# warning 3: I'm not 100% sure this code is rock solid.  I've been
# through it dozens of times, but something ... I don't know ... it's
# worth another deep dive to validate the output of this section.

# notes: with only pairwise matching, the optimizer should have a good
# opportunity to generate a nice fit.  However, subsections within the
# solution could wander away from each other leading to some weird
# projection results at the intersection of these divergent groups.
# Ultimately we need to have at least some 3+ way feature match chains
# to sufficiently constrain the problem set.  This dragon here will
# need to be slayed at some point soon.  It's just a few short lines
# of code, but don't let that fool you!

count = 0
if args.full_grouping:
    done = False
else:
    done = True
while not done:
    print("Iteration:", count)
    count += 1
    matches_new = []
    matches_lookup = {}
    for i, match in enumerate(matches_direct):
        # scan if any of these match points have been previously seen
        # and record the match index
        index = -1
        for p in match[1:]:
            key = "%d-%d" % (p[0], p[1])
            if key in matches_lookup:
                index = matches_lookup[key]
                break
        if index < 0:
            # not found, append to the new list
            for p in match[1:]:
                key = "%d-%d" % (p[0], p[1])
                matches_lookup[key] = len(matches_new)
            matches_new.append(list(match)) # shallow copy
        else:
            # found a previous reference, append these match items
            existing = matches_new[index]
            for p in match[1:]:
                key = "%d-%d" % (p[0], p[1])
                found = False
                for e in existing[1:]:
                    if p[0] == e[0]:
                        found = True
                        break
                if not found:
                    # add
                    existing.append(list(p)) # shallow copy
                    matches_lookup[key] = index
            # print "new:", existing
            # print 
    if len(matches_new) == len(matches_direct):
        done = True
    else:
        matches_direct = list(matches_new) # shallow copy

#for m in matches_direct:
#    print m
    
count = 0.0
sum = 0.0
for match in matches_direct:
    sum += len(match) - 1
    count += 1
        
if count >= 1:
    print("total unique features in image set = %d" % count)
    print("keypoint average instances = %.4f" % (sum / count))

# compute an initial guess at the 3d location of each unique feature
# by averaging the locations of each projection
print("Estimating world coordinates of each keypoint...")
for match in matches_direct:
    sum = np.array( [0.0, 0.0, 0.0] )
    for p in match[1:]:
        #if len(match) >= 4: print proj.image_list[ p[0] ].coord_list[ p[1] ]
        sum += proj.image_list[ p[0] ].coord_list[ p[1] ]
    ned = sum / len(match[1:])
    # if len(match) >= 4: print "avg =", ned
    match[0] = ned.tolist()

print("Writing match file ...")
direct_file = os.path.join(args.project, "matches_direct")
pickle.dump(matches_direct, open(direct_file, "wb"))

#print "temp: writing ascii version..."
#for match in matches_direct:
#    print match
