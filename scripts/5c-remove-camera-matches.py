#!/usr/bin/python3

# remove all match references to a specific image ... can be used as a
# blunt hammer when something is going wrong with that image

import argparse
import pickle
import os

import sys
sys.path.append('../lib')
import Groups
import ProjectMgr

import match_culling as cull

parser = argparse.ArgumentParser(description='Remove all matches referencing the specific image.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--area', default='area-00', help='sub area directory')
parser.add_argument('--group', type=int, default=0, help='group number')
parser.add_argument('--indices', nargs='+', type=int, help='image index')
parser.add_argument('--images', nargs='+', help='image names')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_area_info(args.area)

area_dir = os.path.join(args.project, args.area)

print("Loading matches_grouped...")
matches = pickle.load( open( os.path.join(area_dir, "matches_grouped"), "rb" ) )
print("  features:", len(matches))

# load the group connections within the image set
groups = Groups.load(area_dir)

def split_image_features(index, matches):
    # iterate through the match dictionary and mark any matches for
    # the specified image for deletion
    print("Marking feature matches for image:", index)
    count = 0
    new_matches = []
    for i, match in enumerate(matches):
        found_index = False
        found_group = False
        for j, p in enumerate(match[2:]):
            if p[0] == index:
                found_index = True
            elif proj.image_list[p[0]].name in groups[args.group]:
                found_group = True
        # split match if possible
        if found_index and found_group:
            count += 1
            new_match = [ list(match[0]), -1 ]
            for j, p in enumerate(match[2:]):
                if proj.image_list[p[0]].name in groups[args.group]:
                    if p[0] != index:
                        cull.mark_feature(matches, i, j, 0)
                        # print('p:', p)
                        new_match.append(p)
            if len(new_match) >= 4: # at least 2 images referenced
                new_matches.append(new_match)
    # add all the new match splits
    for m in new_matches:
        matches.append(m)
    return count, len(new_matches)

index = None
count_split = 0
count_added = 0
if not args.images is None:
    for name in args.images:
        index = proj.findIndexByName(name)
        if index == None:
            print("Cannot locate by name:", args.images)
        else:
            s, a = split_image_features(index, matches)
            count_split += s
            count_added += a
elif not args.indices is None:
    for index in args.indices:
        if args.index >= len(proj.image_list):
            print("Index greater than image list size:", args.index)
        else:
            s, a = split_image_features(index, matches)
            count_split += s
            count_added += a
    
if count_split + count_added > 0:
    print('Features with group removed:', count_split)
    print('New Features for group with target removed:', count_added)
    result = input('Update these matches and save? (y/n):')
    if result == 'y' or result == 'Y':
        cull.delete_marked_features(matches)
      
        # write out the updated match dictionaries
        print("Writing: matches_grouped")
        pickle.dump(matches, open(os.path.join(area_dir, "matches_grouped"), "wb"))

