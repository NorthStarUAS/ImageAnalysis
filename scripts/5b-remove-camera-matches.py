#!/usr/bin/python3

# remove all match references to a specific image ... can be used as a
# blunt hammer when something is going wrong with that image

import argparse
import pickle
import os

from props import getNode

from lib import Groups
from lib import ProjectMgr
from lib import match_culling as cull

parser = argparse.ArgumentParser(description='Remove all matches referencing the specific image.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--group', type=int, default=0, help='group number')
parser.add_argument('--indices', nargs='+', type=int, help='image index')
parser.add_argument('--images', nargs='+', help='image names')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()

print("Loading matches_grouped...")
matches = pickle.load( open( os.path.join(proj.analysis_dir, "matches_grouped"), "rb" ) )
print("  features:", len(matches))

# load the group connections within the image set
groups = Groups.load(proj.analysis_dir)

# a value of 2 let's pairs exist which can be trouble ...
matcher_node = getNode('/config/matcher', True)
min_chain_len = matcher_node.getInt("min_chain_len")
print("Notice: min_chain_len is:", min_chain_len)

def mark_image_features(index, matches):
    # iterate through the match dictionary and mark any matches for
    # the specified image for deletion
    print("Marking feature matches for image:", index)
    count = 0
    new_matches = []
    for i, match in enumerate(matches):
        for j, p in enumerate(match[2:]):
            if p[0] == index:
                cull.mark_feature(matches, i, j, 0)
                count += 1
    return count

index = None
count_split = 0
count_added = 0
if not args.images is None:
    for name in args.images:
        index = proj.findIndexByName(name)
        if index == None:
            print("Cannot locate by name:", args.images)
        elif not name in groups[args.group]:
            print(name, "not in selected group.")
        else:
            count = mark_image_features(index, matches)
            groups[args.group].remove(name)
elif not args.indices is None:
    for index in args.indices:
        if index >= len(proj.image_list):
            print("Index greater than image list size:", index)
        else:
            count = mark_image_features(index, matches)
            groups[args.group].remove(proj.image_list[index].name)
    
if count > 0:
    print('Image removed from %d features.' % count)
    result = input('Save these changes? (y/n):')
    if result == 'y' or result == 'Y':
        cull.delete_marked_features(matches, min_chain_len)
        print("Updating groups file")
        Groups.save(proj.analysis_dir, groups)
        print("Writing: matches_grouped")
        pickle.dump(matches, open(os.path.join(proj.analysis_dir, "matches_grouped"), "wb"))
