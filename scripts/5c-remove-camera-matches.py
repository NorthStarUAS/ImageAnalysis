#!/usr/bin/python3

# remove all match references to a specific image ... can be used as a
# blunt hammer when something is going wrong with that image

import argparse
import pickle
import os

import sys
sys.path.append('../lib')
import ProjectMgr

import match_culling as cull

parser = argparse.ArgumentParser(description='Remove all matches referencing the specific image.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--index', type=int, help='image index')
parser.add_argument('--image', help='image name')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()

source = 'matches_grouped'
print("Loading matches:", source)
matches_orig = pickle.load( open( os.path.join(args.project, source), "rb" ) )
print('Number of original features:', len(matches_orig))
print("Loading optimized matches: matches_opt")
matches_opt = pickle.load( open( os.path.join(args.project, "matches_opt"), "rb" ) )
print('Number of optimized features:', len(matches_opt))

def remove_image_features(index):
    # iterate through the match dictionary and mark any matches for
    # the specified image for deletion
    print("Marking feature matches for image:", index)
    count = 0
    for i, match in enumerate(matches_orig):
        for j, p in enumerate(match[1:]):
            if p[0] == index:
                cull.mark_feature(matches_orig, i, j, 0)
                cull.mark_feature(matches_opt, i, j, 0)
                count += 1
    return count

index = None
if args.image:
    index = proj.findIndexByName(args.image)
    if index == None:
        print("Cannot locate by name:", args.image)
elif args.index:
    if args.index >= len(proj.image_list):
        print("Index greater than image list size:", args.index)
    else:
        index = args.index
        
if index != None:
    count = remove_image_features(index)
else:
    count = 0
    
if count > 0:
    print('Features marked:', count)
    result = input('Delete these matches and save? (y/n):')
    if result == 'y' or result == 'Y':
        cull.delete_marked_features(matches_orig)
        cull.delete_marked_features(matches_opt)
        # write out the updated match dictionaries
        print("Writing:", source)
        pickle.dump(matches_orig, open(os.path.join(args.project, source), "wb"))
        print("Writing optimized matches...")
        pickle.dump(matches_opt, open(os.path.join(args.project, "matches_opt"), "wb"))

