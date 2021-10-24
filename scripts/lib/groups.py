# construct groups of connected images.  The inclusion order favors
# images with the most connections (features matches) to neighbors.

import cv2
import json
import math
import numpy as np
import os
import sys

from props import getNode

from .logger import log

#min_group = 10
min_group = 7
min_connections = 25
max_wanted = 250                # possibly overridden later

def my_add(placed_matches, matches, group_level, i):
    # print("adding feature:", i)
    for m in matches[i][2:]:
        placed_matches[m[0]] += 1
    matches[i][1] = group_level
        
# NEW GROUPING TEST
def compute(image_list, matches):
    # notice: we assume that matches have been previously sorted by
    # longest chain first!
    
    log("Start of grouping algorithm...")

    matcher_node = getNode('/config/matcher', True)
    min_chain_len = matcher_node.getInt("min_chain_len")
    if min_chain_len == 0:
        min_chain_len = 3
    log("/config/matcher/min_chain_len:", min_chain_len)
    use_single_pairs = (min_chain_len == 2)

    max_wanted = int(8000 / math.sqrt(len(image_list)))
    if max_wanted < 200:
        max_wanted = 200
    log("max features desired per image:", max_wanted)
    print("Notice: I should really work on this formula ...")
    
    # mark all features as unaffiliated
    for match in matches:
        match[1] = -1
        
    # start with no placed images or features
    placed_images = set()
    groups = []

    done = False
    while not done:
        group_level = len(groups)
        log("Start of new group level:", group_level)
        
        placed_matches = [0] * len(image_list)
        
        # find the unused feature with the most connections to
        # unplaced images
        max_connections = 2
        seed_index = -1
        for i, match in enumerate(matches):
            if match[1] < 0:
                count = 0
                connected = False
                for m in match[2:]:
                    if m[0] in placed_images:
                        connected = True
                    else:
                        count += 1
                if not connected and count > max_connections:
                    max_connections = count
                    seed_index = i
        if seed_index == -1:
            break
        log("Seed index:", seed_index, "connections:", max_connections)
        match = matches[seed_index]
        m = match[3]            # first image referenced by match
        # group_images.add(m[0])
        my_add(placed_matches, matches, group_level, seed_index)
        seed_image = m[0]
        log('Seeding group with:', image_list[seed_image].name)

        still_working = True
        iteration = 0
        while still_working:
            log("Iteration:", iteration)
            still_working = False
            for i, match in enumerate(matches):
                if match[1] < 0 and (use_single_pairs or len(match[2:]) > 2):
                    # determine if we should add this feature
                    placed_count = 0
                    placed_need_count = 0
                    unplaced_count = 0
                    seed_connection = False
                    for m in match[2:]:
                        if m[0] in placed_images:
                            # placed in a previous grouping, skip
                            continue
                        if m[0] == seed_image:
                            seed_connection = True
                        if placed_matches[m[0]] >= max_wanted:
                            placed_count += 1
                        elif placed_matches[m[0]] >= min_connections:
                            placed_count += 1
                            placed_need_count += 1
                        elif placed_matches[m[0]] > 0:
                            placed_need_count += 1
                        else:
                            unplaced_count += 1
                    # print("Match:", i, placed_count, seed_connection, placed_need_count, unplaced_count)
                    if placed_count > 1 or (use_single_pairs and placed_count > 0) or seed_connection:
                        if placed_need_count > 0 or unplaced_count > 0:
                            my_add(placed_matches, matches, group_level, i)
                            still_working = True
            iteration += 1
            
        # count up the placed images in this group
        group_images = set()
        for i in range(len(image_list)):
            if placed_matches[i] >= min_connections:
                group_images.add(i)
        group_list = []
        for i in list(group_images):
            placed_images.add(i)
            group_list.append(image_list[i].name)
        if len(group_images) >= min_group:
            log(group_list)
            groups.append(group_list)
        if len(group_images) < 3:
            done = True
    return groups

def save(path, groups):
    file = os.path.join(path, 'groups.json')
    try:
        fd = open(file, 'w')
        json.dump(groups, fd, indent=4, sort_keys=True)
        fd.close()
    except:
        log('{}: error saving file: {}'.format(file, str(sys.exc_info()[1])))

def load(path):
    file = os.path.join(path, 'groups.json')
    try:
        fd = open(file, 'r')
        groups = json.load(fd)
        fd.close()
    except:
        log('{}: error loading file: {}'.format(file, str(sys.exc_info()[1])))
        groups = []
    return groups
