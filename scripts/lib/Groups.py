# construct groups of connected images.  The inclusion order favors
# images with the most connections (features matches) to neighbors.

import cv2
import json
import numpy as np
import os
import sys

min_group = 10
min_connections = 25
max_wanted = 100

# this builds a simple set structure that records if any image has any
# connection to any other image
# def countFeatureConnections(image_list, matches):
#     for image in image_list:
#         image.connection_set = set()
#     for i, match in enumerate(matches):
#         for mi in match[2:]:
#             for mj in match[2:]:
#                 if mi != mj:
#                     image_list[mi[0]].connection_set.add(mj[0])
#     for i, image in enumerate(image_list):
#         # print image.name
#         # for j in image.connection_set:
#         #     print '  pair len', j, len(image.match_list[j])
#         image.connection_count = len(image.connection_set)
#         # print image.name, i, image.connection_set
#         for j in range(len(image.match_list)):
#             size = len(image.match_list[j])
#             if size > 0 and not j in image.connection_set:
#                 print('  matches, but no connection')

# for unallocated features, count the number of connections into the
# current placed group
def updateAvailableFeatures(group_images, matches, avail_features):
    for i, match in enumerate(matches):
        if match[1] < 0:
            count = 0
            for m in match[2:]:
                if m[0] in group_images:
                    count += 1
            avail_features[i] = count
        else:
            avail_features[i] = 0

# This is the current, best grouping function to use.

# A connection must have at least "min" number of features, and we
# will use at most "max" features even if more are available.
#
# We create a lookaside data structure so we can use (prioritize) the
# longest feature chains first.  This maximize redunancy and links all
# the images together with the most and best connections possible.
#
# Single pair connections are a problem.  If an image only connects to
# a single image in the placed set, there is not enough redundancy and
# the new image (and anything that connects/hangs off that one image
# through single pair connections can end up walking off to an
# arbitrary other location relative to the main set.  We try to deal
# with this by allowing single pair connections, but only if we use a
# balanced set into multiple placed images.  This creates/requires
# some extra accounting work.

def groupByFeatureConnections(image_list, matches):
    # countFeatureConnections(image_list, matches)
    print("Start of top level grouping algorithm...")

    # mark all features as unaffiliated
    for match in matches:
        match[1] = -1
        
    # start with no placed images or features
    placed_images = set()
    groups = []
    avail_features = [0] * len(matches)

    done = False
    while not done:
        print("Start of new group...")
        # start a fresh group
        group_images = set()
        group_level = len(groups)
        
        # find the unaffiliated feature with the most connections to
        # unplaced images
        max_connections = 2
        max_index = -1
        for i, match in enumerate(matches):
            if match[1] < 0:
                count = 0
                for m in match[2:]:
                    if not m[0] in placed_images:
                        count += 1
                if count > max_connections:
                    max_connections = count
                    max_index = i
        if max_index == -1:
            break
        print('Feature with max connections (%d) = %d' % (max_connections, max_index))
        print('Seeding group with:', end=" ")
        match = matches[max_index]
        for m in match[2:]:
            group_images.add(m[0])
            placed_images.add(m[0])
            print(image_list[m[0]].name, end=" ")
        print()
        updateAvailableFeatures(group_images, matches, avail_features)
        
        while True:
            # find the unplaced images with sufficient connections
            # into the placed set

            # per image, per connection feature aggregator
            image_counter = [dict() for x in range(len(image_list))]
            # count up the placed feature references to each image,
            # binned by how many references to already placed images.
            for i, match in enumerate(matches):
                # only proceed if this feature has been placed (i.e. it
                # connects to two or more placed images)
                num = avail_features[i]
                if num >= 1:
                    for m in match[2:]:
                        if num in image_counter[m[0]]:
                            image_counter[m[0]][num].append(i)
                        else:
                            image_counter[m[0]][num] = [ i ]
            # for each unconnected image, count connected images in
            # the placed set
            image_connections = [set() for x in range(len(image_list))]
            for i in range(len(image_list)):
                if not i in placed_images:
                    for key in sorted(image_counter[i].keys(), reverse=True):
                        for j in image_counter[i][key]:
                            match = matches[j]
                            for m in match[2:]:
                                if m[0] != i and m[0] in placed_images:
                                    image_connections[i].add(m[0])
            # add all unplaced images with more than min_connections to
            # the placed set.  Prioritize features with most connections to the placed set
            print("Report:")
            add_count = 0
            for i in range(len(image_counter)):
                if not i in placed_images:
                    total_avail = 0
                    total_found = 0
                    # total placed features for this image
                    for key in image_counter[i].keys():
                        total_avail += len(image_counter[i][key])
                    if total_avail >= min_connections and len(image_connections[i]) > 1:
                        print("%s(%d):" % (image_list[i].name, i), end=" ")
                        # use the most redundant first
                        for key in sorted(image_counter[i].keys(), reverse=True):
                            if total_found < max_wanted:
                                print("%d:%d" % (key, len(image_counter[i][key])),
                                      end=" ")
                                avail = len(image_counter[i][key])
                                want = max_wanted - total_found
                                #print('total_found:', total_found,
                                #      'want:', want,
                                #      'avail:', avail)
                                if avail <= want:
                                    # use the whole set
                                    total_found += len(image_counter[i][key])
                                    for j in image_counter[i][key]:
                                        matches[j][1] = group_level
                                else:
                                    # use what we need
                                    for k in range(0, want):
                                        #print(' using:', k)
                                        total_found += 1
                                        j = image_counter[i][key][k]
                                        matches[j][1] = group_level
                        placed_images.add(i)
                        group_images.add(i)
                        add_count += 1
                        print("[%d/%d]" % (total_found, total_avail), end=" ")
                        print()
            if add_count == 0:
                # no more images could be connected
                if len(group_images) >= min_group:
                    group_list = []
                    for i in list(group_images):
                        group_list.append(image_list[i].name)
                    groups.append(group_list)
                if len(group_images) < 3:
                    done = True
                break
            updateAvailableFeatures(group_images, matches, avail_features)
    return groups

# return the number of connections into the placed set
def numPlacedConnections(image, proj):
    count = 0
    total_matches = 0
    for key in image.match_list:
        num_matches = len(image.match_list[key])
        i2 = proj.findImageByName(key)
        if i2.group_starter:
            # artificially inflate count if we are connected to a group starter
            count += 1000
        if i2.placed:
            count += 1
            total_matches += num_matches
    return count, total_matches

def groupByImageConnections(proj):
    # reset the cycle distance for all images
    for image in proj.image_list:
        image.placed = False
        image.group_starter = False
        
    for image in proj.image_list:
        image.total_connections = 0
        for key in image.match_list:
            if proj.findImageByName(key):
                image.total_connections += 1
        #if image.total_connections > 1:
        #    print("%s: connections: %d" % (image.name, image.total_connections))

    group_list = []
    group = []
    done = False
    while not done:
        done = True
        best_index = -1
        # find the unplaced image with the most placed neighbors
        best_connections = 0
        for i, image in enumerate(proj.image_list):
            if not image.placed:
                connections, total_matches = numPlacedConnections(image, proj)
                if connections > best_connections and total_matches > 25:
                    best_index = i
                    best_connections = connections
                    done = False
        if best_index < 0 or best_connections < 3:
            print("Cannot find an unplaced image with a double connected neighbor.")
            if len(group) >= min_group:
                # commit the previous group (if it is long enough to be useful)
                group_list.append(group)
            # and start a new group
            group = []
            # now find an unplaced image that has the most connections
            # to other images (new cycle start)
            max_connections = 0
            for i, image in enumerate(proj.image_list):
                if not image.placed:
                    if (image.total_connections > max_connections):
                        max_connections = image.total_connections
                        best_index = i
                        done = False
                        # print(" found image {} connections = {}".format(i, max_connections))
            if best_index >= 0:
                # group starter!
                print("Starting a new group with:",
                      proj.image_list[best_index].name)
                proj.image_list[best_index].group_starter = True
        if best_index >= 0:
            image = proj.image_list[best_index]
            image.placed = True
            print("Adding: {} (placed connections = {}, total connections = {})".format(image.name, best_connections, image.total_connections), )
            group.append(image.name)

    print("Group (cycles) report:")
    for group in group_list:
        #if len(group) < 2:
        #    continue
        print("group (size = {}):".format((len(group))))
        for name in group:
            image = proj.findImageByName(name)
            print("{} ({})".format(image.name, image.total_connections))
        print("")

    return group_list

def save(path, groups):
    file = os.path.join(path, 'groups.json')
    try:
        fd = open(file, 'w')
        json.dump(groups, fd, indent=4, sort_keys=True)
        fd.close()
    except:
        print('{}: error saving file: {}'.format(file, str(sys.exc_info()[1])))

def load(path):
    file = os.path.join(path, 'groups.json')
    try:
        fd = open(file, 'r')
        groups = json.load(fd)
        fd.close()
    except:
        print('{}: error loading file: {}'.format(file, str(sys.exc_info()[1])))
        groups = []
    return groups
