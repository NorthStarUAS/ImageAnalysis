# construct groups of connected images.  The inclusion order favors
# images with the most connections (features matches) to neighbors.

import cv2
import json
import numpy as np
import os
import sys

# this builds a simple set structure that records if any image has any
# connection to any other image
def countFeatureConnections(image_list, matches):
    for image in image_list:
        image.connection_set = set()
    for i, match in enumerate(matches):
        for mi in match[1:]:
            for mj in match[1:]:
                if mi != mj:
                    image_list[mi[0]].connection_set.add(mj[0])
    for i, image in enumerate(image_list):
        # print image.name
        # for j in image.connection_set:
        #     print '  pair len', j, len(image.match_list[j])
        image.connection_count = len(image.connection_set)
        # print image.name, i, image.connection_set
        for j in range(len(image.match_list)):
            size = len(image.match_list[j])
            if size > 0 and not j in image.connection_set:
                print '  matches, but no connection'
        
def updatePlacedFeatures(placed_images, matches, placed_features):
    for i, match in enumerate(matches):
        for m in match[1:]:
            if m[0] in placed_images:
                placed_features[i] = True

# This is the current, best grouping function to use
def groupByFeatureConnections(image_list, matches):
    countFeatureConnections(image_list, matches)
    print "Start of top level grouping algorithm..."

    # start with no placed images or features
    placed_images = set()
    groups = []
    placed_features = [False] * len(matches)

    # wipe connection order for all images
    for image in image_list:
        image.connection_order = -1

    done = False
    while not done:
        print "Start of new group..."
        # start a fresh group
        group_images = set()
        
        # find the unplaced image with the most connections to other
        # images
        max_connections = 0
        max_index = -1
        for i, image in enumerate(image_list):
            if image.connection_order < 0 and image.connection_count > max_connections:
                max_connections = image.connection_count
                max_index = i
        max_image = image_list[max_index]
        max_image.connection_order = 0
        print "Image with max connections:", max_image.name, "num:", max_connections
        placed_images.add(max_index)
        group_images.add(max_index)
        updatePlacedFeatures(placed_images, matches, placed_features)

        while True:
            # find the unplaced image with the most connections into
            # the placed set

            # per image counter
            image_counter = [0] * len(image_list)

            # count up the placed feature references to unplaced images
            for i, match in enumerate(matches):
                # only proceed if this feature has been placed (i.e. it
                # connects to two or more placed images)
                if placed_features[i]:
                    for m in match[1:]:
                        if not m[0] in placed_images:
                            image_counter[m[0]] += 1
            # print 'connected image count:', image_counter
            new_index = -1
            max_connections = -1
            for i in range(len(image_counter)):
                if image_counter[i] > max_connections:
                    new_index = i
                    max_connections = image_counter[i]
            if max_connections >= 25:
                # print "New image with max connections:", image_list[new_index].name
                # print "Number of connected features:", max_connections
                placed_images.add(new_index)
                group_images.add(new_index)
            else:
                if len(group_images) > 1:
                    groups.append(list(group_images))
                else:
                    done = True
                break

            updatePlacedFeatures(placed_images, matches, placed_features)

            new_image = image_list[new_index]
            new_image.connection_order = len(placed_images) - 1
            print 'Added:', new_image.name, 'groups:', len(groups)+1, 'in current group', len(group_images), 'total:', len(placed_images)
            
    # add all unplaced images in their own groups of 1
    for i, image in enumerate(image_list):
        if not i in placed_images:
            groups.append( [i] )
            
    print groups
    return groups


# for the specified image estimate the image area covered by
# connections to placed images.
def estimateConnectionArea(image):
    pass


# speculative ....
def groupByConnectedArea(image_list, matches):
    countFeatureConnections(image_list, matches)
    print "Start of top level grouping algorithm..."

    # start with no placed images or features
    placed_images = set()
    groups = []
    placed_features = [False] * len(matches)

    # wipe connection order for all images
    for image in image_list:
        image.connection_order = -1

    done = False
    while not done:
        print "Start of new group..."
        # start a fresh group
        group_images = set()
        
        # find the unplaced image with the most connections to other
        # images
        max_connections = 0
        max_index = -1
        for i, image in enumerate(image_list):
            if image.connection_order < 0 and image.connection_count > max_connections:
                max_connections = image.connection_count
                max_index = i
        max_image = image_list[max_index]
        max_image.connection_order = 0
        print "Image with max connections:", max_image.name, "num:", max_connections
        placed_images.add(max_index)
        group_images.add(max_index)
        updatePlacedFeatures(placed_images, matches, placed_features)

        while True:
            # find the unplaced image with the largest connection area
            # into the placed set

            # per image counter
            # image_counter = [0] * len(image_list)

            # clear the placed feature lists
            for i, image in enumerate(image_list):
                image.placed_feature_list = []
                    
            # assemble the placed feature lists for each unplaced image
            for i, match in enumerate(matches):
                # only proceed if this feature has been placed (i.e. it
                # connects to two or more placed images)
                if placed_features[i]:
                    for m in match[1:]:
                        if not m[0] in placed_images:
                            uv = image_list[m[0]].uv_list[m[1]]
                            image_list[m[0]].placed_feature_list.append(uv)

            # find the minarearect bounds for each the connected
            # points in each image
            for image in image_list:
                if len(image.placed_feature_list):
                    center, (w, h), angle = cv2.minAreaRect(np.array(image.placed_feature_list))
                    print w, h, w*h
                    image.connected_area = w*h
            
            # print 'connected image count:', image_counter
            new_index = -1
            max_area = -1
            for i, image in enumerate(image_list):
                if len(image.placed_feature_list):
                    if image.connected_area > max_area:
                        new_index = i
                        max_area = image.connected_area
            if max_area >= 10000:
                print "New image with max area:", image_list[new_index].name,
                print "area:", image_list[new_index].connected_area
                placed_images.add(new_index)
                group_images.add(new_index)
            else:
                if len(group_images) > 1:
                    groups.append(list(group_images))
                else:
                    done = True
                break

            updatePlacedFeatures(placed_images, matches, placed_features)

            new_image = image_list[new_index]
            new_image.connection_order = len(placed_images) - 1
            print 'Added:', new_image.name, 'groups:', len(groups)+1, 'in current group', len(group_images), 'total:', len(placed_images)
            
    # add all unplaced images in their own groups of 1
    for i, image in enumerate(image_list):
        if not i in placed_images:
            groups.append( [i] )
            
    print groups
    return groups

def save(path, groups):
    file = os.path.join(path, 'Groups.json')
    try:
        fd = open(file, 'w')
        json.dump(groups, fd, indent=4, sort_keys=True)
        fd.close()
    except:
        print file + ": error saving file:", str(sys.exc_info()[1])

def load(path):
    file = os.path.join(path, 'Groups.json')
    try:
        fd = open(file, 'r')
        groups = json.load(fd)
        fd.close()
    except:
        print file + ": error loading file:", str(sys.exc_info()[1])
        groups = []
    return groups
   
##
## For historical/archival purposes, here is an alternative grouping
## function that uses number of neighbor with matches instead of
## number of total number of matches.  The results should be
## identical, but the order images are added to the group may be
## slightly different...
##

# return the neighbor that is closest to the root node of the
# placement tree (i.e. smallest cycle_depth.
def bestNeighbor(image, image_list):
    best_cycle_depth = len(image_list) + 1
    best_index = None
    for i, pairs in enumerate(image.match_list):
        if len(pairs):
            i2 = image_list[i]
            dist = i2.cycle_depth
            # print "  neighbor check %d = %d" % ( i, dist )
            if dist >= 0 and dist < best_cycle_depth:
                # print '    new best:', dist
                best_cycle_depth = dist
                best_index = i
    return best_index, best_cycle_depth

def groupByImageConnections(image_list):
    # reset the cycle distance for all images
    for image in image_list:
        image.cycle_depth = -1
        
    for image in image_list:
        image.connections = 0
        for match in image.match_list:
            if len(match) >= 8:
                image.connections += 1
        if image.connections > 1:
            print "%s connections: %d" % (image.name, image.connections)

    last_cycle_depth = len(image_list) + 1
    group_list = []
    group = []
    done = False
    while not done:
        done = True
        best_index = None
        # find an unplaced image with a placed neighbor that is the
        # closest conection to the root of the placement tree.
        best_cycle_depth = len(image_list) + 1
        for i, image in enumerate(image_list):
            if image.cycle_depth < 0:
                index, cycle_depth = bestNeighbor(image, image_list)
                if cycle_depth >= 0 and (cycle_depth+1 < best_cycle_depth):
                    best_index = i
                    best_cycle_depth = cycle_depth+1
                    done = False
        if best_index == None:
            print "Cannot find an unplaced image with a connected neighbor"
            if len(group):
                # commit the previous group (if it exists)
                group_list.append(group)
                # and start a new group
                group = []
                best_cycle_depth = last_cycle_depth + 1
            else:
                best_cycle_depth = 0
            # now find an unplaced image that has the most connections
            # to other images (new cycle start)
            max_connections = None
            for i, image in enumerate(image_list):
                if image.cycle_depth < 0:
                    if (max_connections == None or image.connections > max_connections):
                        max_connections = image.connections
                        best_index = i
                        done = False
                        print " found image %d connections = %d" % (i, max_connections)
        if best_index != None:
            image = image_list[best_index]
            image.cycle_depth = best_cycle_depth
            last_cycle_depth = best_cycle_depth
            print "Adding %s (cycles = %d)" % (image.name, best_cycle_depth)
            group.append(image)

    print "Group (cycles) report:"
    for group in group_list:
        #if len(group) < 2:
        #    continue
        print "group (size=%d):" % (len(group)),
        for image in group:
            print "%s(%d)" % (image.name, image.cycle_depth),
        print ""

    return group_list


