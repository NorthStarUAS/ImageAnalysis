import cv2
import math
import numpy as np

class Placer():
    def __init__(self):
        self.image_list = []

    def setImageList(self, image_list):
        self.image_list = image_list

    # find affine transform between matching i1, i2 keypoints in map
    # space.  fullAffine=True means unconstrained to include best
    # warp/shear.  fullAffine=False means limit the matrix to only
    # best rotation, translation, and scale.
    def findAffine(self, i1, i2, pairs, fullAffine=False):
        src = []
        dst = []
        for pair in pairs:
            c1 = i1.coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            src.append( c1 )
            dst.append( c2 )
        #print "src = %s" % str(src)
        #print "dst = %s" % str(dst)
        affine = cv2.estimateRigidTransform(np.array([src]).astype(np.float32),
                                            np.array([dst]).astype(np.float32),
                                            fullAffine)
        #print str(affine)
        return affine

    def decomposeAffine(self, affine):
        tx = affine[0][2]
        ty = affine[1][2]

        a = affine[0][0]
        b = affine[0][1]
        c = affine[1][0]
        d = affine[1][1]

        sx = math.sqrt( a*a + b*b )
        if a < 0.0:
            sx = -sx
        sy = math.sqrt( c*c + d*d )
        if d < 0.0:
            sy = -sy

        rotate_deg = math.atan2(-b,a) * 180.0/math.pi
        if rotate_deg < -180.0:
            rotate_deg += 360.0
        if rotate_deg > 180.0:
            rotate_deg -= 360.0
        return (rotate_deg, tx, ty, sx, sy)

    def findWeightedAffine1(self, i1, fullAffine=False):
        # 1. find the affine transform for individual image pairs
        # 2. decompose the affine matrix into scale, rotation, translation
        # 3. weight the decomposed values
        # 4. assemble a final 'weighted' affine matrix from the
        #    weighted average of the decomposed elements

        # initialize sums with the match against ourselves
        sx_sum = 0.0
        sy_sum = 0.0
        tx_sum = 0.0
        ty_sum = 0.0
        rotate_sum = 0.0        # assume rotations are small and near zero
        weight_sum = 0.0
        for i, pairs in enumerate(i1.match_list):
            if len(pairs) < 3:
                # can't compute affine on < 3 points
                continue
            i2 = self.image_list[i]
            if not i2.placed:
                # only match against previously placed images
                continue

            #weight = i2.weight
            weight = i2.connections

            #print "Affine %s vs %s, pairs = %s" % (i1.name, i2.name, str(pairs))
            affine = self.findAffine(i1, i2, pairs, fullAffine=fullAffine)
            if affine == None:
                # it's possible given a degenerate point set, the
                # affine estimator will return None
                continue
            (rotate_deg, tx, ty, sx, sy) = self.decomposeAffine(affine)

            # update sums
            sx_sum += sx * weight
            sy_sum += sy * weight
            tx_sum += tx * weight
            ty_sum += ty * weight
            rotate_sum += rotate_deg * weight
            weight_sum += weight

            #print "  shift = %.2f %.2f" % (tx, ty)
            #print "  scale = %.2f %.2f" % (sx, sy)
            #print "  rotate = %.2f" % (rotate_deg)
            #self.showMatch(i1, i2, pairs)
        if weight_sum > 0.00001:
            new_sx = sx_sum / weight_sum
            new_sy = sy_sum / weight_sum
            new_tx = tx_sum / weight_sum
            new_ty = ty_sum / weight_sum
            new_rot = rotate_sum / weight_sum
        else:
            new_sx = 1.0
            new_sy = 1.0
            new_tx = 0.0
            new_ty = 0.0
            new_rot = 0.0

        # compose a new 'weighted' affine matrix
        rot_rad = new_rot * math.pi / 180.0
        costhe = math.cos(rot_rad)
        sinthe = math.sin(rot_rad)
        row1 = [ new_sx * costhe, -new_sx * sinthe, new_tx ]
        row2 = [ new_sy * sinthe, new_sy * costhe, new_ty ]
        avg_affine = np.array( [ row1, row2 ] )
        #print str(avg_affine)
        #print " image shift = %.2f %.2f" % (new_tx, new_ty)
        #print " image rotate = %.2f" % (new_rot)
        return avg_affine

    def findImageWeightedAffine2(self, i1, fullAffine=False):
        # 1. find the affine transform for individual image pairs
        # 2. find the weighted average of the affine transform matrices

        # initialize sums with the match against ourselves
        affine_sum = np.array( [ [1.0, 0.0, 0.0 ], [0.0, 1.0, 0.0] ] )
        affine_sum *= i1.weight
        weight_sum = i1.weight  # our own weight
        for i, pairs in enumerate(i1.match_list):
            if len(pairs) < 3:
                # can't compute affine on < 3 points
                continue
            i2 = self.image_list[i]
            #print "Affine %s vs %s" % (i1.name, i2.name)
            affine = self.findAffine(i1, i2, pairs, fullAffine)
            if affine == None:
                # it's possible given a degenerate point set, the
                # affine estimator will return None
                continue
            affine_sum += affine * i2.weight
            weight_sum += i2.weight
            #self.showMatch(i1, i2, pairs)
        # weight_sum should always be greater than zero
        i1.newM = affine_sum / weight_sum
        #print str(i1.newM)
    
    # compare only against 'placed' images and do not weight ourselves
    def findImageWeightedAffine3(self, i1, fullAffine=False):
        # 1. find the affine transform for individual image pairs
        # 2. find the weighted average of the affine transform matrices

        # initialize sums with the match against ourselves
        affine_sum = np.array( [ [0.0, 0.0, 0.0 ], [0.0, 0.0, 0.0] ] )
        weight_sum = 0.0
        for i, pairs in enumerate(i1.match_list):
            if len(pairs) < 3:
                # can't compute affine on < 3 points
                continue
            i2 = self.image_list[i]
            if not i2.placed:
                continue
            #print "Affine %s vs %s" % (i1.name, i2.name)
            affine = self.findAffine(i1, i2, pairs, fullAffine)
            if affine == None:
                # it's possible given a degenerate point set, the
                # affine estimator will return None
                continue
            affine_sum += affine * i2.weight
            weight_sum += i2.weight
            #self.showMatch(i1, i2, pairs)
        # weight_sum should always be greater than zero
        if weight_sum > 0.00001:
            result = affine_sum / weight_sum
        else:
            result = np.array( [ [1.0, 0.0, 0.0 ], [0.0, 1.0, 0.0] ] )
        return result
    
    # find homography transform matrix between matching i1, i2
    # keypoints in map space.
    def findHomography(self, i1, i2, pairs):
        src = []
        dst = []
        for pair in pairs:
            c1 = i1.coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            src.append( c1 )
            dst.append( c2 )
        #H, status = cv2.findHomography(np.array([src]).astype(np.float32),
        #                               np.array([dst]).astype(np.float32),
        #                               cv2.RANSAC, 5.0)
        H, status = cv2.findHomography(np.array([src]).astype(np.float32),
                                       np.array([dst]).astype(np.float32))
        #print str(affine)
        return H

    # compare against best 'placed' image (averaging transform
    # matrices together directly doesn't do what we want)
    def findGroupAffine(self, i1, fullAffine=False):
        # find the affine transform matrix representing the best fit
        # against all the placed neighbors.  Builds a cumulative
        # src/dest list with our src points listed once for each image
        # pair.

        src = []
        dst = []
        for i, pairs in enumerate(i1.match_list):
            if len(pairs) < 3:
                # can't compute affine transform on < 3 points
                continue
            i2 = self.image_list[i]
            if not i2.placed:
                # don't consider non-yet-placed neighbors
                continue
            # add coordinate matches for this image pair
            for pair in pairs:
                c1 = i1.coord_list[pair[0]]
                c2 = i2.coord_list[pair[1]]
                src.append( c1 )
                dst.append( c2 )

        if len(src) < 3:
            # not enough points to compute affine transformation
            return np.array( [ [1.0, 0.0, 0.0 ], [0.0, 1.0, 0.0] ] )

        # find the affine matrix on the communlative set of all
        # matching coordinates for all matching image pairs
        # simultaneously...
        affine = cv2.estimateRigidTransform(np.array([src]).astype(np.float32),
                                            np.array([dst]).astype(np.float32),
                                            fullAffine)
        if affine == None:
            # it's possible given a degenerate point set, the affine
            # estimator will return None, so return the identity
            affine = np.array( [ [1.0, 0.0, 0.0 ], [0.0, 1.0, 0.0] ] )
        return affine
    
    # compare against best 'placed' image (averaging transform
    # matrices together directly doesn't do what we want)
    def findGroupHomography(self, i1):
        # find the homography matrix representing the best fit against
        # all the placed neighbors.  Builds a cumulative src/dest list
        # with our src points listed once for each image pair.

        src = []
        dst = []
        for i, pairs in enumerate(i1.match_list):
            if len(pairs) < 4:
                # can't compute homography on < 4 points
                continue
            i2 = self.image_list[i]
            if not i2.placed:
                # don't consider non-yet-placed neighbors
                continue
            # add coordinate matches for this image pair
            for pair in pairs:
                c1 = i1.coord_list[pair[0]]
                c2 = i2.coord_list[pair[1]]
                src.append( c1 )
                dst.append( c2 )
        if len(src) < 4:
            # no placed neighbors, just return the identity matrix
            return np.identity(3)
        # find the homography matrix on the communlative set of all
        # matching coordinates for all matching image pairs
        # simultaneously...
        H, status = cv2.findHomography(np.array([src]).astype(np.float32),
                                       np.array([dst]).astype(np.float32),
                                       cv2.RANSAC, 5.0)
        if H == None:
            # it's possible given a degenerate point set, the
            # homography estimator will return None
            return np.identity(3)
        return H
    
    # compare against best 'placed' image (averaging transform
    # matrices together directly doesn't do what we want)
    def findImageHomography2(self, i1):
        # find the homography matrix for best (most connected) already
        # placed neighbor

        best_index = 0
        best_pairs = 0
        for i, pairs in enumerate(i1.match_list):
            if len(pairs) < 4:
                # can't compute homography on < 4 points
                continue
            i2 = self.image_list[i]
            if not i2.placed:
                continue
            if len(pairs) > best_pairs:
                best_pairs = len(pairs)
                best_index = i
        if best_pairs == 0:
            return np.identity(3)
        i2 = self.image_list[best_index]
        #print "Affine %s vs %s" % (i1.name, i2.name)
        H = self.findHomography(i1, i2, i1.match_list[best_index])
        if H == None:
            # it's possible given a degenerate point set, the
            # affine estimator will return None
            return np.identity(3)
        return H
    
    def transformImage(self, image, gain=1.0, M=None):
        if M == None:
            M = image.newM

        # print "Transforming " + str(image.name)
        for i, coord in enumerate(image.coord_list):
            if not image.kp_usage[i]:
                continue
            newcoord = M.dot([coord[0], coord[1], 1.0])
            dx = (newcoord[0] - coord[0]) * gain
            dy = (newcoord[1] - coord[1]) * gain
            image.coord_list[i][0] += dx
            image.coord_list[i][1] += dy
            # print "old %s -> new %s" % (str(coord), str(newcoord))
        for i, coord in enumerate(image.corner_list):
            newcoord = M.dot([coord[0], coord[1], 1.0])
            dx = (newcoord[0] - coord[0]) * gain
            dy = (newcoord[1] - coord[1]) * gain
            image.corner_list[i][0] += dx
            image.corner_list[i][1] += dy
            # print "old %s -> new %s" % (str(coord), str(newcoord))
        for i, coord in enumerate(image.grid_list):
            newcoord = M.dot([coord[0], coord[1], 1.0])
            dx = (newcoord[0] - coord[0]) * gain
            dy = (newcoord[1] - coord[1]) * gain
            image.grid_list[i][0] += dx
            image.grid_list[i][1] += dy
            # print "old %s -> new %s" % (str(coord), str(newcoord))

    def affineTransformImages(self, gain=0.1, fullAffine=False):
        for image in self.image_list:
            self.findImageWeightedAffine2(image, fullAffine=fullAffine)
        for image in self.image_list:
            self.transformImage(image, gain)

    # return true if this image has a neighbor that is already been placed
    def hasPlacedNeighbor(self, image):
        for i, pairs in enumerate(image.match_list):
             if len(pairs):
                 i2 = self.image_list[i]
                 if i2.placed:
                     return True
        return False
        
    def groupByConnections(self, image_list=None, affine=""):
        if image_list == None:
            image_list = self.image_list

        # reset the placed flag
        for image in image_list:
            image.placed = False
        self.group_list = []
        group = []
        done = False
        while not done:
            done = True
            maxcon = None
            maxidx = None
            # find an unplaced image with a placed neighbor that has
            # the most connections to other images
            for i, image in enumerate(image_list):
                if not image.placed and self.hasPlacedNeighbor(image) and (maxcon == None or image.connections > maxcon):
                    maxcon = image.connections
                    maxidx = i
                    done = False
            if maxidx == None:
                if len(group):
                    # commit the previous group (if it exists)
                    self.group_list.append(group)
                    # and start a new group
                    group = []
                # now find an unplaced image that has the most connections
                # to other images
                for i, image in enumerate(image_list):
                    if not image.placed and (maxcon == None or image.connections > maxcon):
                        maxcon = image.connections
                        maxidx = i
                        done = False
            if maxidx != None:
                image = image_list[maxidx]
                #print "Adding %s (connections = %d)" % (image.name, maxcon)
                image.placed = True
                group.append(image)

        print "Group (cycles) report:"
        for group in self.group_list:
            print "  group:",
            for image in group:
                print " %s" % image.name,
            print ""

        return self.group_list

    def placeImagesByConnections(self, image_list=None, affine=""):
        if image_list == None:
            image_list = self.image_list

        # reset the placed flag
        for image in image_list:
            image.placed = False
        placed_list = []
        done = False
        while not done:
            done = True
            maxcon = None
            maxidx = None
            # find an unplaced image with a placed neighbor that has
            # the most connections to other images
            for i, image in enumerate(image_list):
                if not image.placed and self.hasPlacedNeighbor(image) and (maxcon == None or image.connections > maxcon):
                    maxcon = image.connections
                    maxidx = i
                    done = False
            if maxidx == None:
                # find an unplaced image that has the most connections
                # to other images
                for i, image in enumerate(image_list):
                    if not image.placed and (maxcon == None or image.connections > maxcon):
                        maxcon = image.connections
                        maxidx = i
                        done = False
            if maxidx != None:
                image = image_list[maxidx]
                print "Placing %s (connections = %d)" % (image.name, maxcon)
                if affine == "rigid" or affine == "full":
                    fullAffine = (affine == "full")
                    #M = self.findGroupAffine(image, fullAffine=fullAffine)
                    #M = self.findGroupHomography(image)
                    M = self.findWeightedAffine1(image, fullAffine=fullAffine)
                    self.transformImage(image, gain=1.0, M=M)
                image.placed = True
                placed_list.append(image)
        return placed_list

    def getImageCenter(self, image):
        x_sum = 0.0
        y_sum = 0.0
        for pt in image.corner_list:
            x_sum += pt[0]
            y_sum += pt[1]
        count = float(len(image.corner_list))
        return (x_sum/count, y_sum/count)

    def placeImagesByScore(self, image_list=None, affine=""):
        if image_list == None:
            image_list = self.image_list

        # reset the placed flag
        for image in image_list:
            image.placed = False
        placed_list = []
        done = False
        while not done:
            done = True
            minscore = None
            minidx = None

            # score = distance from center / connections, lowest score wins
            
            # find the unplaced image with the lowest score
            for i, image in enumerate(image_list):
                if not image.placed and self.hasPlacedNeighbor(image):
                    (dx, dy) = self.getImageCenter(image)
                    score = math.sqrt(dx*dx + dy*dy)
                    if image.connections == 0:
                        # don't place unconnected images
                        continue
                    # more connections = lower score, but what should
                    # the exact formula be?
                    #score -= image.connections
                    if minscore == None or score < minscore:
                        minscore = score
                        minidx = i
                        done = False
            if minidx == None:
                # couldn't find an unplaced image with placed
                # neighbors, jump to the next group
                for i, image in enumerate(image_list):
                    if not image.placed:
                        (dx, dy) = self.getImageCenter(image)
                        score = math.sqrt(dx*dx + dy*dy)
                        if image.connections == 0:
                            # don't place unconnected images
                            continue
                        # more connections = lower score, but what should
                        # the exact formula be?
                        #score -= image.connections
                        if minscore == None or score < minscore:
                            minscore = score
                            minidx = i
                            done = False
            if minidx != None:
                image = image_list[minidx]
                print "Placing %s (score = %.3f)" % (image.name, minscore)
                if affine == "rigid" or affine == "full":
                    fullAffine = (affine == "full")
                    #M = self.findGroupAffine(image, fullAffine=fullAffine)
                    #M = self.findGroupHomography(image)
                    M = self.findWeightedAffine1(image, fullAffine=fullAffine)
                    self.transformImage(image, gain=1.0, M=M)
                image.placed = True
                placed_list.append(image)
        return placed_list

