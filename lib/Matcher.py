import copy
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np

from find_obj import filter_matches,explore_match
import ImageList
import transformations


class Matcher():
    def __init__(self):
        self.image_list = []
        self.matcher = None
        self.match_ratio = 0.75
        self.min_pairs = 2      # minimum number of pairs to consider a match
        #self.bf = cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True)
        #self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def configure(self, dparams={}, mparams={}):
        if 'detector' in dparams:
            if dparams['detector'] == 'SIFT':
                norm = cv2.NORM_L2
            elif dparams['detector'] == 'SURF':
                norm = cv2.NORM_L2
            elif dparams['detector'] == 'ORB':
                norm = cv2.NORM_HAMMING
            elif dparams['detector'] == 'Star':
                norm = cv2.NORM_HAMMING

        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        FLANN_INDEX_LSH    = 6
        if 'matcher' in mparams:
            if mparams['matcher'] == 'FLANN':
                if norm == cv2.NORM_L2:
                    flann_params = { 'algorithm': FLANN_INDEX_KDTREE,
                                     'trees': 5 }
                else:
                    flann_params = { 'algorithm': FLANN_INDEX_LSH,
                                     'table_number': 6,     # 12
                                     'key_size': 12,        # 20
                                     'multi_probe_level': 1 #2
                                     }
                self.matcher = cv2.FlannBasedMatcher(flann_params, {}) # bug : need to pass empty dict (#1329)
            elif mparams['matcher'] == 'BF':
                print "brute force norm = %d" % norm
                self.matcher = cv2.BFMatcher(norm)
        if 'match-ratio' in mparams:
            self.match_ratio = mparams['match-ratio']

    def setImageList(self, image_list):
        self.image_list = image_list

    def filter_by_feature(self, i1, i2, matches):
        kp1 = i1.kp_list
        kp2 = i2.kp_list
        mkp1, mkp2 = [], []
        idx_pairs = []
        used = np.zeros(len(kp2), np.bool_)
        for m in matches:
            if len(m) != 2:
                # we asked for the two best matches
                continue
            #print "dist = %.2f %.2f (%.2f)" % (m[0].distance, m[1].distance, m[0].distance/m[1].distance)
            if m[0].distance > m[1].distance * self.match_ratio:
                # must pass the feature vector distance ratio test
                continue

            m = m[0]
            if not used[m.trainIdx]:
                used[m.trainIdx] = True
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
                idx_pairs.append( [m.queryIdx, m.trainIdx] )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, kp_pairs, idx_pairs

    def filter_by_location(self, i1, i2, idx_pairs, dist):
        result = []
        for pair in idx_pairs:
            c1 = np.array(i1.coord_list[pair[0]])
            c2 = np.array(i2.coord_list[pair[1]])
            d = np.linalg.norm(c1 - c2)
            if d > dist:
                # must be in physical proximity
                continue
            result.append(pair)
        return result

    # Iterate through all the matches for the specified image and
    # delete keypoints that don't satisfy the homography (or
    # fundamental) relationship.  Returns true if match set is clean, false
    # if keypoints were removed.
    #
    # Notice: this tends to eliminate matches that aren't all on the
    # same plane, so if the scene has a lot of depth, this could knock
    # out a lot of good matches.
    def filter_by_homography(self, i1, i2, j, filter):
        clean = True
        
        tol = float(i1.width) / 800.0 # rejection range in pixels
        if tol < 1.0:
            tol = 1.0
        # print "tol = %.4f" % tol
        matches = i1.match_list[j]
        if len(matches) < self.min_pairs:
            i1.match_list[j] = []
            return True
        p1 = []
        p2 = []
        for k, pair in enumerate(matches):
            use_raw_uv = False
            if use_raw_uv:
                p1.append( i1.kp_list[pair[0]].pt )
                p2.append( i2.kp_list[pair[1]].pt )
            else:
                # undistorted uv points should be better, right?
                p1.append( i1.uv_list[pair[0]] )
                p2.append( i2.uv_list[pair[1]] )

        p1 = np.float32(p1)
        p2 = np.float32(p2)
        #print "p1 = %s" % str(p1)
        #print "p2 = %s" % str(p2)
        if filter == "homography":
            M, status = cv2.findHomography(p1, p2, cv2.RANSAC, tol)
        elif filter == "fundamental":
            M, status = cv2.findFundamentalMat(p1, p2, cv2.RANSAC, tol)
        elif filter == "none":
            status = np.ones(len(matches))
        else:
            # fail
            M, status = None, None
        print '%s vs %s: %d / %d  inliers/matched' \
            % (i1.name, i2.name, np.sum(status), len(status))
        # remove outliers
        for k, flag in enumerate(status):
            if not flag:
                print "    deleting: " + str(matches[k])
                clean = False
                matches[k] = (-1, -1)
        for pair in reversed(matches):
            if pair == (-1, -1):
                matches.remove(pair)
        return clean

    def filter_non_reciprocal_pair(self, image_list, i, j):
        clean = True
        i1 = image_list[i]
        i2 = image_list[j]
        #print "testing %i vs %i" % (i, j)
        matches = i1.match_list[j]
        rmatches = i2.match_list[i]
        before = len(matches)
        for k, pair in enumerate(matches):
            rpair = [pair[1], pair[0]]
            found = False
            for r in rmatches:
                #print "%s - %s" % (rpair, r)
                if rpair == r:
                    found = True
                    break
            if not found:
                #print "not found =", rpair
                matches[k] = [-1, -1]
        for pair in reversed(matches):
            if pair == [-1, -1]:
                matches.remove(pair)
        after = len(matches)
        if before != after:
            clean = False
            print "  (%d vs. %d) matches %d -> %d" % (i, j, before, after)
        return clean

    def filter_non_reciprocal(self, image_list):
        clean = True
        print "Removing non-reciprocal matches:"
        for i, i1 in enumerate(image_list):
            for j, i2 in enumerate(image_list):
                if not filter_non_reciprocal_pair(image_list, i, j):
                    clean = False
        return clean

    # Iterative Closest Point algorithm
    def ICP(self, i1, i2):
        if i1 == i2:
            return []
        # create a new copy of i2.coord_list
        coord_list2 = np.array(i2.coord_list, copy=True).T
        #print coord_list2

        # make homogeneous
        newcol = np.ones( (1, coord_list2.shape[1]) )
        #print newcol
        #print coord_list2.shape, newcol.shape
        coord_list2 = np.vstack((coord_list2, newcol))

        done = False
        while not done:
            # find a pairing for the closest points.  If the closest point
            # is already taken, then if the distance is less, the old
            # pairing is dropped and the new one accepted. If the distance
            # is greater than a previous pairing, the new pairing is
            # skipped
            i1.icp_index = np.zeros( len(i1.coord_list), dtype=int )
            i1.icp_index.fill(-1)
            i1.icp_dist = np.zeros( len(i1.coord_list) )
            i1.icp_dist.fill(np.inf) # a big number
            for i in range(coord_list2.shape[1]):
                c2 = coord_list2[:3,i]
                (dist, index) = i1.kdtree.query(c2, k=1)
                if dist < i1.icp_dist[index]:
                    i1.icp_dist[index] = dist
                    i1.icp_index[index] = i
            pairs = []
            for i, index in enumerate(i1.icp_index):
                if index >= 0:
                    c1 = i1.coord_list[i]
                    c2 = coord_list2[:3,index]
                    #print "c1=%s c2=%s" % (c1, c2)
                    pairs.append( [c1, c2] )

            do_plot = False
            if do_plot:
                # This can be plotted in gnuplot with:
                # plot "c1.txt", "c2.txt", "vector.txt" u 1:2:($3-$1):($4-$2) title "pairs" with vectors
                f = open('c1.txt', 'w')
                for c1 in i1.coord_list:
                    f.write("%.3f %.3f %.3f\n" % (c1[1], c1[0], -c1[2]))
                f.close()
                f = open('c2.txt', 'w')
                for i in range(coord_list2.shape[1]):
                    c2 = coord_list2[:3,i]
                    f.write("%.3f %.3f %.3f\n" % (c2[1], c2[0], -c2[2]))
                f.close()
                f = open('vector.txt', 'w')
                for pair in pairs:
                    c1 = pair[0]
                    c2 = pair[1]
                    f.write("%.3f %.3f %.3f %.3f %.3f %.3f\n" % ( c2[1], c2[0], -c2[2], c1[1], c1[0], -c1[2] ))
                f.close()

            # find the affine transform matrix that brings the paired
            # points together
            #print "icp pairs =", len(pairs)
            v0 = np.zeros( (3, len(pairs)) )
            v1 = np.zeros( (3, len(pairs)) )
            weights = np.zeros( len(pairs) )
            weights.fill(1.0)
            for i, pair in enumerate(pairs):
                v0[:,i] = pair[0]
                v1[:,i] = pair[1]
            #print "v0\n", v0
            #print "v1\n", v1
            #print "weights\n", weights
            M = transformations.affine_matrix_from_points(v1, v0, shear=False, scale=False)
            #M = transformations.affine_matrix_from_points_weighted(v0, v1, weights, shear=False, scale=False)
            #print M
            scale, shear, angles, trans, persp = transformations.decompose_matrix(M)
            #print "scale=", scale
            #print "shear=", shear
            #print "angles=", angles
            #print "trans=", trans
            #print "persp=", persp

            coord_list2 = np.dot(M, coord_list2)
            coord_list2 /= coord_list2[3]
            # print coord_list2

            rot = np.linalg.norm(angles)
            dist = np.linalg.norm(trans)
            print "rot=%.6f dist=%.6f" % (rot, dist)
            if rot < 0.001 and dist < 0.001:
                done = True
                a = raw_input("Press Enter to continue...")

    def basic_matches(self, i1, i2, des_list1, des_list2,
                      result1, result2, feature_fuzz):
        # all vs. all match between overlapping i1 keypoints and i2
        # keypoints (forward match)
        matches = self.matcher.knnMatch(np.array(des_list1),
                                        trainDescriptors=np.array(des_list2),
                                        k=2)
        # rewrite the query/train indices (from our query subset)
        # back to referencing the full set of keypoints
        for match in matches:
            for m in match:
                qi = m.queryIdx
                m.queryIdx = result1[qi]
                ti = m.trainIdx
                m.trainIdx = result2[ti]
        print "initial matches =", len(matches)
        
        # run the classic feature distance ratio test
        p1, p2, kp_pairs, idx_pairs = self.filter_by_feature(i1, i2, matches)
        print "after distance ratio test =", len(idx_pairs)

        do_direct_geo_individual_distance_test = False
        if do_direct_geo_individual_distance_test:
            # test each individual match pair for proximity to each
            # other (fine grain distance check)
            # print "  pairs (before location filter) =", len(idx_pairs)
            idx_pairs = self.filter_by_location(i1, i2, idx_pairs, feature_fuzz)
            print "after direct goereference distance test =", len(idx_pairs)

        if len(idx_pairs) < self.min_pairs:
            idx_pairs = []
        print "  pairs =", len(idx_pairs)
        return idx_pairs
    
    # do initial feature matching of specified image against every
    # image in the provided image list (except self)
    # fuzz units is meters
    def hybridImageMatches(self, i1, i2, image_fuzz=40, feature_fuzz=20, review=False):
        if i1 == i2:
            print "We shouldn't see this, but i1 == i2"
            return [], []
        if np.any(i1.des_list):
            size = len(i1.des_list)
        else:
            size = 0

        # all the points in image1 that are within range of image2
        if i1.kdtree != None:
            result1 = i1.kdtree.query_ball_point(i2.center,
                                                 i2.radius + image_fuzz)
        else:
            result1 = [], []
        if len(result1) <= 1:
            return [], []

        # all the points in image2 that are within range of image1
        if i2.kdtree != None:
            result2 = i2.kdtree.query_ball_point(i1.center,
                                                 i1.radius + image_fuzz)
        else:
            result2 = [], []
        if len(result2) <= 1:
            return [], []

        # build the subset lists for matching
        des_list1 = []
        for k in result1:
            des_list1.append(i1.des_list[k])
        des_list2 = []
        for k in result2:
            des_list2.append(i2.des_list[k])

        print "i1 vs. i2 features = %d vs %d" % (len(des_list1), len(des_list2))

        # all vs. all match between overlapping i1 keypoints and i2
        # keypoints (forward match)
        idx_pairs1 = self.basic_matches( i1, i2, des_list1, des_list2,
                                         result1, result2, feature_fuzz )
        
        # all vs. all match between overlapping i2 keypoints and i1
        # keypoints (reverse match)
        idx_pairs2 = self.basic_matches( i2, i1, des_list2, des_list1,
                                         result2, result1, feature_fuzz )

        plot_matches = True
        if plot_matches:
            self.plot_matches(i1, i2, idx_pairs1)
            self.plot_matches(i2, i1, idx_pairs2)
            
        if review:
            if len(idx_pairs1):
                status = self.showMatch(i1, i2, idx_pairs1)
                # remove deselected pairs
                for k, flag in enumerate(status):
                    if not flag:
                        print "    deleting: " + str(idx_pairs1[k])
                        idx_pairs1[k] = (-1, -1)
                for pair in reversed(idx_pairs1):
                    if pair == (-1, -1):
                        idx_pairs1.remove(pair)
                        
            if len(idx_pairs2):
                status = self.showMatch(i2, i1, idx_pairs2)
                # remove deselected pairs
                for k, flag in enumerate(status):
                    if not flag:
                        print "    deleting: " + str(idx_pairs2[k])
                        idx_pairs2[k] = (-1, -1)
                for pair in reversed(idx_pairs2):
                    if pair == (-1, -1):
                        idx_pairs2.remove(pair)

        return idx_pairs1, idx_pairs2

    def plot_matches(self, i1, i2, idx_pairs):
        # This can be plotted in gnuplot with:
        # plot "c1.txt", "c2.txt", "vector.txt" u 1:2:($3-$1):($4-$2) title "pairs" with vectors
        f = open('c1.txt', 'w')
        for c1 in i1.coord_list:
            f.write("%.3f %.3f %.3f\n" % (c1[1], c1[0], -c1[2]))
        f.close()
        f = open('c2.txt', 'w')
        for c2 in i2.coord_list:
            f.write("%.3f %.3f %.3f\n" % (c2[1], c2[0], -c2[2]))
        f.close()
        f = open('vector.txt', 'w')
        for pair in idx_pairs:
            c1 = i1.coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            f.write("%.3f %.3f %.3f %.3f %.3f %.3f\n" % ( c2[1], c2[0], -c2[2], c1[1], c1[0], -c1[2] ))
        f.close()
        # a = raw_input("Press Enter to continue...")
 

    def robustGroupMatches(self, image_list, filter="fundamental",
                           image_fuzz=40, feature_fuzz=20, review=False):
        n = len(image_list) - 1
        n_work = float(n*(n+1)/2)
        n_count = float(0)
        
        # find basic matches and filter by match ratio and ned
        # location
        for i, i1 in enumerate(image_list):
            if len(i1.match_list) == 0:
                i1.match_list = [[]] * len(image_list)

            for j, i2 in enumerate(image_list):
                if j <= i:
                    continue
                print "Matching %s vs %s" % (i1.name, i2.name)
                if len(i2.match_list) == 0:
                    # create if needed
                    i2.match_list = [[]] * len(image_list)
                i1.match_list[j], i2.match_list[i] \
                    = self.hybridImageMatches(i1, i2, image_fuzz, feature_fuzz,
                                              review)
                n_count += 1

                done = False
                while not done:
                    done = True
                    if not self.filter_non_reciprocal_pair(image_list, i, j):
                        done = False
                    if not self.filter_non_reciprocal_pair(image_list, j, i):
                        done = False
                    if not self.filter_by_homography(i1, i2, j, filter):
                        done = False
                    if not self.filter_by_homography(i2, i1, i, filter):
                        done = False
                        
                print "%.1f %% done" % ((n_count / n_work) * 100.0)

        # so nothing sneaks through
        self.cullShortMatches(image_list)

        # and save
        self.saveMatches(image_list)

        
###########################################################
###   STUFF BELOW HERE IS OLD AND POSSIBLY DEPRECATED   ###
###########################################################

    def safeAddPair(self, i1, i2, refpair):
        image1 = self.image_list[i1]
        image2 = self.image_list[i2]
        matches = image1.match_list[i2]
        hasit = False
        for pair in matches:
            if pair[0] == refpair[0] and pair[1] == refpair[1]:
                hasit = True
                # print " already exists: %s->%s (%d %d)" \
                #    % (image1.name, image2.name, refpair[0], refpair[1])
        if not hasit:
            print "Adding %s->%s (%d %d)" % (image1.name, image2.name, refpair[0], refpair[1])
            matches.append(refpair)

    # the matcher doesn't always find the same matches between a pair
    # of images in both directions.  This will make sure a found match
    # in one direction also exists in the reciprocol direction.
    def addInverseMatches(self):
        for i, i1 in enumerate(self.image_list):
            print "add inverse matches for %s" % i1.name
            for j, matches in enumerate(i1.match_list):
                if i == j:
                    continue
                i2 = self.image_list[j]
                for pair in matches:
                    inv_pair = (pair[1], pair[0])
                    self.safeAddPair(j, i, inv_pair)
        for i1 in self.image_list:
            i1.save_matches()

    # remove any match sets shorter than self.min_pair
    def cullShortMatches(self, image_list):
        for i, i1 in enumerate(image_list):
            print "Cull matches for %s" % i1.name
            for j, matches in enumerate(i1.match_list):
                if len(matches) < self.min_pairs:
                    i1.match_list[j] = []

    def saveMatches(self, image_list):
        for image in image_list:
            image.save_matches()

    def showMatch(self, i1, i2, idx_pairs, status=None):
        #print " -- idx_pairs = " + str(idx_pairs)
        kp_pairs = []
        for p in idx_pairs:
            kp1 = i1.kp_list[p[0]]
            kp2 = i2.kp_list[p[1]]
            kp_pairs.append( (kp1, kp2) )
        img1 = i1.load_gray()
        img2 = i2.load_gray()
        if status == None:
            status = np.ones(len(kp_pairs), np.bool_)
        h, w = img1.shape[:2]
        scale = 790.0/float(w)
        si1 = cv2.resize(img1, (0,0), fx=scale, fy=scale)
        si2 = cv2.resize(img2, (0,0), fx=scale, fy=scale)
        explore_match('find_obj', si1, si2, kp_pairs,
                      hscale=scale, wscale=scale, status=status)
        # status structure will be correct here and represent
        # in/outlier choices of user
        cv2.destroyAllWindows()

        # status is an array of booleans that parallels the pair array
        # and represents the users choice to keep or discard the
        # respective pairs.
        return status

    # pasted from stackoverflow.com ....
    def rotateAndScale(self, img, degreesCCW=30, scaleFactor=1.0 ):
        (oldY,oldX) = img.shape[:2]
        
        # rotate about center of image.
        M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW,
                                    scale=scaleFactor)

        # choose a new image size.
        newX, newY = oldX * scaleFactor, oldY * scaleFactor
        
        # include this if you want to prevent corners being cut off
        r = np.deg2rad(degreesCCW)
        newX, newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),
                      abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

        # the warpAffine function call, below, basically works like this:
        # 1. apply the M transformation on each pixel of the original image
        # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

        # So I will find the translation that moves the result to the
        # center of that region.
        (tx, ty) = ((newX-oldX)/2, (newY-oldY)/2)
        M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
        M[1,2] += ty

        rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
        return rotatedImg, M

    def copyKeyPoint(self, k):
        return cv2.KeyPoint(x=k.pt[0], y=k.pt[1],
                            _size=k.size, _angle=k.angle,
                            _response=k.response, _octave=k.octave,
                            _class_id=k.class_id)
    
    def showMatchOrient(self, i1, i2, idx_pairs, status=None, orient='none'):
        #print " -- idx_pairs = " + str(idx_pairs)
        img1 = i1.load_gray()
        img2 = i2.load_gray()

        print 'orient:', orient
        if orient == 'aircraft':
            yaw1 = i1.aircraft_pose['ypr'][0]
            yaw2 = i2.aircraft_pose['ypr'][0]
        elif orient == 'camera':
            yaw1 = i1.camera_pose['ypr'][0]
            yaw2 = i2.camera_pose['ypr'][0]
        elif orient == 'sba':
            yaw1 = i1.camera_pose_sba['ypr'][0]
            yaw2 = i2.camera_pose_sba['ypr'][0]
        else:
            yaw1 = 0.0
            yaw2 = 0.0
        print yaw1, yaw2
        h, w = img1.shape[:2]
        scale = 790.0/float(w)
        si1, M1 = self.rotateAndScale(img1, yaw1, scale)
        si2, M2 = self.rotateAndScale(img2, yaw2, scale)

        kp_pairs = []
        for p in idx_pairs:
            kp1 = self.copyKeyPoint(i1.kp_list[p[0]])
            p1 = M1.dot( np.hstack((kp1.pt, 1.0)) )[:2]
            kp1.pt = (p1[0], p1[1])
            kp2 = self.copyKeyPoint(i2.kp_list[p[1]])
            p2 = M2.dot( np.hstack((kp2.pt, 1.0)) )[:2]
            kp2.pt = (p2[0], p2[1])
            print p1, p2
            kp_pairs.append( (kp1, kp2) )
        if status == None:
            status = np.ones(len(kp_pairs), np.bool_)

        #explore_match('find_obj', si1, si2, kp_pairs,
        #              hscale=scale, wscale=scale, status=status)
        explore_match('find_obj', si1, si2, kp_pairs,
                      hscale=1.0, wscale=1.0, status=status)
        # status structure will be correct here and represent
        # in/outlier choices of user
        cv2.destroyAllWindows()

        # status is an array of booleans that parallels the pair array
        # and represents the users choice to keep or discard the
        # respective pairs.
        return status

    def showMatches(self, i1):
        for j, i2 in enumerate(self.image_list):
            #print str(i1.match_list[j])
            idx_pairs = i1.match_list[j]
            if len(idx_pairs) > self.min_pairs:
                print "Showing matches for image %s and %s" % (i1.name, i2.name)
                self.showMatch( i1, i2, idx_pairs )

    def showAllMatches(self):
        # O(n,n) compare
        for i, i1 in enumerate(self.image_list):
            self.showMatches(i1)

    # compute the error between a pair of images
    def imagePairError(self, i, alt_coord_list, j, match, emax=False):
        i1 = self.image_list[i]
        i2 = self.image_list[j]
        #print "%s %s" % (i1.name, i2.name)
        coord_list = i1.coord_list
        if alt_coord_list != None:
            coord_list = alt_coord_list
        emax_value = 0.0
        dist2_sum = 0.0
        error_sum = 0.0
        for pair in match:
            c1 = coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            dist2 = dx*dx + dy*dy
            dist2_sum += dist2
            error = math.sqrt(dist2)
            if emax_value < error:
                emax_value = error
        if emax:
            return emax_value
        else:
            return math.sqrt(dist2_sum / len(match))

    # considers only total distance between points (thanks to Knuth
    # and Welford)
    def imagePairVariance1(self, i, alt_coord_list, j, match):
        i1 = self.image_list[i]
        i2 = self.image_list[j]
        coord_list = i1.coord_list
        if alt_coord_list != None:
            coord_list = alt_coord_list
        mean = 0.0
        M2 = 0.0
        n = 0
        for pair in match:
            c1 = coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            n += 1
            x = math.sqrt(dx*dx + dy*dy)
            delta = x - mean
            mean += delta/n
            M2 = M2 + delta*(x - mean)

        if n < 2:
            return 0.0
 
        variance = M2/(n - 1)
        return variance

    # considers x, y errors separated (thanks to Knuth and Welford)
    def imagePairVariance2(self, i, alt_coord_list, j, match):
        i1 = self.image_list[i]
        i2 = self.image_list[j]
        coord_list = i1.coord_list
        if alt_coord_list != None:
            coord_list = alt_coord_list
        xsum = 0.0
        ysum = 0.0
        # pass 1, compute x and y means
        for pair in match:
            c1 = coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            xsum += dx
            ysum += dy
        xmean = xsum / len(match)
        ymean = ysum / len(match)

        # pass 2, compute average error in x and y
        xsum2 = 0.0
        ysum2 = 0.0
        for pair in match:
            c1 = coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            ex = xmean - dx
            ey = ymean - dy
            xsum2 += ex * ex
            ysum2 += ey * ey
        xerror = math.sqrt( xsum2 / len(match) )
        yerror = math.sqrt( ysum2 / len(match) )
        return xerror*xerror + yerror*yerror

    # Compute an error metric related to image placement among the
    # group.  If an alternate coordinate list is provided, that is
    # used to compute the error metric (useful for doing test fits.)
    # if max=True then return the maximum pair error, not the weighted
    # average error
    def imageError(self, i, alt_coord_list=None, method="average",
                   variance=False, max=False):
        if method == "average":
            variance = False
            emax = False
        elif method == "stddev":
            variance = True
            emax = False
        elif method == "max":
            variance = False
            emax = True

        i1 = self.image_list[i]
        emax_value = 0.0
        dist2_sum = 0.0
        var_sum = 0.0
        weight_sum = i1.weight  # give ourselves an appropriate weight
        i1.num_matches = 0
        for j, match in enumerate(i1.match_list):
            if len(match):
                i1.num_matches += 1
                i2 = self.image_list[j]
                #print "Matching %s vs %s " % (i1.name, i2.name)
                error = 0.0
                if variance:
                    var = self.imagePairVariance2(i, alt_coord_list, j,
                                                 match)
                    #print "  %s var = %.2f" % (i1.name, var)
                    var_sum += var * i2.weight
                else:
                    error = self.imagePairError(i, alt_coord_list, j,
                                                match, emax)
                    dist2_sum += error * error * i2.weight
                weight_sum += i2.weight
                if emax_value < error:
                    emax_value = error
        if emax:
            return emax_value
        elif variance:
            #print "  var_sum = %.2f  weight_sum = %.2f" % (var_sum, weight_sum)
            return math.sqrt(var_sum / weight_sum)
        else:
            return math.sqrt(dist2_sum / weight_sum)

    # delete the pair (and it's reciprocal if it can be found)
    # i = image1 index, j = image2 index
    def deletePair(self, i, j, pair):
        i1 = self.image_list[i]
        i2 = self.image_list[j]
        #print "%s v. %s" % (i1.name, i2.name)
        #print "i1 pairs before = %s" % str(i1.match_list[j])
        #print "pair = %s" % str(pair)
        i1.match_list[j].remove(pair)
        #print "i1 pairs after = %s" % str(i1.match_list[j])
        pair_rev = (pair[1], pair[0])
        #print "i2 pairs before = %s" % str(i2.match_list[i])
        i2.match_list[i].remove(pair_rev)
        #print "i2 pairs after = %s" % str(i2.match_list[i])

    # compute the error between a pair of images
    def pairErrorReport(self, i, alt_coord_list, j, minError):
        i1 = self.image_list[i]
        i2 = self.image_list[j]
        match = i1.match_list[j]
        print "pair error report %s v. %s (%d)" % (i1.name, i2.name, len(match))
        if len(match) == 0:
            return
        report_list = []
        coord_list = i1.coord_list
        if alt_coord_list != None:
            coord_list = alt_coord_list
        dist2_sum = 0.0
        for k, pair in enumerate(match):
            c1 = coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            dist2 = dx*dx + dy*dy
            dist2_sum += dist2
            error = math.sqrt(dist2)
            report_list.append( (error, k) )
        report_list = sorted(report_list,
                             key=lambda fields: fields[0],
                             reverse=True)

        # meta stats on error values
        error_avg = math.sqrt(dist2_sum / len(match))
        stddev_sum = 0.0
        for line in report_list:
            error = line[0]
            stddev_sum += (error_avg-error)*(error_avg-error)
        stddev = math.sqrt(stddev_sum / len(match))
        print "   error avg = %.2f stddev = %.2f" % (error_avg, stddev)

        # compute best estimation of valid vs. suspect pairs
        dirty = False
        if error_avg >= minError:
            dirty = True
        if minError < 0.1:
            dirty = True
        status = np.ones(len(match), np.bool_)
        for line in report_list:
            if line[0] > 50.0 or line[0] > (error_avg + 2*stddev):
                status[line[1]] = False
                dirty = True

        if dirty:
            status = self.showMatch(i1, i2, match, status)
            delete_list = []
            for k, flag in enumerate(status):
                if not flag:
                    print "    deleting: " + str(match[k])
                    #match[i] = (-1, -1)
                    delete_list.append(match[k])

        if False: # for line in report_list:
            print "    %.1f %s" % (line[0], str(match[line[1]]))
            if line[0] > 50.0 or line[0] > (error_avg + 5*stddev):
                # if positional error > 50m delete pair
                done = False
                while not done:
                    print "Found a suspect match: d)elete v)iew [o]k: ",
                    reply = find_getch()
                    print ""
                    if reply == 'd':
                        match[line[1]] = (-1, -1)
                        dirty = True
                        done = True;
                        print "    (deleted) " + str(match[line[1]])
                    elif reply == 'v':
                        self.showMatch(i1, i2, match, line[1])
                    else:
                        done = True

        if dirty:
            # update match list to remove the marked pairs
            #print "before = " + str(match)
            #for pair in reversed(match):
            #    if pair == (-1, -1):
            #        match.remove(pair)
            for pair in delete_list:
                self.deletePair(i, j, pair)
            #print "after = " + str(match)

    def findImageIndex(self, search):
        for i, image in enumerate(self.image_list):
            if search == image:
                return i
        return None

    # fuzz factor increases (decreases) the ransac tolerance and is in
    # pixel units so it makes sense to bump this up or down in integer
    # increments.
    def reviewFundamentalErrors(self, fuzz_factor=1.0, interactive=True):
        total_removed = 0

        # Test fundametal matrix constraint
        for i, i1 in enumerate(self.image_list):
            # rejection range in pixels
            tol = float(i1.width) / 800.0 + fuzz_factor
            print "tol = %.4f" % tol
            if tol < 0.0:
                tol = 0.0
            for j, matches in enumerate(i1.match_list):
                if i == j:
                    continue
                if len(matches) < self.min_pairs:
                    i1.match_list[j] = []
                    continue
                i2 = self.image_list[j]
                p1 = []
                p2 = []
                for k, pair in enumerate(matches):
                    p1.append( i1.kp_list[pair[0]].pt )
                    p2.append( i2.kp_list[pair[1]].pt )

                p1 = np.float32(p1)
                p2 = np.float32(p2)
                #print "p1 = %s" % str(p1)
                #print "p2 = %s" % str(p2)
                M, status = cv2.findFundamentalMat(p1, p2, cv2.RANSAC, tol)

                size = len(status)
                inliers = np.sum(status)
                print '%s vs %s: %d / %d  inliers/matched' \
                    % (i1.name, i2.name, inliers, size)

                if inliers < size:
                    total_removed += (size - inliers)
                    if interactive:
                        status = self.showMatch(i1, i2, matches, status)

                    delete_list = []
                    for k, flag in enumerate(status):
                        if not flag:
                            print "    deleting: " + str(matches[k])
                            #match[i] = (-1, -1)
                            delete_list.append(matches[k])

                    for pair in delete_list:
                        self.deletePair(i, j, pair)
        return total_removed

    # return true if point set is pretty close to linear
    def isLinear(self, points, threshold=20.0):
        x = []
        y = []
        for pt in points:
            x.append(pt[0])
            y.append(pt[1])
        z = np.polyfit(x, y, 1) 
        p = np.poly1d(z)
        sum = 0.0
        for pt in points:
            e = pt[1] - p(pt[0])
            sum += e*e
        return math.sqrt(sum) <= threshold

    # look for linear/degenerate match sets
    def reviewLinearSets(self, threshold=20.0):
        # Test fundametal matrix constraint
        for i, i1 in enumerate(self.image_list):
            for j, matches in enumerate(i1.match_list):
                if i == j:
                    continue
                if len(matches) < self.min_pairs:
                    i1.match_list[j] = []
                    continue
                i2 = self.image_list[j]
                pts = []
                status = []
                for k, pair in enumerate(matches):
                    pts.append( i1.kp_list[pair[0]].pt )
                    status.append(False)

                # check for degenerate case of all matches being
                # pretty close to a straight line
                if self.isLinear(pts, threshold):
                    print "%s vs %s is a linear match, probably should discard" % (i1.name, i2.name)
                    
                    status = self.showMatch(i1, i2, matches, status)

                    delete_list = []
                    for k, flag in enumerate(status):
                        if not flag:
                            print "    deleting: " + str(matches[k])
                            #match[i] = (-1, -1)
                            delete_list.append(matches[k])

                    for pair in delete_list:
                        self.deletePair(i, j, pair)

    # Review matches by fewest pairs of keypoints.  The fewer
    # keypoints that join a pair of images, the greater the chance
    # that we could have stumbled on a degenerate or incorrect set.
    def reviewByFewestPairs(self, maxpairs=8):
        print "Review matches by fewest number of pairs"
        if len(self.image_list):
            report_list = []
            for i, image in enumerate(self.image_list):
                for j, pairs in enumerate(image.match_list):
                    e = len(pairs)
                    if e > 0 and e <= maxpairs:
                        report_list.append( (e, i, j) )
            report_list = sorted(report_list, key=lambda fields: fields[0],
                                 reverse=False)
            # show images sorted by largest positional disagreement first
            for line in report_list:
                i1 = self.image_list[line[1]]
                i2 = self.image_list[line[2]]
                pairs = i1.match_list[line[2]]
                if len(pairs) == 0:
                    # probably already deleted by the other match order
                    continue
                print "showing %s vs %s: %d pairs" \
                    % (i1.name, i2.name, len(pairs))
                status = self.showMatch(i1, i2, pairs)
                delete_list = []
                for k, flag in enumerate(status):
                    if not flag:
                        print "    deleting: " + str(pairs[k])
                        #match[i] = (-1, -1)
                        delete_list.append(pairs[k])
                for pair in delete_list:
                    self.deletePair(line[1], line[2], pair)

    # sort and review match pairs by worst positional error
    def matchErrorReport(self, i, minError=20.0):
        i1 = self.image_list[i]
        # now for each image, find/show worst individual matches
        report_list = []
        for j, pairs in enumerate(i1.match_list):
            if len(pairs):
                i2 = self.image_list[j]
                #print "Matching %s vs %s " % (i1.name, i2.name)
                e = self.imagePairError(i, None, j, pairs)
                if e > minError:
                    report_list.append( (e, i, j) )

        report_list = sorted(report_list,
                             key=lambda fields: fields[0],
                             reverse=True)

        for line in report_list:
            i1 = self.image_list[line[1]]
            i2 = self.image_list[line[2]]
            print "min error = %.3f" % minError
            print "%.1f %s %s" % (line[0], i1.name, i2.name)
            if line[0] > minError:
                #print "  %s" % str(pairs)
                self.pairErrorReport(line[1], None, line[2], minError)
                #print "  after %s" % str(match)
                #self.showMatch(i1, i2, match)

    # sort and review all match pairs by worst positional error
    def fullMatchErrorReport(self, minError=20.0):
        report_list = []
        for i, image in enumerate(self.image_list):
            i1 = self.image_list[i]
            # now for each image, find/show worst individual matches
            for j, pairs in enumerate(i1.match_list):
                if len(pairs):
                    i2 = self.image_list[j]
                    #print "Matching %s vs %s " % (i1.name, i2.name)
                    e = self.imagePairError(i, None, j, pairs)
                    if e > minError:
                        report_list.append( (e, i, j) )

        report_list = sorted(report_list,
                             key=lambda fields: fields[0],
                             reverse=True)

        for line in report_list:
            i1 = self.image_list[line[1]]
            i2 = self.image_list[line[2]]
            print "min error = %.3f" % minError
            print "%.1f %s %s" % (line[0], i1.name, i2.name)
            if line[0] > minError:
                #print "  %s" % str(pairs)
                self.pairErrorReport(line[1], None, line[2], minError)
                #print "  after %s" % str(match)
                #self.showMatch(i1, i2, match)

    def reviewPoint(self, lon_deg, lat_deg, ref_lon, ref_lat):
        (x, y) = ImageList.wgs842cart(lon_deg, lat_deg, ref_lon, ref_lat)
        print "Review images touching %.2f %.2f" % (x, y)
        review_list = ImageList.getImagesCoveringPoint(self.image_list, x, y, pad=25.0, only_placed=False)
        #print "  Images = %s" % str(review_list)
        for image in review_list:
            print "    %s -> " % image.name,
            for j, pairs in enumerate(image.match_list):
                if len(pairs):
                    print "%s (%d) " % (self.image_list[j].name, len(pairs)),
            print
            r2 = image.coverage()
            p = ImageList.getImagesCoveringRectangle(self.image_list, r2)
            p_names = []
            for i in p:
                p_names.append(i.name)
            print "      possible matches: %d" % len(p_names)

            
# the following functions do not have class dependencies but can live
# here for functional grouping.

def buildConnectionDetail(image_list, matches_direct):
    # wipe any existing connection detail
    for image in image_list:
        image.connection_detail = [0] * len(image_list)
    for match in matches_direct:
        # record all v. all connections
        for p in match[1:]:
            for q in match[1:]:
                i1 = p[0]
                i2 = q[0]
                if i1 != i2:
                    image_list[i1].connection_detail[i2] += 1
                    
def reportConnectionDetail():
    print "Connection detail report"
    print "(will add in extra 3+ way matches to the count when they exist.)"
    for image in image_list:
        print image.name
        for i, count in enumerate(image.connection_detail):
            if count > 0:
                print "  ", image_list[i].name, count

# return the neighbor that is closest to the root node of the
# placement tree (i.e. smallest cycle_depth.
def bestNeighbor(image, image_list):
    best_cycle_depth = len(image_list) + 1
    best_index = None
    for i, pairs in enumerate(image.match_list):
        if len(pairs):
            i2 = image_list[i]
            dist = i2.cycle_depth
            #print "  neighbor check %d = %d" % ( i, dist )
            if dist >= 0 and dist < best_cycle_depth:
                best_cycle_depth = dist
                best_index = i
    return best_index, best_cycle_depth

def groupByConnections(image_list, matches_direct):
    # reset the cycle distance for all images
    for image in image_list:
        image.cycle_depth = -1
        
    # compute number of connections per image
    buildConnectionDetail(image_list, matches_direct)
    for image in image_list:
        image.connections = 0
        for pair_count in image.connection_detail:
            if pair_count >= 8:
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
            #print "Cannot find an unplaced image with a connected neighbor"
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
                        #print " found image %d connections = %d" % (i, max_connections)
        if best_index != None:
            image = image_list[best_index]
            image.cycle_depth = best_cycle_depth
            last_cycle_depth = best_cycle_depth
            #print "Adding %s (cycles = %d)" % (image.name, best_cycle_depth)
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

