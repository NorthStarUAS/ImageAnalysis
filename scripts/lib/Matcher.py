#!/usr/bin/python3

from __future__ import print_function

# put most of our eggs in the gms matching basket:
# https://github.com/JiawangBian/GMS-Feature-Matcher/blob/master/python/gms_matcher.py

import copy
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import time
from tqdm import tqdm

from props import getNode

from .find_obj import filter_matches,explore_match
from . import ImageList
from . import transformations

class Matcher():
    def __init__(self):
        self.detector_node = getNode('/config/detector', True)
        self.matcher_node = getNode('/config/matcher', True)
        self.image_list = []
        self.matcher = None
        self.match_ratio = 0.70
        self.min_pairs = 25
        # probably cleaner ways to do this...
        self.camera_node = getNode('/config/camera', True)

    def configure(self):
        detector_str = self.detector_node.getString('detector')
        if detector_str == 'SIFT':
            norm = cv2.NORM_L2
            self.max_distance = 270.0
        elif detector_str == 'SURF':
            norm = cv2.NORM_L2
            self.max_distance = 270.0
        elif detector_str == 'ORB':
            norm = cv2.NORM_HAMMING
            self.max_distance = 64
        elif detector_str == 'Star':
            norm = cv2.NORM_HAMMING
            self.max_distance = 64
        else:
            print('No detector defined???')
            quit()

        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        FLANN_INDEX_LSH    = 6
        matcher_str = self.matcher_node.getString('matcher')
        if matcher_str == 'FLANN':
            if norm == cv2.NORM_L2:
                flann_params = {
                    'algorithm': FLANN_INDEX_KDTREE,
                    'trees': 5
                }
            else:
                flann_params = {
                    'algorithm': FLANN_INDEX_LSH,
                    'table_number': 6,     # 12
                    'key_size': 12,        # 20
                    'multi_probe_level': 1 #2
                }
            self.matcher = cv2.FlannBasedMatcher(flann_params, {}) # bug : need to pass empty dict (#1329)
        elif matcher_str == 'BF':
            print("brute force norm = %d" % norm)
            self.matcher = cv2.BFMatcher(norm)
        self.match_ratio = self.matcher_node.getFloat('match_ratio')
        self.min_pairs = self.matcher_node.getFloat('min_pairs')

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

    # SIFT (for example) can detect the same feature at different
    # scales which can lead to duplicate match pairs, or possibly one
    # feature in image1 matching two or more features in images2.
    # Find and filter these out of the set.
    def filter_duplicates(self, i1, i2, idx_pairs):
        count = 0
        result = []
        kp1_dict = {}
        kp2_dict = {}
        for pair in idx_pairs:
            kp1 = i1.kp_list[pair[0]]
            kp2 = i2.kp_list[pair[1]]
            key1 = "%.2f-%.2f" % (kp1.pt[0], kp1.pt[1])
            key2 = "%.2f-%.2f" % (kp2.pt[0], kp2.pt[1])
            if key1 in kp1_dict and key2 in kp2_dict:
                # print("image1 and image2 key point already used:", key1, key2)
                count += 1
            elif key1 in kp1_dict:
                # print("image1 key point already used:", key1)
                count += 1
            elif key2 in kp2_dict:
                # print( "image2 key point already used:", key2)
                count += 1
            else:
                kp1_dict[key1] = True
                kp2_dict[key2] = True
                result.append(pair)
        if count > 0:
            print("  removed %d duplicate features" % count)
        return result

    # Iterate through all the matches for the specified image and
    # delete keypoints that don't satisfy the homography (or
    # fundamental) relationship.  Returns true if match set is clean, false
    # if keypoints were removed.
    #
    # Notice: this tends to eliminate matches that aren't all on the
    # same plane, so if the scene has a lot of depth, this could knock
    # out a lot of good matches.
    def filter_by_homography(self, K, i1, i2, filter):
        clean = True
        
        # tol = float(i1.width) / 200.0 # rejection range in pixels
        tol = math.pow(i1.width, 0.25)
        if tol < 1.0:
            tol = 1.0
        # print "tol = %.4f" % tol 
        matches = i1.match_list[i2.name]
        if len(matches) < self.min_pairs:
            i1.match_list[i2.name] = []
            return True
        p1 = []
        p2 = []
        for k, pair in enumerate(matches):
            use_raw_uv = False
            if use_raw_uv:
                p1.append( i1.kp_list[pair[0]].pt )
                p2.append( i2.kp_list[pair[1]].pt )
            else:
                # undistorted uv points should be better if the camera
                # calibration is known, right?
                p1.append( i1.uv_list[pair[0]] )
                p2.append( i2.uv_list[pair[1]] )

        p1 = np.float32(p1)
        p2 = np.float32(p2)
        #print "p1 = %s" % str(p1)
        #print "p2 = %s" % str(p2)
        if filter == "homography":
            #method = cv2.RANSAC
            method = cv2.LMEDS
            M, status = cv2.findHomography(p1, p2, method, tol)
        elif filter == "fundamental":
            method = cv2.FM_RANSAC # more stable
            #method = cv2.FM_LMEDS # keeps dropping more points
            M, status = cv2.findFundamentalMat(p1, p2, method, tol)
        elif filter == "essential":
            # method = cv2.FM_RANSAC     
            method = cv2.FM_LMEDS
            M, status = cv2.findEssentialMat(p1, p2, K, method, threshold=tol)
        elif filter == "none":
            status = np.ones(len(matches))
        else:
            # fail
            M, status = None, None
        print('  %s vs %s: %d / %d  inliers/matched' \
            % (i1.name, i2.name, np.sum(status), len(status)))
        # remove outliers
        for k, flag in enumerate(status):
            if not flag:
                print("    deleting: " + str(matches[k]))
                clean = False
                matches[k] = (-1, -1)
        for pair in reversed(matches):
            if pair == (-1, -1):
                matches.remove(pair)
        return clean

    # iterate through idx_pairs1 and mark/remove any pairs that don't
    # exist in idx_pairs2.  Then recreate idx_pairs2 as the inverse of
    # idx_pairs1
    def filter_cross_check(self, idx_pairs1, idx_pairs2):
        new1 = []
        new2 = []
        for k, pair in enumerate(idx_pairs1):
            rpair = [pair[1], pair[0]]
            for r in idx_pairs2:
                #print "%s - %s" % (rpair, r)
                if rpair == r:
                    new1.append( pair )
                    new2.append( rpair )
                    break
        if len(idx_pairs1) != len(new1) or len(idx_pairs2) != len(new2):
            print("  cross check: (%d, %d) => (%d, %d)" % (len(idx_pairs1), len(idx_pairs2), len(new1), len(new2)))
        return new1, new2                                               
            
    def filter_non_reciprocal_pair(self, image_list, i, j):
        clean = True
        i1 = image_list[i]
        i2 = image_list[j]
        #print "testing %i vs %i" % (i, j)
        matches = i1.match_list[i2.name]
        rmatches = i2.match_list[i1.name]
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
            print("  (%d vs. %d) matches %d -> %d" % (i, j, before, after))
        return clean

    def filter_non_reciprocal(self, image_list):
        clean = True
        print("Removing non-reciprocal matches:")
        for i, i1 in enumerate(image_list):
            for j, i2 in enumerate(image_list):
                if not self.filter_non_reciprocal_pair(image_list, i, j):
                    clean = False
        return clean

    def basic_matches(self, i1, i2):
        # all vs. all match between overlapping i1 keypoints and i2
        # keypoints (forward match)
        
        # sanity check
        if i1.des_list is None or i2.des_list is None:
            return []
        if len(i1.des_list.shape) == 0 or i1.des_list.shape[0] <= 1:
            return []
        if len(i2.des_list.shape) == 0 or i2.des_list.shape[0] <= 1:
            return []

        matches = self.matcher.knnMatch(np.array(i1.des_list),
                                        trainDescriptors=np.array(i2.des_list),
                                        k=2)
        print('  raw matches:', len(matches))

        sum = 0.0
        max_good = 0
        sum_good = 0.0
        count_good = 0
        for m in matches:
            sum += m[0].distance
            if m[0].distance <= m[1].distance * self.match_ratio:
                sum_good += m[0].distance
                count_good += 1
                if m[0].distance > max_good:
                    max_good = m[0].distance
        print('  avg dist:', sum / len(matches))
        if count_good:
            print('  avg good dist:', sum_good / count_good, '(%d)' % count_good)
        print('  max good dist:', max_good)

        if False:
            # filter by absolute distance (for ORB, statistically all real
            # matches will have a distance < 64, for SIFT I don't know,
            # but I'm guessing anything more than 270.0 is a bad match.
            matches_thresh = []
            for m in matches:
                if m[0].distance < self.max_distance and m[0].distance <= m[1].distance * self.match_ratio:
                    matches_thresh.append(m[0])
            print('  quality matches:', len(matches_thresh))

        if True:
            # generate a quality metric for each match, sort and only
            # pass along the best 'n' matches that pass the distance
            # ratio test.  (Testing the idea that 2000 matches aren't
            # better than 20 if they are good matches with respect to
            # optimizing the fit.)
            by_metric = []
            for m in matches:
                ratio = m[0].distance / m[1].distance # smaller is better
                metric = m[0].distance * ratio
                by_metric.append( [metric, m[0]] )
            by_metric = sorted(by_metric, key=lambda fields: fields[0])
            matches_thresh = []
            for line in by_metric:
                if line[0] < self.max_distance * self.match_ratio:
                    matches_thresh.append(line[1])
            print('  quality matches:', len(matches_thresh))
            # fixme, make this a command line option or parameter?
            mymax = 2000
            if len(matches_thresh) > mymax:
                # clip list to n best rated matches
                matches_thresh = matches_thresh[:mymax]
                print('  clipping to:', mymax)

        if len(matches_thresh) < self.min_pairs:
            # just quit now
            return []

        w = self.camera_node.getInt('width_px')
        h = self.camera_node.getInt('height_px')
        if not w or not h:
            print("Zero image sizes will crash matchGMS():", w, h)
            print("Possibly the detect feature step was killed and restarted?")
            print("Recommend removing all meta/*.feat and meta/*.desc and")
            print("rerun the feature detection step.")
            print("... or do some coding to add this information to the")
            print("ImageAnalysis/meta/<image_name>.json files")
            quit()
        size = (w, h)
            
        matchesGMS = cv2.xfeatures2d.matchGMS(size, size, i1.kp_list, i2.kp_list, matches_thresh, withRotation=True, withScale=False, thresholdFactor=5.0)
        #matchesGMS = cv2.xfeatures2d.matchGMS(size1, size2, i1.uv_list, i2.uv_list, matches_thresh, withRotation=True, withScale=False)
        #print('matchesGMS:', matchesGMS)
            
        idx_pairs = []
        for i, m in enumerate(matchesGMS):
            idx_pairs.append( [m.queryIdx, m.trainIdx] )
            
        if False:
            # run the classic feature distance ratio test (already
            # handled above in a slightly more strategic way by
            # passing the best matches, not just all the matches.)
            p1, p2, kp_pairs, idx_pairs = self.filter_by_feature(i1, i2, matches)
            print("  dist ratio matches =", len(idx_pairs))

        # check for duplicate matches (based on different scales or attributes)
        idx_pairs = self.filter_duplicates(i1, i2, idx_pairs)

        # look for common feature angle difference (should we
        # depricate this step?)
        if False and len(idx_pairs):
            # do a quick test of relative feature angles
            offsets = []
            for pair in idx_pairs:
                p1 = i1.kp_list[pair[0]]
                p2 = i2.kp_list[pair[1]]
                offset = p2.angle - p1.angle
                if offset < -180: offset += 360
                if offset > 180: offset -= 360
                offsets.append(offset)
                #print('angles:', p1.angle, p2.angle, offset)
            offsets = np.array(offsets)
            offset_avg = np.mean(offsets)
            offset_std = np.std(offsets)
            print('gms inlier offset.  avg: %.1f std: %.1f' % (offset_avg, offset_std))
            # carry forward the aligned pairs
            aligned_pairs = []
            for pair in idx_pairs:
                p1 = i1.kp_list[pair[0]]
                p2 = i2.kp_list[pair[1]]
                offset = p2.angle - p1.angle
                if offset < -180: offset += 360
                if offset > 180: offset -= 360
                diff = offset - offset_avg
                if diff < -180: diff += 360
                if diff > 180: diff -= 360
                if abs(diff) <= 10:
                    aligned_pairs.append(pair)
            if len(idx_pairs) > len(aligned_pairs):
                print('  feature alignment:', len(idx_pairs), '->', len(aligned_pairs))
                idx_pairs = aligned_pairs
            
        print("  initial matches =", len(idx_pairs))
        if len(idx_pairs) < self.min_pairs:
            # sorry
            return []
        else:
            return idx_pairs
    
    # do initial feature matching (both ways) for the specified image
    # pair.
    def bidirectional_matches(self, image_list, i, j, review=False):
        if i == j:
            print("We shouldn't see this, but i == j", i, j)
            return [], []

        i1 = image_list[i]
        i2 = image_list[j]

        # all vs. all match between overlapping i1 keypoints and i2
        # keypoints (forward match)
        idx_pairs1 = self.basic_matches(i1, i2)

        if len(idx_pairs1) >= self.min_pairs:
            # all vs. all match between overlapping i2 keypoints and i1
            # keypoints (reverse match)
            idx_pairs2 = self.basic_matches(i2, i1)
        else:
            # don't bother checking the reciprocal direction if we
            # didn't find matches in the forward direction
            idx_pairs2 = []

        idx_pairs1, idx_pairs2 = self.filter_cross_check(idx_pairs1, idx_pairs2)
        
        plot_matches = False
        if plot_matches:
            self.plot_matches(i1, i2, idx_pairs1)
            self.plot_matches(i2, i1, idx_pairs2)
            
        if review:
            if len(idx_pairs1):
                status, key = self.showMatchOrient(i1, i2, idx_pairs1)
                # remove deselected pairs
                for k, flag in enumerate(status):
                    if not flag:
                        print("    deleting: " + str(idx_pairs1[k]))
                        idx_pairs1[k] = (-1, -1)
                for pair in reversed(idx_pairs1):
                    if pair == (-1, -1):
                        idx_pairs1.remove(pair)
                        
            if len(idx_pairs2):
                status, key = self.showMatchOrient(i2, i1, idx_pairs2)
                # remove deselected pairs
                for k, flag in enumerate(status):
                    if not flag:
                        print("    deleting: " + str(idx_pairs2[k]))
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


    def robustGroupMatches(self, image_list, K,
                           filter="fundamental", review=False):
        min_dist = self.matcher_node.getFloat('min_dist')
        max_dist = self.matcher_node.getFloat('max_dist')
        print('Generating work list for range:', min_dist, '-', max_dist)
        
        n = len(image_list) - 1
        n_work = float(n*(n+1)/2)
        t_start = time.time()

        # camera separation vs. matches stats
        dist_stats = []

        # pass 1, make a list of all the match pairs with their
        # physical camera separation, then sort by distance and matche
        # closest first
        work_list = []
        for i, i1 in enumerate(tqdm(image_list)):
            #shouldn't need to allocate space now that it's a dict
            #if len(i1.match_list) == 0:
            #    i1.match_list = {}

            for j, i2 in enumerate(image_list):
                if j <= i:
                    continue
                
                # camera pose distance check
                ned1, ypr1, q1 = i1.get_camera_pose()
                ned2, ypr2, q2 = i2.get_camera_pose()
                dist = np.linalg.norm(np.array(ned2) - np.array(ned1))
                if dist >= min_dist and dist <= max_dist:
                    work_list.append( [dist, i, j] )

        # (optional) sort worklist from closest pairs to furthest pairs
        #
        # benefits of sorting by distance: most important work is done
        # first (chance to quit early)
        #
        # benefits of sorting by order: for large memory usage, active
        # memory pool decreases as work progresses (becoming more and
        # more system friendly.)
        
        # work_list = sorted(work_list, key=lambda fields: fields[0])

        # note: image.desc_timestamp is used to unload not recently
        # used descriptors ... these burn a ton of memory so unloading
        # things not recently used should help our memory foot print
        # at hopefully not too much of a performance expense.
        
        # proces the work list
        n_count = 0
        save_time = time.time()
        save_interval = 120     # seconds
        for line in work_list:
            dist = line[0]
            i = line[1]
            j = line[2]
            i1 = image_list[i]
            i2 = image_list[j]

            # eta estimation
            percent = n_count / float(len(work_list))
            n_count += 1
            t_elapsed = time.time() - t_start
            if percent > 0:
                t_end = t_elapsed / percent
            else:
                t_end = t_start
            t_remain = t_end - t_elapsed

            # skip if match has already been computed
            if i2.name in i1.match_list and i1.name in i2.match_list:
                print('Skipping: ', i1.name, 'vs', i2.name, 'already done.')
                continue
            
            print('Matching %s vs %s - ' % (i1.name, i2.name), end='')
            print('%.1f%% done: ' % (percent * 100.0), end='')
            if t_remain < 3600:
                print('%.1f (min)' % (t_remain / 60.0))
            else:
                print('%.1f (hr)' % (t_remain / 3600.0))
            print("  separation = %.1f (m)" % dist)

            # update cache timers and make sure features are loaded
            i1.desc_timestamp = time.time()
            i2.desc_timestamp = time.time()
            i1.load_descriptors()
            i2.load_descriptors()

            #shouldn't need to do this
            #if len(i2.match_list) == 0:
            #    # create if needed
            #    i2.match_list = [[]] * len(image_list)
            i1.match_list[i2.name], i2.match_list[i1.name] \
                = self.bidirectional_matches(image_list, i, j, review)

            scheme = 'none'
            # scheme = 'one_step'
            # scheme = 'iterative'

            if scheme == 'iterative':
                done = False
                while not done:
                    done = True
                    if not self.filter_non_reciprocal_pair(image_list, i, j):
                        done = False
                    if not self.filter_non_reciprocal_pair(image_list, j, i):
                        done = False
                    if not self.filter_by_homography(K, i1, i2, j, filter):
                        done = False
                    if not self.filter_by_homography(K, i2, i1, i, filter):
                        done = False
            elif scheme == 'one_step':
                # quickly dump non-reciprocals from initial results
                self.filter_non_reciprocal_pair(image_list, i, j)
                self.filter_non_reciprocal_pair(image_list, j, i)
                # filter the remaining features by 'filter' relationship
                self.filter_by_homography(K, i1, i2, j, filter)
                self.filter_by_homography(K, i2, i1, i, filter)
                # cull any new non-reciprocals
                self.filter_non_reciprocal_pair(image_list, i, j)
                self.filter_non_reciprocal_pair(image_list, j, i)
            dist_stats.append( [ dist, len(i1.match_list[i2.name]) ] )

            # save our work so far, and flush descriptor cache
            if time.time() >= save_time + save_interval:
                print('saving matches ...')
                self.saveMatches(image_list)
                save_time = time.time()

                time_list = []
                for i3 in image_list:
                    if not i3.des_list is None:
                        time_list.append( [i3.desc_timestamp, i3] )
                time_list = sorted(time_list, key=lambda fields: fields[0],
                                   reverse=True)
                # may wish to monitor and update cache_size formula
                cache_size = 20 + 3 * (int(math.sqrt(len(image_list))) + 1)
                flush_list = time_list[cache_size:]
                print('flushing descriptor cache - size: %d (over by: %d)' % (cache_size, len(flush_list)) )
                for line in flush_list:
                    print('  clearing descriptors for:', line[1].name)
                    line[1].des_list = None
                    
        # and save
        self.saveMatches(image_list)
        print('Pair-wise matches successfully saved.')

        dist_stats = np.array(dist_stats)
        plt.plot(dist_stats[:,0], dist_stats[:,1], 'ro')
        # TODO: Make a feature switch to stop showing when using the project runner.
        plt.show()

    # remove any match sets shorter than self.min_pairs (this shouldn't
    # probably ever happen now?)
    def cullShortMatches(self, image_list):
        for i, i1 in enumerate(image_list):
            print("(needed?) Cull matches for %s" % i1.name)
            for key in i1.match_list:
                matches = i1.match_list[key]
                if len(matches) < self.min_pairs:
                    print('  Culling pair index:', j)
                    i1.match_list[key] = []

    def saveMatches(self, image_list):
        for image in image_list:
            image.save_matches()

        
###########################################################
###   STUFF BELOW HERE IS OLD AND POSSIBLY DEPRECATED   ###
###########################################################

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

    def showMatchOrient(self, i1, i2, idx_pairs, status=None, orient='relative'):
        #print " -- idx_pairs = " + str(idx_pairs)
        img1 = i1.load_gray()
        img2 = i2.load_gray()

        # compute the affine transformation between points.  This is
        # used to determine relative orientation of the two images,
        # and possibly estimate outliers if no status array is
        # provided.
        
        src = []
        dst = []
        for pair in idx_pairs:
            src.append( i1.kp_list[pair[0]].pt )
            dst.append( i2.kp_list[pair[1]].pt )
        affine, status = \
            cv2.estimateAffinePartial2D(np.array([src]).astype(np.float32),
                                        np.array([dst]).astype(np.float32))
        print('affine:', affine)
        if affine is None:
            affine = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]])
            # auto reject any pairs where we can't determine a proper affine transformation at all
            status = np.zeros(len(idx_pairs), np.bool_)
            return status, ord(' ')
        (rot, tx, ty, sx, sy) = self.decomposeAffine(affine)
        print(' ', rot, tx, ty, sx, sy)

        if status is None:
            status = np.ones(len(idx_pairs), np.bool_)
            # for each src point, compute dst_est[i] = src[i] * affine
            error = []
            for i, p in enumerate(src):
                p_est = affine.dot( np.hstack((p, 1.0)) )[:2]
                # print('p est:', p_est, 'act:', dst[i])
                #np1 = np.array(i1.coord_list[pair[0]])
                #np2 = np.array(i2.coord_list[pair[1]])
                d = np.linalg.norm(p_est - dst[i])
                # print('dist:', d)
                error.append(d)
            # print('errors:', error)
            error = np.array(error)
            avg = np.mean(error)
            std = np.std(error)
            print('avg:', avg, 'std:', std)

            # mark the potential outliers
            for i in range(len(idx_pairs)):
                if error[i] > avg + 3*std:
                    status[i] = False
                
        print('orientation:', orient)
        if orient == 'relative':
            # estimate relative orientation between features
            yaw1 = 0
            yaw2 = rot
        elif orient == 'aircraft':
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
        print( yaw1, yaw2)
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
            # print p1, p2
            kp_pairs.append( (kp1, kp2) )

        key = explore_match('find_obj', si1, si2, kp_pairs,
                            hscale=1.0, wscale=1.0, status=status)
        
        # status structure represents in/outlier choices of user.
        # explore_match() modifies the status array in place.
        
        cv2.destroyAllWindows()

        # status is an array of booleans that parallels the pair array
        # and represents the users choice to keep or discard the
        # respective pairs.
        return status, key

    def showMatches(self, i1):
        for key in i1.match_list:
            idx_pairs = i1.match_list[key]
            if len(idx_pairs) >= self.min_pairs:
                i2 = self.findImageByName(key)
                print("Showing matches for image %s and %s" % (i1.name, i2.name))
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
        for key in enumerate(i1.match_list):
            matches = i1.match_list[key]
            if len(matches):
                i1.num_matches += 1
                i2 = self.findImageByName[key]
                #print "Matching %s vs %s " % (i1.name, i2.name)
                error = 0.0
                if variance:
                    var = self.imagePairVariance2(i, alt_coord_list, j, matches)
                    #print "  %s var = %.2f" % (i1.name, var)
                    var_sum += var * i2.weight
                else:
                    error = self.imagePairError(i, alt_coord_list, j,
                                                matches, emax)
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
        i1.match_list[i2.name].remove(pair)
        #print "i1 pairs after = %s" % str(i1.match_list[j])
        pair_rev = (pair[1], pair[0])
        #print "i2 pairs before = %s" % str(i2.match_list[i])
        i2.match_list[i1.name].remove(pair_rev)
        #print "i2 pairs after = %s" % str(i2.match_list[i])

    # compute the error between a pair of images
    def pairErrorReport(self, i, alt_coord_list, j, minError):
        i1 = self.image_list[i]
        i2 = self.image_list[j]
        match = i1.match_list[i2.name]
        print("pair error report %s v. %s (%d)" % (i1.name, i2.name, len(match)))
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
        print("   error avg = %.2f stddev = %.2f" % (error_avg, stddev))

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
                    print("    deleting: " + str(match[k]))
                    #match[i] = (-1, -1)
                    delete_list.append(match[k])

        if False: # for line in report_list:
            print("    %.1f %s" % (line[0], str(match[line[1]])))
            if line[0] > 50.0 or line[0] > (error_avg + 5*stddev):
                # if positional error > 50m delete pair
                done = False
                while not done:
                    print("Found a suspect match: d)elete v)iew [o]k: ",)
                    reply = find_getch()
                    print
                    if reply == 'd':
                        match[line[1]] = (-1, -1)
                        dirty = True
                        done = True;
                        print("    (deleted) " + str(match[line[1]]))
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

    def findImageByName(self, search):
        for image in self.image_list:
            if search == image.name:
                return image
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
            print("tol = %.4f" % tol)
            if tol < 0.0:
                tol = 0.0
            for key in i1.match_list:
                matches = i1.match_list[key]
                i2 = self.findImageByName[key]
                if i1.name == i2.name:
                    continue
                if len(matches) < self.min_pairs:
                    i1.match_list[i2.name] = []
                    continue
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
                print('  %s vs %s: %d / %d  inliers/matched' \
                    % (i1.name, i2.name, inliers, size))

                if inliers < size:
                    total_removed += (size - inliers)
                    if interactive:
                        status = self.showMatch(i1, i2, matches, status)

                    delete_list = []
                    for k, flag in enumerate(status):
                        if not flag:
                            print("    deleting: " + str(matches[k]))
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
            for key in i1.match_list:
                matches = i1.match_list[key]
                i2 = self.findImageByName[key]
                if i1.name == i2.name:
                    continue
                if len(matches) < self.min_pairs:
                    i1.match_list[i2.name] = []
                    continue
                pts = []
                status = []
                for k, pair in enumerate(matches):
                    pts.append( i1.kp_list[pair[0]].pt )
                    status.append(False)

                # check for degenerate case of all matches being
                # pretty close to a straight line
                if self.isLinear(pts, threshold):
                    print("%s vs %s is a linear match, probably should discard" % (i1.name, i2.name))
                    
                    status = self.showMatch(i1, i2, matches, status)

                    delete_list = []
                    for k, flag in enumerate(status):
                        if not flag:
                            print("    deleting: " + str(matches[k]))
                            #match[i] = (-1, -1)
                            delete_list.append(matches[k])

                    for pair in delete_list:
                        self.deletePair(i, j, pair)

    # Review matches by fewest pairs of keypoints.  The fewer
    # keypoints that join a pair of images, the greater the chance
    # that we could have stumbled on a degenerate or incorrect set.
    def reviewByFewestPairs(self, maxpairs=8):
        print("Review matches by fewest number of pairs")
        if len(self.image_list):
            report_list = []
            for i, image in enumerate(self.image_list):
                for key in image.match_list:
                    matches = image.match_list[key]
                    e = len(matches)
                    if e > 0 and e <= maxpairs:
                        report_list.append( (e, i, key) )
            report_list = sorted(report_list, key=lambda fields: fields[0],
                                 reverse=False)
            # show images sorted by largest positional disagreement first
            for line in report_list:
                i1 = self.image_list[line[1]]
                i2 = self.findImageByName(line[2])
                pairs = i1.match_list[line[2]]
                if len(pairs) == 0:
                    # probably already deleted by the other match order
                    continue
                print("showing %s vs %s: %d pairs" \
                    % (i1.name, i2.name, len(pairs)))
                status = self.showMatch(i1, i2, pairs)
                delete_list = []
                for k, flag in enumerate(status):
                    if not flag:
                        print("    deleting: " + str(pairs[k]))
                        #match[i] = (-1, -1)
                        delete_list.append(pairs[k])
                for pair in delete_list:
                    self.deletePair(line[1], line[2], pair)

    # sort and review match pairs by worst positional error
    def matchErrorReport(self, i, minError=20.0):
        i1 = self.image_list[i]
        # now for each image, find/show worst individual matches
        report_list = []
        for key in i1.match_list:
            matches = i1.match_list[key]
            if len(matches):
                i2 = self.findImageByName[key]
                #print "Matching %s vs %s " % (i1.name, i2.name)
                e = self.imagePairError(i, None, key, matches)
                if e > minError:
                    report_list.append( (e, i, key) )

        report_list = sorted(report_list,
                             key=lambda fields: fields[0],
                             reverse=True)

        for line in report_list:
            i1 = self.image_list[line[1]]
            i2 = self.findImageByName(line[2])
            print("min error = %.3f" % minError)
            print("%.1f %s %s" % (line[0], i1.name, i2.name))
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
            for key in i1.match_list:
                matches = i1.match_list[key]
                if len(matches):
                    i2 = self.findImageByName(key)
                    #print "Matching %s vs %s " % (i1.name, i2.name)
                    e = self.imagePairError(i, None, key, matches)
                    if e > minError:
                        report_list.append( (e, i, key) )

        report_list = sorted(report_list,
                             key=lambda fields: fields[0],
                             reverse=True)

        for line in report_list:
            i1 = self.image_list[line[1]]
            i2 = self.findImageByName(line[2])
            print("min error = %.3f" % minError)
            print("%.1f %s %s" % (line[0], i1.name, i2.name))
            if line[0] > minError:
                #print "  %s" % str(pairs)
                self.pairErrorReport(line[1], None, line[2], minError)
                #print "  after %s" % str(match)
                #self.showMatch(i1, i2, match)

    def reviewPoint(self, lon_deg, lat_deg, ref_lon, ref_lat):
        (x, y) = ImageList.wgs842cart(lon_deg, lat_deg, ref_lon, ref_lat)
        print("Review images touching %.2f %.2f" % (x, y))
        review_list = ImageList.getImagesCoveringPoint(self.image_list, x, y, pad=25.0, only_placed=False)
        #print "  Images = %s" % str(review_list)
        for image in review_list:
            print("    %s -> " % image.name,)
            for key in image.match_list:
                matches = image.match_list[key]
                if len(matches):
                    print("%s (%d) " % (self.image_list[j].name, len(matches)),)
            print
            r2 = image.coverage()
            p = ImageList.getImagesCoveringRectangle(self.image_list, r2)
            p_names = []
            for i in p:
                p_names.append(i.name)
            print("      possible matches: %d" % len(p_names))

            
# collect and group match chains that refer to the same keypoint
def group_matches(matches_direct):
    # this match grouping function appears to product more entries
    # rather than reduce the entries.  I don't understand how that
    # could be, but it needs to be fixed bore the function is used for
    # anything.
    print('WARNING: this function does not seem to work correctly, need to debug it before using!!!!')
    print('Number of pair-wise matches:', len(matches_direct))
    matches_group = list(matches_direct) # shallow copy
    count = 0
    done = False
    while not done:
        print("Iteration:", count, len(matches_group))
        count += 1
        matches_new = []
        matches_lookup = {}
        for i, match in enumerate(matches_group):
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
                # only append items that don't already exist in the early
                # match, and only one match per image (!)
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
        if len(matches_new) == len(matches_group):
            done = True
        else:
            matches_group = list(matches_new) # shallow copy
    print("Unique features (after grouping):", len(matches_group))
    return matches_group
            
