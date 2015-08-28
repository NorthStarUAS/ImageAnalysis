import cv2
import math
from matplotlib import pyplot as plt
import numpy as np

from find_obj import filter_matches,explore_match
import ImageList

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
    # if keypoints were removed
    def filter_by_homography(self, i1, i2, j, filter):
        clean = True
        
        tol = float(i1.width) / 250.0 # rejection range in pixels
        # print "tol = %.4f" % tol
        matches = i1.match_list[j]
        if len(matches) < 8:
            i1.match_list[j] = []
            return True
        p1 = []
        p2 = []
        for k, pair in enumerate(matches):
            p1.append( i1.kp_list[pair[0]].pt )
            p2.append( i2.kp_list[pair[1]].pt )

        p1 = np.float32(p1)
        p2 = np.float32(p2)
        #print "p1 = %s" % str(p1)
        #print "p2 = %s" % str(p2)
        if filter == "homography":
            M, status = cv2.findHomography(p1, p2, cv2.RANSAC, tol)
        elif filter == "fundamental":
            M, status = cv2.findFundamentalMat(p1, p2, cv2.RANSAC, tol)
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

    def filter_non_reciprocal(self, image_list):
        clean = True
        print "Removing non-reciprocal matches:"
        for i, i1 in enumerate(image_list):
            for j, i2 in enumerate(image_list):
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
        
    # do initial feature matching of specified image against every
    # image in the provided image list (except self)
    def hybridImageMatches(self, i1, i2, review=False):
        if i1 == i2:
            return []
        if np.any(i1.des_list):
            size = len(i1.des_list)
        else:
            size = 0
        des_list2 = []
        # find candidate features within a location proximity
        if i2.kdtree != None:
            result = i2.kdtree.query_ball_point(i1.center, i1.radius + 20.0)
        else:
            result = []
        if len(result) == 0:
            return []

        # all vs. all match between i1 keypoints and any i2
        # keypoints within the query radius
        for k in result:
            des_list2.append(i2.des_list[k])
        matches = self.matcher.knnMatch(np.array(i1.des_list),
                                        trainDescriptors=np.array(des_list2),
                                        k=2)
        # rewrite the train indices (from our query subset)
        # back to referencing the full set of keypoints
        for match in matches:
            for m in match:
                qi = m.trainIdx
                m.trainIdx = result[qi]

        # run the classic feature distance ratio test
        p1, p2, kp_pairs, idx_pairs = self.filter_by_feature(i1, i2, matches)

        # test each individual match pair for proximity to each
        # other (fine grain distance check)
        idx_pairs = self.filter_by_location(i1, i2, idx_pairs, 10.0)

        if len(idx_pairs) < self.min_pairs:
            return []
        print "  pairs = ", len(idx_pairs)

        if review:
            status = self.showMatch(i1, i2, idx_pairs)
            # remove deselected pairs
            for k, flag in enumerate(status):
                if not flag:
                    print "    deleting: " + str(idx_pairs[k])
                    idx_pairs[k] = (-1, -1)
            for pair in reversed(idx_pairs):
                if pair == (-1, -1):
                    idx_pairs.remove(pair)

        return idx_pairs

    def robustGroupMatches(self, image_list, filter="fundamental",
                           review=False):
        # find basic matches and filter by match ratio and ned
        # location
        for i, i1 in enumerate(image_list):
            if len(i1.match_list):
                continue
            
            i1.match_list = [[]] * len(image_list)

            for j, i2 in enumerate(image_list):
                if i1 == i2:
                    continue
                print "Matching %s vs %s" % (i1.name, i2.name)
                i1.match_list[j] = self.hybridImageMatches(i1, i2, review)

        done = False
        while not done:
            done = True
            
            # eliminate any non-reciprocal matches (a good sign they
            # are bad or weak matches)
            if not self.filter_non_reciprocal(image_list):
                done = False
        
            # filter against a Homography or Fundametal matrix
            # constraint
            for i, i1 in enumerate(image_list):
                for j, i2 in enumerate(image_list):
                    if i1 == i2:
                        continue
                    print "Matching %s vs %s" % (i1.name, i2.name)
                    if not self.filter_by_homography(i1, i2, j, filter):
                        done = False

        for i1 in image_list:
            i1.save_matches()

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
    def cullShortMatches(self):
        for i, i1 in enumerate(self.image_list):
            print "Cull matches for %s" % i1.name
            for j, matches in enumerate(i1.match_list):
                if len(matches) < self.min_pairs:
                    i1.match_list[j] = []
        for i1 in self.image_list:
            i1.save_matches()

    def saveMatches(self):
        for image in self.image_list:
            image.save_matches()

    def showMatch(self, i1, i2, idx_pairs, status=None):
        #print " -- idx_pairs = " + str(idx_pairs)
        kp_pairs = []
        for p in idx_pairs:
            kp1 = i1.kp_list[p[0]]
            kp2 = i2.kp_list[p[1]]
            kp_pairs.append( (kp1, kp2) )
        if i1.img == None:
            i1.load_rgb()
        if i2.img == None:
            i2.load_rgb()
        if status == None:
            status = np.ones(len(kp_pairs), np.bool_)
        h, w = i1.img.shape
        scale = 790.0/float(w)
        si1 = cv2.resize(i1.img, (0,0), fx=scale, fy=scale)
        si2 = cv2.resize(i2.img, (0,0), fx=scale, fy=scale)
        explore_match('find_obj', si1, si2, kp_pairs,
                      hscale=scale, wscale=scale, status=status)
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

        # computers best estimation of valid vs. suspect pairs
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
            tol = float(i1.width) / 400.0 + fuzz_factor
            print "tol = %.4f" % tol
            if tol < 0.0:
                tol = 0.0
            for j, matches in enumerate(i1.match_list):
                if i == j:
                    continue
                if len(matches) < 8:
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
                if len(matches) < 8:
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
                
