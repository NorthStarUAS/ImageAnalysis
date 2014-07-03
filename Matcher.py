import cv2
import math
from matplotlib import pyplot as plt
import numpy as np

from find_obj import filter_matches,explore_match

class Match():
    def __init__(self):
        self.image_list = []
        self.detector = None
        self.matcher = None
        self.dense_detect_grid = 1
        self.match_ratio = 0.75
        #self.bf = cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True)
        #self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def configure(self, dparams={}, mparams={}):
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        FLANN_INDEX_LSH    = 6
        if "detector" in dparams:
            if dparams["detector"] == "sift":
                self.detector = cv2.SIFT()
                norm = cv2.NORM_L2
            elif dparams["detector"] == "surf":
                threshold = dparams["hessian_threshold"]
                self.detector = cv2.SURF(threshold)
                norm = cv2.NORM_L2
            elif dparams["detector"] == "orb":
                dmax_features = dparams["orb_max_features"]
                self.detector = cv2.ORB(dmax_features)
                norm = cv2.NORM_HAMMING
        if "dense_detect_grid" in dparams:
            self.dense_detect_grid = dparams["dense_detect_grid"]

        if "matcher" in mparams:
            if mparams["matcher"] == "flann":
                if norm == cv2.NORM_L2:
                    flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                                        trees = 5)
                else:
                    flann_params = dict(algorithm = FLANN_INDEX_LSH,
                                        table_number = 6, # 12
                                        key_size = 12,     # 20
                                        multi_probe_level = 1) #2
                self.matcher = cv2.FlannBasedMatcher(flann_params, {}) # bug : need to pass empty dict (#1329)
            elif mparams["matcher"] == "bruteforce":
                print "brute force norm = %d" % norm
                self.matcher = cv2.BFMatcher(norm)
        if "match_ratio" in mparams:
            self.match_ratio = mparams["match_ratio"]

    def setImageList(self, image_list):
        self.image_list = image_list

    def denseDetect(self, image):
        steps = self.dense_detect_grid
        kp_list = []
        h, w = image.shape
        dx = 1.0 / float(steps)
        dy = 1.0 / float(steps)
        x = 0.0
        for i in xrange(steps):
            y = 0.0
            for j in xrange(steps):
                #print "create mask (%dx%d) %d %d" % (w, h, i, j)
                #print "  roi = %.2f,%.2f %.2f,%2f" % (y*h,(y+dy)*h-1, x*w,(x+dx)*w-1)
                mask = np.zeros((h,w,1), np.uint8)
                mask[y*h:(y+dy)*h-1,x*w:(x+dx)*w-1] = 255
                kps = self.detector.detect(image, mask)
                if False:
                    res = cv2.drawKeypoints(image, kps,
                                            color=(0,255,0), flags=0)
                    fig1, plt1 = plt.subplots(1)
                    plt1 = plt.imshow(res)
                    plt.show()
                kp_list.extend( kps )
                y += dy
            x += dx
        return kp_list

    def computeDescriptors(self, image, kp_list):
        kp_list, des_list = self.detector.compute(image, kp_list)
        return kp_list, des_list
 
    def filterMatches1(self, kp1, kp2, matches):
        mkp1, mkp2 = [], []
        idx_pairs = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.match_ratio:
                #print " dist[0] = %d  dist[1] = %d" % (m[0].distance, m[1].distance)
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
                idx_pairs.append( (m.queryIdx, m.trainIdx) )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, kp_pairs, idx_pairs

    def filterMatches2(self, kp1, kp2, matches):
        mkp1, mkp2 = [], []
        idx_pairs = []
        used = np.zeros(len(kp2), np.bool_)
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.match_ratio:
                #print " dist[0] = %d  dist[1] = %d" % (m[0].distance, m[1].distance)
                m = m[0]
                if not used[m.trainIdx]:
                    used[m.trainIdx] = True
                    mkp1.append( kp1[m.queryIdx] )
                    mkp2.append( kp2[m.trainIdx] )
                    idx_pairs.append( (m.queryIdx, m.trainIdx) )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, kp_pairs, idx_pairs

    def computeMatches(self, showpairs=False):
        # O(n,n) compare
        for i, i1 in enumerate(self.image_list):
            if len(i1.match_list):
                continue
            i1.match_list = [[]] * len(self.image_list)
            for j, i2 in enumerate(self.image_list):
                if i == j:
                    continue
                matches = self.matcher.knnMatch(i1.des_list, trainDescriptors=i2.des_list, k=2)
                p1, p2, kp_pairs, idx_pairs = self.filterMatches2(i1.kp_list, i2.kp_list, matches)
                if len(p1) >= 4:
                    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                    print '%d / %d  inliers/matched' % (np.sum(status), len(status))
                    # remove outliers
                    for k, flag in enumerate(status):
                        if not flag:
                            print "    deleting: " + str(idx_pairs[k])
                            idx_pairs[k] = (-1, -1)
                    for pair in reversed(idx_pairs):
                        if idx_pairs == (-1, -1):
                            idx_pairs.remove(pair)
 
                else:
                    H, status = None, None
                    # print '%d matches found, not enough for homography estimation' % len(p1)
                i1.match_list[j] = idx_pairs

                if len(idx_pairs):
                    print "Matching %s vs %s (%d vs %d) = %d" \
                        % (i1.name, i2.name, i, j, len(idx_pairs))

                if len(idx_pairs) >= 4:
                    if False:
                        # draw only keypoints location,not size and orientation (flags=0)
                        # draw rich keypoints (flags=4)
                        if i1.img == None:
                            i1.load_image()
                        if i2.img == None:
                            i2.load_image()
                        res1 = cv2.drawKeypoints(i1.img, i1.kp_list, color=(0,255,0), flags=0)
                        res2 = cv2.drawKeypoints(i2.img, i2.kp_list, color=(0,255,0), flags=0)
                        fig1, plt1 = plt.subplots(1)
                        plt1 = plt.imshow(res1)
                        fig2, plt2 = plt.subplots(1)
                        plt2 = plt.imshow(res2)
                        plt.show(block=False)

                    if showpairs:
                        if i1.img == None:
                            i1.load_image()
                        if i2.img == None:
                            i2.load_image()
                        explore_match('find_obj', i1.img, i2.img, kp_pairs, status, H) #cv2 shows image
                        cv2.waitKey()
                        cv2.destroyAllWindows()
            i1.save_matches()

    def safeAddPair(self, i1, i2, refpair):
        image1 = self.image_list[i1]
        image2 = self.image_list[i2]
        matches = image1.match_list[i2]
        hasit = False
        for pair in matches:
            if pair[0] == refpair[0] and pair[1] == refpair[1]:
                hasit = True
                print " already exists: %s->%s (%d %d)" % (image1.name, image2.name, refpair[0], refpair[1])
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
            i1.load_image()
        if i2.img == None:
            i2.load_image()
        if status == None:
            status = np.ones(len(kp_pairs), np.bool_)
        explore_match('find_obj', i1.img, i2.img, kp_pairs, status) #cv2 shows image
        cv2.waitKey()
        # status structure will be correct here and represent
        # in/outlier choices of user
        cv2.destroyAllWindows()

        # status is an array of booleans that parallels the pair array
        # and represents the users choice to keep or discard the
        # respective pairs.
        return status

    def showMatches(self, i1):
        for j, i2 in enumerate(self.image_list):
            print str(i1.match_list[j])
            idx_pairs = i1.match_list[j]
            if len(idx_pairs) > 0:
                print "Showing matches for image %s and %s" % (i1.name, i2.name)
                self.showMatch( i1, i2, idx_pairs )

    def showAllMatches(self):
        # O(n,n) compare
        for i, i1 in enumerate(self.image_list):
            showMatches(i1)

    # compute the error between a pair of images
    def pairErrorReport(self, i1, alt_coord_list, i2, match, minError):
        print "min error = %.3f" % minError
        report_list = []
        coord_list = i1.coord_list
        if alt_coord_list != None:
            coord_list = alt_coord_list
        error_sum = 0.0
        for i, pair in enumerate(match):
            c1 = coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            error = math.sqrt(dx*dx + dy*dy)
            error_sum += error
            report_list.append( (error, i) )
        report_list = sorted(report_list,
                             key=lambda fields: fields[0],
                             reverse=True)

        # meta stats on error values
        error_avg = error_sum / len(match)
        stddev_sum = 0.0
        for line in report_list:
            error = line[0]
            stddev_sum += (error_avg-error)*(error_avg-error)
        stddev = math.sqrt(stddev_sum / len(match))
        print "   error avg = %.2f stddev = %.2f" % (error_avg, stddev)

        # computers best estimation of valid vs. suspect pairs
        dirty = False
        if minError < 0.1:
            dirty = True
        status = np.ones(len(match), np.bool_)
        for i, line in enumerate(report_list):
            if line[0] > 50.0 or line[0] > (error_avg + 2*stddev):
                status[line[1]] = False
                dirty = True

        if dirty:
            status = self.showMatch(i1, i2, match, status)
            for i, flag in enumerate(status):
                if not flag:
                    print "    deleting: " + str(match[i])
                    match[i] = (-1, -1)

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
            print "before = " + str(match)
            for pair in reversed(match):
                if pair == (-1, -1):
                    match.remove(pair)
            print "after = " + str(match)

    # sort and review match pairs by worst positional error
    def matchErrorReport(self, i1, minError=20.0):
        # now for each image, find/show worst individual matches
        report_list = []
        for i, match in enumerate(i1.match_list):
            if len(match):
                i2 = self.image_list[i]
                #print "Matching %s vs %s " % (i1.name, i2.name)
                e = self.ig.imagePairError(i1, None, i2, match, emax=True)
                if e > minError:
                    report_list.append( (e, i1.name, i2.name, i) )

        report_list = sorted(report_list,
                             key=lambda fields: fields[0],
                             reverse=True)

        for line in report_list:
            i1 = self.ig.findImageByName(line[1])
            i2 = self.ig.findImageByName(line[2])
            match = i1.match_list[line[3]]
            print "  %.1f %s %s" % (line[0], line[1], line[2])
            if line[0] > minError:
                print "  %s" % str(match)
                self.pairErrorReport(i1, None, i2, match, minError)
                #print "  after %s" % str(match)
                #self.showMatch(i1, i2, match)

    # sort and review images by worst positional error
    def reviewImageErrors(self, name=None, minError=20.0):
        if len(self.image_list):
            report_list = []
            if name != None:
                image = self.ig.findImageByName(name)
                e = self.ig.imageError(image, None, max=True)
                report_list.append( (e, image.name) )
            else:
                for image in self.image_list:
                    e = self.ig.imageError(image, None, max=True)
                    report_list.append( (e, image.name) )
            report_list = sorted(report_list, key=lambda fields: fields[0],
                                 reverse=True)
            # show images sorted by largest positional disagreement first
            for line in report_list:
                print "%.1f %s" % (line[0], line[1])
                if line[0] >= minError:
                    self.matchErrorReport( self.ig.findImageByName(line[1]),
                                           minError )

