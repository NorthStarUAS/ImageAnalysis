import cv2
import math
import numpy as np

from find_obj import filter_matches,explore_match

class Match():
    def __init__(self, image_group):
        self.ig = image_group
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True)
        #self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def computeMatches(self, showpairs=False):
        # O(n,n) compare
        for i, i1 in enumerate(self.ig.image_list):
            if len(i1.match_list):
                continue
            for j, i2 in enumerate(self.ig.image_list):
                matches = self.ig.bf.knnMatch(i1.des_list, trainDescriptors=i2.des_list, k=2)
                p1, p2, kp_pairs, idx_pairs = self.ig.filterMatches2(i1.kp_list, i2.kp_list, matches)
                #print "index pairs:"
                #print str(idx_pairs)
                #if i == j:
                #    continue

                if i != j:
                    i1.match_list.append( idx_pairs )
                else:
                    i1.match_list.append( [] )

                if len(idx_pairs):
                    print "Matching " + str(i) + " vs " + str(j) + " = " + str(len(idx_pairs))

                if len(idx_pairs) > 2:
                    if False:
                        # draw only keypoints location,not size and orientation (flags=0)
                        # draw rich keypoints (flags=4)
                        res1 = cv2.drawKeypoints(img_list[i], kp_list[i], color=(0,255,0), flags=0)
                        res2 = cv2.drawKeypoints(img_list[j], kp_list[j], color=(0,255,0), flags=0)
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
                        explore_match('find_obj', i1.img, i2.img, kp_pairs) #cv2 shows image
                        cv2.waitKey()
                        cv2.destroyAllWindows()
            i1.save_matches()

    def safeAddPair(self, i1, i2, refpair):
        image1 = self.ig.image_list[i1]
        image2 = self.ig.image_list[i2]
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
        for i, i1 in enumerate(self.ig.image_list):
            print "add inverse matches for %s" % i1.name
            for j, matches in enumerate(i1.match_list):
                if i == j:
                    continue
                i2 = self.ig.image_list[j]
                for pair in matches:
                    inv_pair = (pair[1], pair[0])
                    self.safeAddPair(j, i, inv_pair)
        for i1 in self.ig.image_list:
            i1.save_matches()


    def saveMatches(self):
        for image in self.ig.image_list:
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
        for j, i2 in enumerate(self.ig.image_list):
            print str(i1.match_list[j])
            idx_pairs = i1.match_list[j]
            if len(idx_pairs) > 0:
                print "Showing matches for image %s and %s" % (i1.name, i2.name)
                self.showMatch( i1, i2, idx_pairs )

    def showAllMatches(self):
        # O(n,n) compare
        for i, i1 in enumerate(self.ig.image_list):
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
                i2 = self.ig.image_list[i]
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
        if len(self.ig.image_list):
            report_list = []
            if name != None:
                image = self.ig.findImageByName(name)
                e = self.ig.imageError(image, None, max=True)
                report_list.append( (e, image.name) )
            else:
                for image in self.ig.image_list:
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

