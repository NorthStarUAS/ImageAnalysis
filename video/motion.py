# python module to package my favorite motion tracking tricks
# compute the homography matrix representing the best fit motion from
# the previous frame to the current frame.  The H matrix can further
# be decomposed into camera rotation and translation (requires the
# camera calibration matrix)

import cv2
import numpy as np


# a fast optical flow based method, generally is the best choice for
# unobstructed views. Tracked featurs are distributed well across the
# entire image for a best homography fit.
#
# Based on this tutorial: https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/

class myOpticalFlow():
    def __init__(self):
        self.prev_gray = np.zeros(0)

    def update(self, frame):
        # convert to gray scale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # prime the pump if needed
        if self.prev_gray.shape[0] == 0:
            self.prev_gray = curr_gray.copy()
    
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
	                                   minDistance=30,
                                           blockSize=3)

        # compute the optical flow
        if prev_pts is not None:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts, None)
        else:
            prev_pts = np.zeros(0)
            curr_pts = np.zeros(0)
            status = np.zeros(0)

        self.prev_gray = curr_gray.copy()
        
        # Sanity check
        if prev_pts.shape != curr_pts.shape:
            prev_pts = curr_pts

        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        if curr_pts.shape[0] >= 4:
            tol = 2
            H, status = cv2.findHomography(prev_pts, curr_pts, cv2.LMEDS, tol)
        else:
            H = np.eye(3)

        return H, prev_pts, curr_pts


# a feature matching based method.  This can be more robust with
# difficult situations like a spinning propellor dominating the view,
# but tracked feature distribution can be poor depending on the scene
# composition.

class myFeatureFlow():
    def __init__(self, K):
        self.K = K
        self.match_ratio = 0.75
        self.max_features = 500
        self.kp_list_last = []
        self.des_list_last = []

        if True:
            # ORB (much faster)
            self.detector = cv2.ORB_create(self.max_features)
            self.extractor = self.detector
            norm = cv2.NORM_HAMMING
            self.matcher = cv2.BFMatcher(norm)
        else:
            # SIFT (in sparse cases can find better features, but a
            # lot slower)
            self.detector = cv2.SIFT_create(nfeatures=self.max_features,
                                            nOctaveLayers=5)
            self.extractor = self.detector
            norm = cv2.NORM_L2
            FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
            FLANN_INDEX_LSH    = 6
            flann_params = { 'algorithm': FLANN_INDEX_KDTREE,
                             'trees': 5 }
            self.matcher = cv2.FlannBasedMatcher(flann_params, {}) # bug : need to pass empty dict (#1329)
            
    def filterMatches(self, kp1, kp2, matches):
        mkp1, mkp2 = [], []
        idx_pairs = []
        used = np.zeros(len(kp2), np.bool_)
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.match_ratio:
                #print " dist[0] = %d  dist[1] = %d" % (m[0].distance, m[1].distance)
                m = m[0]
                # FIXME: ignore the bottom section of movie for feature detection
                #if kp1[m.queryIdx].pt[1] > h*0.75:
                #    continue
                if not used[m.trainIdx]:
                    used[m.trainIdx] = True
                    mkp1.append( kp1[m.queryIdx] )
                    mkp2.append( kp2[m.trainIdx] )
                    idx_pairs.append( (m.queryIdx, m.trainIdx) )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, kp_pairs, idx_pairs

    def filterFeatures(self, p1, p2, K, method):
        inliers = 0
        total = len(p1)
        space = ""
        status = []
        M = None
        if len(p1) < 7:
            # not enough points
            return None, np.zeros(total), [], []
        if method == 'homography':
            #M, status = cv2.findHomography(p1, p2, cv2.RANSAC, self.tol)
            M, status = cv2.findHomography(p1, p2, cv2.LMEDS, self.tol)
        elif method == 'fundamental':
            M, status = cv2.findFundamentalMat(p1, p2, cv2.RANSAC, self.tol)
        elif method == 'essential':
            M, status = cv2.findEssentialMat(p1, p2, K, cv2.LMEDS, prob=0.99999, threshold=self.tol)
        elif method == 'none':
            M = None
            status = np.ones(total)
        newp1 = []
        newp2 = []
        for i, flag in enumerate(status):
            if flag:
                newp1.append(p1[i])
                newp2.append(p2[i])
        inliers = np.sum(status)
        total = len(status)
        #print('%s%d / %d  inliers/matched' % (space, np.sum(status), len(status)))
        return M, status, np.float32(newp1), np.float32(newp2)

    def update(self, frame):
        # convert to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        self.tol = gray.shape[1] / 300.0
        if self.tol < 1.0: self.tol = 1.0

        kp_list = self.detector.detect(gray)
        kp_list, des_list = self.extractor.compute(gray, kp_list)
        
        # Fixme: make configurable?
        # possible values are "homography", "fundamental", "essential", "none"
        filter_method = "homography"

        if self.des_list_last is None or des_list is None or len(self.des_list_last) == 0 or len(des_list) == 0:
            self.kp_list_last = kp_list
            self.des_list_last = des_list

        #print(len(self.des_list_last), len(des_list))
        matches = self.matcher.knnMatch(des_list, trainDescriptors=self.des_list_last, k=2)
        p1, p2, kp_pairs, idx_pairs = self.filterMatches(kp_list, self.kp_list_last, matches)
        self.kp_list_last = kp_list
        self.des_list_last = des_list

        M, status, newp1, newp2 = self.filterFeatures(p1, p2, self.K, filter_method)
        if len(newp1) < 1:
            M = np.eye(3)

        print("M:\n", M)
        return M, newp1, newp2
