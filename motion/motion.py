"""
@author: clolson / curt@flightgear.org
"""

# python module to package several optical flow tracking strategies.
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

class SparseLK():
    def __init__(self, use_mask=True, winSize=None, add_feature_interval=1):
        print("init winsize:", winSize)
        print("use mask:", use_mask)
        if winSize is not None:
            self.winSize = (winSize, winSize)
        else:
            self.winSize = None
        self.use_mask = use_mask
        self.add_feature_interval = int(round(add_feature_interval))
        if self.add_feature_interval < 1:
            self.add_feature_interval = 1
        self.prev_gray = None
        self.frame_count = 0
        self.num_new_pts = 0

    def make_mask(self, pts, shape, radius):
        mask = 255 * np.ones( shape ).astype('uint8')
        # print(radius, mask.shape
        # print(type(pts), pts.shape, pts)
        if len(pts):
            for i in range(len(pts)):
                # print("mask point:", pts[i][0])
                cv2.circle(mask, tuple( pts[i].reshape(2).astype('int') ),
                           radius, (0,0,0), -1, cv2.LINE_AA)
        # cv2.imshow("mask", mask)
        return mask

    def update(self, frame, mask_pts=None):
        if len(frame.shape) == 3:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = frame

        # prime the pump if needed
        if self.prev_gray is None:
            self.prev_gray = curr_gray.copy()

        if False:
            kernel = cv2.getGaussianKernel(15,1)
            # kernel = np.ones((5,5), np.uint8)
            closing = cv2.morphologyEx(curr, cv2.MORPH_CLOSE, kernel)
            cv2.imshow("closing", closing)
            opening = cv2.morphologyEx(curr, cv2.MORPH_OPEN, kernel)
            cv2.imshow("opening", opening)
            cmo = closing - opening
            cv2.imshow("CMO", cmo)
            curr = cmo.astype("uint8")

        # scale parameters with frame size
        diag = np.linalg.norm(frame.shape[:2])
        maxc = int(diag / 5)
        if maxc < 200: maxc = 200
        mind = int(diag / 30)
        if mind < 20: mind = 20
        bsize = int(diag / 100)
        if bsize < 5: bsize = 5
        # print("maxc:", maxc, "mask_pts:", mask_pts.shape)

        # draw.show_points(self.prev_gray, "self.prev_gray", mask_pts)
        if self.frame_count % self.add_feature_interval == 0 or len(mask_pts) < int(maxc/10):
            # draw.show_points(self.prev_gray, "mask points", mask_pts)
            # cv2.waitKey()
            if self.use_mask:
                # Detect feature points in previous frame
                # print(mask)
                # print(mask.shape)
                mask = self.make_mask(mask_pts, curr_gray.shape[:2], int(mind*0.75))
                # cv2.imshow("mask", mask)
                # cv2.waitKey()
            else:
                mask = None

            # print("self.prev_gray:", self.prev_gray)
            # if self.prev_gray is not None:
                # print("self.prev_gray:", self.prev_gray.shape)
            # print("mask:", mask.shape)
            # print("minDistance:", mind)
            new_prev_pts = cv2.goodFeaturesToTrack( self.prev_gray,
                                                    maxCorners=maxc,
                                                    qualityLevel=0.01,
                                                    minDistance=mind,
                                                    blockSize=bsize,
                                                    mask=mask )
            # draw.show_points(self.prev_gray, "new_prev_pts", new_prev_pts)
        else:
            new_prev_pts = None
            mask = None

        if self.use_mask and len(mask_pts):
            if new_prev_pts is None:
                prev_pts = mask_pts
                self.num_new_pts = 0
            else:
                num_new = maxc - len(mask_pts)
                prev_pts = np.concatenate( ( mask_pts.reshape(-1,1,2), new_prev_pts[:num_new,:,:]) )
                self.num_new_pts = len(new_prev_pts[:num_new,:,:])
        else:
            prev_pts = new_prev_pts
            self.num_new_pts = 0
        # print("prev_pts:", prev_pts)

        # compute the optical flow
        if prev_pts is not None:
            if True:
                # print("self.prev_gray:", self.prev_gray.shape, "curr_frame:", curr_frame.shape)
                if self.winSize is None:
                    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts.astype("float32"), None)
                else:
                    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts, None, winSize=self.winSize)
            else:
                print("ERROR: calcOpticalFlowPyrLK failed!!!")
                prev_pts = np.zeros(0)
                curr_pts = np.zeros(0)
                status = np.zeros(0)
        else:
            prev_pts = np.zeros(0)
            curr_pts = np.zeros(0)
            status = np.zeros(0)

        self.prev_gray = curr_gray.copy()

        # draw.show_points(curr_frame, "raw flow", curr_pts)
        # cv2.waitKey()
        # print(prev_pts)
        # print(self.prev_gray.shape, curr_frame.shape)
        self.frame_count += 1

        # Sanity check
        if prev_pts.shape != curr_pts.shape:
            prev_pts = curr_pts

        # return only the points where the algorithm found flow
        idx = np.where(status==1)[0]
        print("flow:", "%.1f%%" % (100*len(idx)/maxc), len(idx), maxc)
        return prev_pts[idx].reshape(-1,1,2), curr_pts[idx].reshape(-1,1,2), mask

class myOpticalFlow():
    def __init__(self):
        self.prev_gray = np.zeros(0)

    def update(self, frame, mask=None):
        # convert to gray scale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # prime the pump if needed
        if self.prev_gray.shape[0] == 0:
            self.prev_gray = curr_gray.copy()

        # scale parameters with frame size
        diag = np.linalg.norm(frame.shape[:2])
        maxc = int(diag / 5)
        if maxc < 200: maxc = 200
        mind = int(diag / 30)
        if mind < 30: mind = 30
        bsize = int(diag / 300)
        if bsize < 3: bsize = 3

        # Detect feature points in previous frame
        #print(mask)
        #print(mask.shape)
        #print(self.prev_gray)
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray,
                                           maxCorners=maxc,
                                           qualityLevel=0.01,
	                                       minDistance=mind,
                                           blockSize=bsize,
                                           mask=mask)

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
            tol = 1
            H, status = cv2.findHomography(prev_pts, curr_pts, cv2.LMEDS, tol)
        else:
            H = np.eye(3)

        return H, prev_pts, curr_pts, status


# a feature matching based method.  This can be more robust with
# difficult situations like a spinning propellor dominating the view,
# but tracked feature distribution can be poor depending on the scene
# composition.

class SparseSIFT():
    def __init__(self, K):
        self.K = K
        self.match_ratio = 0.75
        self.max_features = 500
        self.kp_list_last = []
        self.des_list_last = []

        if False:
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
        return np.linalg.inv(M), newp1, newp2, status



class DenseFarneback():
    def __init__(self, steps=20):
        self.steps = steps
        self.prev_gray = np.zeros(0)
        self.hsv_mask = np.zeros(0)
        self.avg = np.zeros(0)
        self.fm = np.zeros( (steps*steps, 2) )

    def sample_flow(self, array):
        rows, cols = array.shape[:2]
        rstep = rows // self.steps + 1
        cstep = cols // self.steps + 1
        result = []
        prev_pts = []
        curr_pts = []
        status = []
        count = 0
        for i in range(0, rows, rstep):
            for j in range(0, cols, cstep):
                #print(i,j, rstep, cstep)
                A = array[i:i+rstep, j:j+cstep, :]
                #print(A.shape)
                p = np.array([j+cstep//2, i+rstep//2])
                delta = np.array( [ np.median(A[:,:,0]), np.median(A[:,:,1]) ] )
                if np.linalg.norm(delta) > 1.0:
                    # build faster
                    self.fm[count] = 0.9 * self.fm[count] + 0.1 * delta
                else:
                    # decay slower
                    self.fm[count] = 0.98 * self.fm[count] + 0.02 * delta
                if np.linalg.norm(self.fm[count]) > 1.0:
                    curr_pts.append( p )
                    prev_pts.append( p - self.fm[count])
                    status.append( True )
                count += 1
        #print("sample:", prev_pts, curr_pts, status)
        return np.array(prev_pts).reshape(-1,1,2), np.array(curr_pts).reshape(-1,1,2), np.array(status)

    def update(self, frame):
        # convert to gray scale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # prime the pump if needed
        if self.prev_gray.shape[0] == 0:
            self.prev_gray = curr_gray.copy()

        # Create mask
        if self.hsv_mask.shape[0] == 0:
            self.hsv_mask = np.zeros_like(frame)
            # Make image saturation to a maximum value
            self.hsv_mask[..., 1] = 255

        # calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute magnitude and angle of 2D vector
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Set image hue value according to the angle of optical flow
        self.hsv_mask[..., 0] = ang * 180 / np.pi / 2

        # Set value as per the normalized magnitude of optical flow
        #print(np.min(mag), np.max(mag), np.mean(mag))
        if True:
            norm = mag * (255 / 40)
            norm[norm>255] = 255
        else:
            norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        self.hsv_mask[..., 2] = norm

        if True:
            rgb = cv2.cvtColor(self.hsv_mask.astype('uint8'), cv2.COLOR_HSV2BGR)
        else:
            if self.avg.shape[0] == 0:
                self.avg = self.hsv_mask.copy().astype('float32')
            else:
                self.avg = 0.9*self.avg + 0.1*self.hsv_mask

            # Convert to rgb
            rgb = cv2.cvtColor(self.avg.astype('uint8'), cv2.COLOR_HSV2BGR)
            #cv2.imshow('Farneback motion', rgb)

        self.prev_gray = curr_gray.copy()

        prev_pts, curr_pts, status = self.sample_flow(flow)
        return prev_pts, curr_pts, status, rgb

# project prev points onto the normal vector and the curr points onto
# the rotated normal vector.  if any lead to a negative projection,
# they are behind the plane and that solution is not the one want.
def filterHomographyByPoints(Rs, Ns, prev_pts, curr_pts, status):
    # fixme: look for better optimizations (numpy block ops)
    # fixme: this code is not 100% validated, use with caution
    result = []
    for num in range(len(Rs)):
        valid = True
        for i in range(len(prev_pts)):
            if not status[i]:
                continue
            p1 = np.append(prev_pts[i], 1.0)
            p2 = np.append(curr_pts[i], 1.0)
            pnd = np.dot(p1, Ns[num])           # prev dot norm
            cnd = np.dot(p2, Rs[num] @ Ns[num]) # curr dot rot(norm)
            #print(num, p1, pnd, " | ", p2, cnd)
            if pnd <= 0 or cnd <= 0:
            #   if pnd >= 0:
                print(pnd)
                valid = False
                break
        result.append(valid)
    return result


