# python module to package my favorite motion tracking tricks

# compute the homography matrix representing the best fit motion from
# the previous frame to the current frame.  The H matrix can further
# be decomposed into camera rotation and translation (requires the
# camera calibration matrix)

import cv2
import numpy as np

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
