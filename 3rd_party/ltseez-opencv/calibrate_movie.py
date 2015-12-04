#!/usr/bin/env python

# find our custom built opencv first
import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")


import argparse
import numpy as np
import cv2

# local modules
# from common import splitfn

# built-in modules
import os


parser = argparse.ArgumentParser(description='Estimate gyro biases from movie.')
parser.add_argument('--movie', required=True, help='movie file')
parser.add_argument('--square-size', type=float, default=1.0, help='square size')
parser.add_argument('--samples', type=int, default=100, help='samples to extract from movie')
parser.add_argument('--debug', action='store_true', help='draw debugging output')
args = parser.parse_args()


if __name__ == '__main__':
    import sys

    #pattern_size = (9, 6)
    pattern_size = (9, 6)
    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= args.square_size

    tmp_img_points_list = []
    obj_points = []
    img_points = []
    h, w = 0, 0

    print "ready to start ..."
    try:
        print "Opening ", args.movie
        capture = cv2.VideoCapture(args.movie)
    except:
        print "error opening video"
        quit()
    print "ok"
    tmp_image_points_list = []
    count = 0
    while True:
        ret, img = capture.read()
        if not ret:
            print "end of movie"
            break
        cv2.imshow('input', img)
        
        print 'processing frame:', count,
        count += 1
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if found:
            term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
        if args.debug:
            cv2.drawChessboardCorners(gray, pattern_size, corners, found)
            path, basefile = os.path.split(fn)
            name, ext = os.path.splitext(basefile)
            cv2.imshow('corners', gray)
        if not found:
            print 'chessboard not found'
            continue
        tmp_image_points_list.append(corners)
        #img_points.append(corners.reshape(-1, 2))
        #obj_points.append(pattern_points)

        if 0xFF & cv2.waitKey(5) == 27:
            break
        
        print 'ok'

    # select 'n=samples' of the found frames
    size = len(tmp_image_points_list)
    step = int( size / args.samples )
    if step < 1:
        step = 1
    for i in range(0, len(tmp_image_points_list), step):
        print i
        corners = tmp_image_points_list[i]
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)
    
    print np.array(img_points).size
    print "Computing camera calibration (this may take quite a bit of time)..."
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    np.set_printoptions(suppress=True)
    print "RMS:", rms
    print "camera matrix:\n", camera_matrix
    print "distortion coefficients: ", dist_coefs.ravel()
    cv2.destroyAllWindows()
    
