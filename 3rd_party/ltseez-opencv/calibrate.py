#!/usr/bin/env python3

import numpy as np
import cv2

# local modules
# from common import splitfn

# built-in modules
import os


USAGE = '''
USAGE: calibration.py [--save <filename>] [--debug <output path>] [--square_size] [<image mask>]
'''



if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['save=', 'debug=', 'square_size='])
    args = dict(args)
    try:
        img_mask = img_mask[0]
        print("img_mask:", img_mask)
    except:
        img_mask = '../data/left*.jpg'

    img_names = glob(img_mask)
    print("Image names: ", img_names)
    debug_dir = args.get('--debug')
    square_size = float(args.get('--square_size', 1.0))

    #pattern_size = (9, 7)
    #pattern_size = (8, 6)
    pattern_size = (11, 8)
    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    print("ready to start ...")
    for fn in img_names:
        print('processing %s...' % fn,)
        #img = cv2.imread(fn, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
        #img = cv2.imread(fn, 0)
        img = cv2.imread(fn, flags=cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
          print("Failed to load", fn)
          continue

        h, w = img.shape[:2]
        print(w, h)
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        if debug_dir:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, basefile = os.path.split(fn)
            name, ext = os.path.splitext(basefile)
            cv2.imwrite('%s/%s_chess.jpg' % (debug_dir, name), vis)
        if not found:
            print('  ...chessboard not found')
            continue
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        print('  ...ok')

    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    np.set_printoptions(suppress=True)
    print("RMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())
    cv2.destroyAllWindows()
    
    if debug_dir:
        for fn in img_names:
            img = cv2.imread(fn, 0)
            undist = cv2.undistort(img, camera_matrix, dist_coefs.ravel())
            path, basefile = os.path.split(fn)
            name, ext = os.path.splitext(basefile)
            cv2.imwrite('%s/%s_undistort.jpg' % (debug_dir, name), undist)
