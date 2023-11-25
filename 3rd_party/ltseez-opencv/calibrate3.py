#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Compute camera/lens calibration from a set of chessboard images')
parser.add_argument('--debug_dir', help='path to a directory for debug images')
parser.add_argument('--square_size', default=1.0, type=float, help='square size')
parser.add_argument('files', metavar='file', nargs='+', help="list of images with chessboard pattern.")
args = parser.parse_args()

# number of interior corners (i.e. one less that number of grid cells in each
# direction)

# pattern_size = (9, 7)
# pattern_size = (8, 6)
pattern_size = (11, 8)

pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= args.square_size

obj_points = []
img_points = []
h, w = 0, 0
print("Loading images and finding chessboard ...")
for fn in tqdm(args.files, smoothing=0.05):
    # print('processing %s...' % fn,)
    img = cv2.imread(fn, flags=cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        print("Failed to load", fn)
        continue

    h, w = img.shape[:2]
    # print(w, h)
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    if not found:
        print(" ", fn, "...chessboard not found")
        continue
    term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
    cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    if args.debug_dir:
        if not os.path.exists(args.debug_dir):
            os.makedirs(args.debug_dir)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        path, basefile = os.path.split(fn)
        name, ext = os.path.splitext(basefile)
        outfile = os.path.join(args.debug_dir, "%s_chess.jpg" % name)
        cv2.imwrite(outfile, vis)
    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)

print("")
print("Running optimizer on %d images ..." % len(img_points))
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

np.set_printoptions(suppress=True)
print("RMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())
cv2.destroyAllWindows()

if args.debug_dir:
    print("")
    print("writing undistorted version of images...")
    for fn in tqdm(args.files, smoothing=0.05):
        img = cv2.imread(fn, 0)
        undist = cv2.undistort(img, camera_matrix, dist_coefs.ravel())
        path, basefile = os.path.split(fn)
        name, ext = os.path.splitext(basefile)
        outfile = os.path.join(args.debug_dir, "%s_undistort.jpg" % name)
        cv2.imwrite(outfile, undist)
