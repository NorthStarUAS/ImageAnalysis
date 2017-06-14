#!/usr/bin/python

# scan a flight movie and track aruco markers to estimate
# twist rate and x, y deflection

# find our custom built opencv first
import sys
sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import argparse
import csv
import cv2
import cv2.aruco as aruco
import math
import numpy as np
import os
import time

from props import PropertyNode
import props_json

import cam_calib

d2r = math.pi / 180.0
r2d = 180.0 / math.pi

affine_minpts = 7

parser = argparse.ArgumentParser(description='Estimate gyro biases from movie.')
parser.add_argument('--movie', required=True, help='movie file')
parser.add_argument('--select-cam', type=int, help='select camera calibration')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--skip-frames', type=int, default=0, help='skip n initial frames')
parser.add_argument('--draw-keypoints', action='store_true', help='draw keypoints on output')
parser.add_argument('--draw-masks', action='store_true', help='draw stabilization masks')
parser.add_argument('--stop-count', type=int, default=1, help='how many non-frames to absorb before we decide the movie is over')
args = parser.parse_args()

#file = args.movie
scale = args.scale
skip_frames = args.skip_frames

# pathname work
abspath = os.path.abspath(args.movie)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.movie)
output_csv = filename + ".csv"
camera_config = dirname + "/camera.json"

# load config file if it exists
config = PropertyNode()
props_json.load(camera_config, config)
cam_yaw = config.getFloatEnum('mount_ypr', 0)
cam_pitch = config.getFloatEnum('mount_ypr', 1)
cam_roll = config.getFloatEnum('mount_ypr', 2)

# setup camera calibration and distortion coefficients
if args.select_cam:
    # set the camera calibration from known preconfigured setups
    name, K, dist = cam_calib.set_camera_calibration(args.select_cam)
    config.setString('name', name)
    config.setFloat("fx", K[0][0])
    config.setFloat("fy", K[1][1])
    config.setFloat("cu", K[0][2])
    config.setFloat("cv", K[1][2])
    for i, d in enumerate(dist):
        config.setFloatEnum("dist_coeffs", i, d)
    props_json.save(camera_config, config)
else:
    # load the camera calibration from the config file
    name = config.getString('name')
    size = config.getLen("dist_coeffs")
    dist = []
    for i in range(size):
        dist.append( config.getFloatEnum("dist_coeffs", i) )
    K = np.zeros( (3,3) )
    K[0][0] = config.getFloat("fx")
    K[1][1] = config.getFloat("fy")
    K[0][2] = config.getFloat("cu")
    K[1][2] = config.getFloat("cv")
    K[2][2] = 1.0
    print 'Camera:', name
    
K = K * args.scale
K[2,2] = 1.0

print "Opening ", args.movie
try:
    capture = cv2.VideoCapture(args.movie)
    #capture = cv2.VideoCapture() # webcam
except:
    print "error opening video"

print "ok opening video"
capture.read()
print "ok reading first frame"

fps = capture.get(cv2.CAP_PROP_FPS)
print "fps = %.2f" % fps
fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * scale )
h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale )

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters_create()

# produce a consistant ordering of corner points
def order_corner_points(corners, ids):
    num_markers = 4
    if len(corners) != num_markers or len(corners) != len(ids):
        print "error: need %d markers and ids" % num_markers
        return

    new_corners = []
    for id in range(num_markers):
        # find k = 1, 2, 3, 4 in ids in that order (ids is unordered)
        for i in range(num_markers):
            if id+1 == ids[i][0]:
                index = i
                break
        # print id, ids[index]
        c = corners[index]
        for p in c[0]:
            new_corners.append((p[0], p[1]))
    return new_corners
        
# find affine transform between matching keypoints in pixel
# coordinate space.  fullAffine=True means unconstrained to
# include best warp/shear.  fullAffine=False means limit the
# matrix to only best rotation, translation, and scale.
def findAffine(src, dst, fullAffine=False):
    #print "src = %s" % str(src)
    #print "dst = %s" % str(dst)
    if len(src) >= affine_minpts:
        affine = cv2.estimateRigidTransform(np.array([src]), np.array([dst]),
                                            fullAffine)
    else:
        affine = None
    #print str(affine)
    return affine

def decomposeAffine(affine):
    if affine == None:
        return (0.0, 0.0, 0.0, 1.0, 1.0)

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

points_ref = None
corners_ref = None
affine_last = None
kp_list_last = []
des_list_last = []
p1 = []
p2 = []
mkp1 = []
counter = 0

rot_avg = 0.0
tx_avg = 0.0
ty_avg = 0.0
sx_avg = 1.0
sy_avg = 1.0

rot_sum = 0.0
tx_sum = 0.0
ty_sum = 0.0
sx_sum = 1.0
sy_sum = 1.0

# umn test
abs_rot = 0.0
abs_x = 0.0
abs_y = 0.0

result = []
stop_count = 0

csvfile = open(output_csv, 'wb')
writer = csv.DictWriter(csvfile, fieldnames=['frame', 'rotation (deg)',
                                             'translation x (px)',
                                             'translation y (px)'])
writer.writeheader()

start = time.time()

while True:
    counter += 1

    ret, frame = capture.read()
    if not ret:
        # no frame
        stop_count += 1
        print "no more frames:", stop_count
        if stop_count > args.stop_count:
            break
    else:
        stop_count = 0    

    if counter < skip_frames:
        if counter % 1000 == 0:
            print "Skipping %d frames..." % counter
        continue

    if frame == None:
        print "Skipping bad frame ..."
        continue

    method = cv2.INTER_AREA
    #method = cv2.INTER_LANCZOS4
    frame_scale = cv2.resize(frame, (0,0), fx=scale, fy=scale,
                             interpolation=method)
    # cv2.imshow('scaled orig', frame_scale)

    distort = False
    if distort:
        frame_undist = cv2.undistort(frame_scale, K, np.array(dist))
    else:
        frame_undist = frame_scale    
        
    gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

    # aruco stuff
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if len(corners) == 4:
        points = order_corner_points(corners, ids)

        if points_ref == None:
            points_ref = points
            corners_ref = corners
        
        affine = findAffine(points, points_ref, fullAffine=False)
        (rot, tx, ty, sx, sy) = decomposeAffine(affine)

        row = {'frame': counter,
               'rotation (deg)': rot,
               'translation x (px)': tx,
               'translation y (px)': ty}
        writer.writerow(row)

        datapt = [ counter / fps, counter, -rot*fps*d2r, ty, tx ]
        result.append(datapt)

        img = aruco.drawDetectedMarkers(gray, corners_ref,
                                        borderColor=(256,0,0))
        img = aruco.drawDetectedMarkers(img, corners,
                                        borderColor=(128,0,0))
        cv2.imshow('aruco', img)
        if 0xFF & cv2.waitKey(5) == 27:
            break

        cur = time.time()
        elapsed = cur - start
        fps = counter / elapsed
        print "frame: %d fps: %.1f rot: %.2f x: %.1f y: %.1f" % (counter, fps, rot, tx, ty)

cv2.destroyAllWindows()
