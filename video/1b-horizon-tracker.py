#!/usr/bin/python3

import argparse
import csv
import cv2
import skvideo.io               # pip3 install sk-video
import json
import math
import numpy as np
import os
import sys

from props import PropertyNode
import props_json

sys.path.append('../scripts')
from lib import transformations

d2r = math.pi / 180.0
r2d = 180.0 / math.pi

match_ratio = 0.75
max_features = 500
smooth = 0.005
catchup = 0.02
affine_minpts = 7
tol = 2.0

parser = argparse.ArgumentParser(description='Estimate gyro biases from movie.')
parser.add_argument('video', help='video file')
parser.add_argument('--camera', help='select camera calibration file')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--skip-frames', type=int, default=0, help='skip n initial frames')
parser.add_argument('--write', action='store_true', help='write out the final video')
args = parser.parse_args()

#file = args.video
scale = args.scale
skip_frames = args.skip_frames

# pathname work
abspath = os.path.abspath(args.video)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.video)
output_csv = filename + ".csv"
output_avi = filename + "_horiz.avi"
local_config = os.path.join(dirname, "camera.json")

config = PropertyNode()

if args.camera:
    # seed the camera calibration and distortion coefficients from a
    # known camera config
    print('Setting camera config from:', args.camera)
    props_json.load(args.camera, config)
    config.setString('name', args.camera)
    props_json.save(local_config, config)
elif os.path.exists(local_config):
    # load local config file if it exists
    props_json.load(local_config, config)
    
name = config.getString('name')
cam_yaw = config.getFloatEnum('mount_ypr', 0)
cam_pitch = config.getFloatEnum('mount_ypr', 1)
cam_roll = config.getFloatEnum('mount_ypr', 2)
K_list = []
for i in range(9):
    K_list.append( config.getFloatEnum('K', i) )
K = np.copy(np.array(K_list)).reshape(3,3)
dist = []
for i in range(5):
    dist.append( config.getFloatEnum("dist_coeffs", i) )

print('Camera:', name)
print('K:\n', K)
print('dist:', dist)

K = K * args.scale
K[2,2] = 1.0
cu = K[0,2]
cv = K[1,2]
IK = np.linalg.inv(K)

metadata = skvideo.io.ffprobe(args.video)
#print(metadata.keys())
print(json.dumps(metadata["video"], indent=4))
fps_string = metadata['video']['@avg_frame_rate']
(num, den) = fps_string.split('/')
fps = float(num) / float(den)
codec = metadata['video']['@codec_long_name']
w = int(round(int(metadata['video']['@width']) * scale))
h = int(round(int(metadata['video']['@height']) * scale))
total_frames = int(round(float(metadata['video']['@duration']) * fps))

print('fps:', fps)
print('codec:', codec)
print('output size:', w, 'x', h)
print('total frames:', total_frames)

print("Opening ", args.video)
reader = skvideo.io.FFmpegReader(args.video, inputdict={}, outputdict={})

if args.write:
    #outfourcc = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
    outfourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #outfourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')
    #outfourcc = cv2.cv.CV_FOURCC('X', '2', '6', '4')
    #outfourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(output_avi, outfourcc, fps, (w, h))

def ClosestPointOnLine(a, b, p):
    ap = p - a
    ab = b - a
    return a + np.dot(ap,ab) / np.dot(ab,ab) * ab

# locate horizon and estimate relative roll/pitch of the camera
def horizon(frame):
    # attempt to threshold on high blue values (blue<->white)
    b, g, r = cv2.split(frame)
    cv2.imshow("b", b)
    #cv2.imshow("g", g)
    #cv2.imshow("r", r)
    #print("shape:", frame.shape)
    #print("blue range:", np.min(b), np.max(b))
    #print('ave:', np.average(b), np.average(g), np.average(r))

    # Otsu thresholding on blue channel
    ret2, thresh = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print('ret2:', ret2)
    cv2.imshow('otsu mask', thresh)

    # dilate the mask a small bit
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
  
    # global thresholding on blue channel before edge detection
    #thresh = cv2.inRange(frame, (210, 0, 0), (255, 255, 255))
    #cv2.imshow('global mask', thresh)
        
    preview = cv2.bitwise_and(frame, frame, mask=thresh)
    cv2.imshow("threshold", preview)

    # the lower the 1st canny number the more total edges are accepted
    # the lower the 2nd canny number the less hard the edges need to
    # be to accept an edge
    edges = cv2.Canny(b, 50, 150)
    #edges = cv2.Canny(b, 200, 600)
    cv2.imshow("edges", edges)

    # Use the blue mask (Otsu) to filter out edge noise in area we
    # don't care about (to improve performance of the hough transform)
    edges = cv2.bitwise_and(edges, edges, mask=thresh)
    cv2.imshow("masked edges", edges)
    
    #theta_res = np.pi/180       # 1 degree
    theta_res = np.pi/1800      # 0.1 degree
    threshold = int(frame.shape[1] / 8)
    if True:
        # Standard hough transform.  Presumably returns lines sorted
        # by most dominant first
        lines = cv2.HoughLines(edges, 1, theta_res, threshold)
        for line in lines[:1]:  # just the 1st/most dominant
            print(line[0])
            rho, theta = line[0]
            #print("theta:", theta * r2d)
            roll = 90 - theta*r2d
            # this will be wrong, but maybe approximate right a little bit
            if np.abs(theta) > 0.00001:
                m = -(np.cos(theta) / np.sin(theta))
                b = rho / np.sin(theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            print("p0:", x0, y0)
            len2 = 1000
            x1 = int(x0 + len2*(-b))
            y1 = int(y0 + len2*(a))
            x2 = int(x0 - len2*(-b))
            y2 = int(y0 - len2*(a))
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),2,cv2.LINE_AA)
            p0 = ClosestPointOnLine(np.array([x1,y1]),
                                    np.array([x2,y2]),
                                    np.array([cu,cv]))
            uvh = np.array([p0[0], p0[1], 1.0])
            proj = IK.dot(uvh)
            #print("proj:", proj, proj/np.linalg.norm(proj))
            dot_product = np.dot(np.array([0,0,1]), proj/np.linalg.norm(proj))
            pitch = np.arccos(dot_product) * r2d
            if p0[1] < cv:
                pitch = -pitch
            print("roll: %.1f pitch: %.1f" % (roll, pitch))
            cv2.circle(frame,(int(p0[0]), int(p0[1])),10,(255,0,255), 2, cv2.LINE_AA)
            cv2.line(frame,(int(p0[0]), int(p0[1])), (int(cu),int(cv)),(255,0,255),1,cv2.LINE_AA)
            
    else:
        # probabalistic hough transform (faster?)
        lines = cv2.HoughLinesP(edges, 1, theta_res, threshold, maxLineGap=50)
        if not lines is None:
            for line in lines[:1]:
                for x0,y0,x1,y1 in line:
                    #print("theta:", theta * r2d, "roll:", 90 - theta * r2d)
                    # this will be wrong, but maybe approximate right a little bit
                    #if np.abs(theta) > 0.00001:
                    #    m = -(np.cos(theta) / np.sin(theta))
                    #    b = rho / np.sin(theta)
                    #a = np.cos(theta)
                    #b = np.sin(theta)
                    #x0 = a*rho
                    #y0 = b*rho
                    #print("p0:", x0, y0)
                    #len2 = 1000
                    #x1 = int(x0 + len2*(-b))
                    #y1 = int(y0 + len2*(a))
                    #x2 = int(x0 - len2*(-b))
                    #y2 = int(y0 - len2*(a))
                    cv2.line(frame,(x0,y0),(x1,y1),(255,0,255),2,cv2.LINE_AA)
        
    cv2.imshow("horizon", frame)
        
counter = 0

for frame in reader.nextFrame():
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    counter += 1

    filtered = []

    if counter < skip_frames:
        if counter % 100 == 0:
            print("Skipping %d frames..." % counter)
        else:
            continue

    print("Frame %d" % counter)

    method = cv2.INTER_AREA
    #method = cv2.INTER_LANCZOS4
    frame_scale = cv2.resize(frame, (0,0), fx=scale, fy=scale,
                             interpolation=method)
    cv2.imshow('scaled orig', frame_scale)
    shape = frame_scale.shape
    tol = shape[1] / 100.0
    if tol < 1.0: tol = 1.0

    distort = True
    if distort:
        frame_undist = cv2.undistort(frame_scale, K, np.array(dist))
    else:
        frame_undist = frame_scale    

    # test horizon detection
    horizon(frame_undist)
    
    if args.write:
        output.write(frame_undist)
    if 0xFF & cv2.waitKey(5) == 27:
        break

cv2.destroyAllWindows()

