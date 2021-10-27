#!/usr/bin/env python3

print("FYI: so far, this script does /not/ work as intended.")

import argparse
import csv
import cv2
import skvideo.io               # pip3 install sk-video
import json
import math
import numpy as np
import os
from tqdm import tqdm
import time

from props import PropertyNode
import props_json

import sys
sys.path.append("../video")
import camera

from motion import myOpticalFlow

parser = argparse.ArgumentParser(description='Track motion with homography transformation.')
parser.add_argument('video', help='video file')
parser.add_argument('--camera', help='select camera calibration file')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--skip-frames', type=int, default=0, help='skip n initial frames')
parser.add_argument('--collect-frames', type=int, default=0, help='collect n frames')
parser.add_argument('--write', action='store_true', help='write out video with keypoints shown')
args = parser.parse_args()

scale = args.scale

# pathname work
abspath = os.path.abspath(args.video)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.video)
bg_video = filename + "_bg.mp4"
motion_video = filename + "_motion.mp4"
local_config = os.path.join(dirname, "camera.json")

camera = camera.VirtualCamera()
camera.load(args.camera, local_config, args.scale)
K = camera.get_K()
IK = camera.get_IK()
dist = camera.get_dist()
print('Camera:', camera.get_name())
print('K:\n', K)
print('IK:\n', IK)
print('dist:', dist)

metadata = skvideo.io.ffprobe(args.video)
#print(metadata.keys())
print(json.dumps(metadata["video"], indent=4))
fps_string = metadata['video']['@avg_frame_rate']
(num, den) = fps_string.split('/')
fps = float(num) / float(den)
codec = metadata['video']['@codec_long_name']
w = int(round(int(metadata['video']['@width']) * scale))
h = int(round(int(metadata['video']['@height']) * scale))
if "@duration" in metadata["video"]:
    total_frames = int(round(float(metadata['video']['@duration']) * fps))
else:
    total_frames = 1

print('fps:', fps)
print('codec:', codec)
print('output size:', w, 'x', h)
print('total frames:', total_frames)

mind = w
if mind < h: mind = h
fx = mind * args.scale * 0.9
cu = w * 0.5
cv = h * 0.5
K = np.matrix( [ [fx, 0, cu],
                 [0, fx, cv],
                 [0, 0, 1] ] )
dist = np.zeros(5)
#dist = np.array( [         -0.26910665,        0.10580125,        0.0,        0.0,        -0.02321387    ] )
print("K:\n", K)
print("dist:", dist)

print("Opening ", args.video)
reader = skvideo.io.FFmpegReader(args.video, inputdict={}, outputdict={})

if args.write:
    inputdict = {
        '-r': str(fps)
    }
    lossless = {
        # See all options: https://trac.ffmpeg.org/wiki/Encode/H.264
        '-vcodec': 'libx264',  # use the h.264 codec
        '-crf': '0',           # set the constant rate factor to 0, (lossless)
        '-preset': 'veryslow', # maximum compression
        '-r': str(fps)         # match input fps
    }
    sane = {
        # See all options: https://trac.ffmpeg.org/wiki/Encode/H.264
        '-vcodec': 'libx264',  # use the h.264 codec
        '-crf': '17',          # visually lossless (or nearly so)
        '-preset': 'medium',   # default compression
        '-r': str(fps)         # match input fps
    }
    motion_writer = skvideo.io.FFmpegWriter(motion_video, inputdict=inputdict, outputdict=sane)
    bg_writer = skvideo.io.FFmpegWriter(bg_video, inputdict=inputdict, outputdict=sane)

flow = myOpticalFlow()

counter = -1
pairs = []

coll_num = 0
# build a list of feature matched pairs using the optical flow
# algorithm which generally works really well for video.

frame_scale = None

pbar = tqdm(total=int(total_frames), smoothing=0.05)
for frame in reader.nextFrame():
    counter += 1
    if counter < args.skip_frames:
        continue

    if counter % 5 != 0:
        continue
    
    if args.collect_frames:
        if coll_num > args.collect_frames:
            break

    coll_num += 1
    
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    #if counter % 2 != 0:
    #    continue
    
    frame_scale = cv2.resize(frame, (0,0), fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
    cv2.imshow('scaled orig', frame_scale)

    # update the flow estimate
    M, prev_pts, curr_pts = flow.update(frame_scale)
    print("M:\n", M)

    pairs.append( [prev_pts, curr_pts] )
    
    if True:
        for pt in curr_pts:
            cv2.circle(frame_scale, (int(pt[0][0]), int(pt[0][1])), 3, (0,255,0), 1, cv2.LINE_AA)
        for pt in prev_pts:
            cv2.circle(frame_scale, (int(pt[0][0]), int(pt[0][1])), 2, (0,0,255), 1, cv2.LINE_AA)

    cv2.imshow('features', frame_scale)

    # highlight = frame_scale.astype('float32') + 2*cv2.merge((diff, diff, diff))
    # cv2.imshow("highlight", (255*highlight.astype('float32')/np.max(highlight)).astype('uint8'))
    
    if 0xFF & cv2.waitKey(1) == 27:
        break
    pbar.update(1)
pbar.close()

cv2.destroyAllWindows()

from scipy.optimize import least_squares

def errorFunc(xk):
    print("  trying:", xk)
    result = []
    for pair in pairs[1:]:
        prev_pts = pair[0]
        curr_pts = pair[1]
        #print("prev:", prev_pts)
        #print("curr:", curr_pts)
        if prev_pts.shape[0] < 4 or curr_pts.shape[0] < 4:
            continue
        dist = np.array( [xk[0], xk[1], 0, 0, xk[2]] )
        prev_undist = cv2.undistortPoints( prev_pts, K, dist )
        curr_undist = cv2.undistortPoints( curr_pts, K, dist )
        prev_undist[:,:,0] *= fx
        prev_undist[:,:,0] += cu
        prev_undist[:,:,1] *= fx
        prev_undist[:,:,1] += cv
        curr_undist[:,:,0] *= fx
        curr_undist[:,:,0] += cu
        curr_undist[:,:,1] *= fx
        curr_undist[:,:,1] += cv
        # try rescaling
        uscale = np.max(np.abs(prev_undist[:,:,0]))
        vscale = np.max(np.abs(prev_undist[:,:,1]))
        prev_undist[:,:,0] *= w / uscale
        prev_undist[:,:,1] *= h / vscale
        curr_undist[:,:,0] *= w / uscale
        curr_undist[:,:,1] *= h / vscale
        #print("prev_undist:", prev_undist.shape, prev_undist)
        #print("curr_undist:", curr_undist.shape, curr_undist)
        H, status = cv2.findHomography(prev_undist, curr_undist, cv2.RANSAC)
        #print("H:\n", H)
        if H is not None:
            prev_trans = cv2.perspectiveTransform(prev_undist, H)
            # print(prev_trans - curr_undist)
            error = curr_undist - prev_trans
            #print("error:", error.shape)
            #print("error:", error.shape, error)
            for i in range(error.shape[0]):
                if status[i]:
                    result.append(np.linalg.norm(error[i]))
                else:
                    result.append(0)
                    #print(i, prev_undist[i], prev_trans[i], curr_undist[i], error[i])
            if False:
                frame = np.zeros_like(frame_scale)
                for pt in curr_undist:
                    cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 2, (0,255,0), 1, cv2.LINE_AA)
                #for pt in prev_undist:
                #    cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 3, (0,0,255), 1, cv2.LINE_AA)
                for pt in prev_trans:
                    cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 4, (255,0,0), 1, cv2.LINE_AA)
                cv2.imshow("frame", frame)
                cv2.waitKey(1)

    cv2.imshow('features', frame_scale)
    return np.array(result)

#print("Starting error:", errorFunc(np.zeros(3)))
print("Optimizing...")
res = least_squares(errorFunc, np.zeros(3), verbose=2)
print(res)
print(res['x'])
