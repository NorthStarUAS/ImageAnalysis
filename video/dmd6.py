#!/usr/bin/env python3

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

import camera
from motion import myOpticalFlow
#from motion import myFarnebackFlow

parser = argparse.ArgumentParser(description='Track motion with homography transformation.')
parser.add_argument('video', help='video file')
parser.add_argument('--camera', help='select camera calibration file')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--skip-frames', type=int, default=0, help='skip n initial frames')
parser.add_argument('--write', action='store_true', help='write out video with keypoints shown')
args = parser.parse_args()

scale = args.scale

# pathname work
abspath = os.path.abspath(args.video)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.video)
bg_video = filename + "_bg.mp4"
motion_video = filename + "_motion.mp4"
feat_video = filename + "_feat.mp4"
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
    feat_writer = skvideo.io.FFmpegWriter(feat_video, inputdict=inputdict, outputdict=sane)

flow = myOpticalFlow()
#farneback = myFarnebackFlow()

prev_filt = np.array( [] )
curr_filt = np.array( [] )
bg_filt = np.array( [] )
kp_list_last = []
des_list_last = []
p1 = []
p2 = []
counter = -1
warp_flags = cv2.INTER_LANCZOS4
diff_factor = 255

pbar = tqdm(total=int(total_frames), smoothing=0.05)
for frame in reader.nextFrame():
    counter += 1
    if counter < args.skip_frames:
        continue
    
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    #if counter % 2 != 0:
    #    continue
    
    frame_scale = cv2.resize(frame, (0,0), fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
    cv2.imshow('scaled orig', frame_scale)
    frame_undist = cv2.undistort(frame_scale, K, np.array(dist))
    cv2.imshow("frame undist", frame_undist)

    # update the flow estimate
    M, prev_pts, curr_pts = flow.update(frame_undist)
    print("M:\n", M)
    
    #farneback.update(frame_undist)
    
    if M is None or prev_filt.shape[0] == 0 or curr_filt.shape[0] == 0:
        prev_filt = frame_undist.copy().astype('float32')
        curr_filt = frame_undist.copy().astype('float32')
        diff = cv2.absdiff(prev_filt, curr_filt)
        bg_filt = frame_undist.copy().astype('float32')
    else:
        prev_proj = frame_undist.copy()
        curr_proj = frame_undist.copy()
        bg_proj = frame_undist.copy()
        prev_proj = cv2.warpPerspective(prev_filt.astype('uint8'), M, (frame_undist.shape[1], frame_undist.shape[0]), prev_proj, flags=warp_flags, borderMode=cv2.BORDER_TRANSPARENT)
        curr_proj = cv2.warpPerspective(curr_filt.astype('uint8'), M, (frame_undist.shape[1], frame_undist.shape[0]), curr_proj, flags=warp_flags, borderMode=cv2.BORDER_TRANSPARENT)
        bg_proj = cv2.warpPerspective(bg_filt.astype('uint8'), M, (frame_undist.shape[1], frame_undist.shape[0]), bg_proj, flags=warp_flags, borderMode=cv2.BORDER_TRANSPARENT)
        curr_filt = cv2.addWeighted(curr_proj.astype('float32'), 0.5, frame_undist.astype('float32'), 0.5, 0)
        cv2.imshow("prev_filt", prev_filt.astype('uint8'))
        cv2.imshow("curr_filt", curr_filt.astype('uint8'))
        diff = cv2.absdiff(prev_proj.astype('uint8'), curr_filt.astype('uint8'))
        bg_filt = 0.95*bg_proj + 0.05*frame_undist
        prev_filt = cv2.addWeighted(prev_proj.astype('float32'), 0.5, frame_undist.astype('float32'), 0.5, 0)
    diff_max = np.max(diff)
    diff_factor = 0.95*diff_factor + 0.05*diff_max
    if diff_factor < diff_max:
        diff_factor = diff_max
    print("diff_factor:", diff_factor)
    diff_img = (255*diff.astype('float32')/diff_factor).astype('uint8')
    cv2.imshow("diff", diff_img.astype('uint8'))
    cv2.imshow("background", bg_filt.astype('uint8'))

    if True:
        for pt in curr_pts:
            cv2.circle(frame_undist, (int(pt[0][0]), int(pt[0][1])), 3, (0,255,0), 1, cv2.LINE_AA)
        for pt in prev_pts:
            cv2.circle(frame_undist, (int(pt[0][0]), int(pt[0][1])), 2, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow('features', frame_undist)

    if args.write:
        # if rgb
        motion_writer.writeFrame(diff_img[:,:,::-1])
        bg_writer.writeFrame(bg_filt[:,:,::-1])
        feat_writer.writeFrame(frame_undist[:,:,::-1])
        # if gray
        #motion_writer.writeFrame(diff_img)
        #bg_writer.writeFrame(prev_filt)
    
    if 0xFF & cv2.waitKey(1) == 27:
        break
    pbar.update(1)
pbar.close()

cv2.destroyAllWindows()

