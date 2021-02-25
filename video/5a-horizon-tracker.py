#!/usr/bin/env python3

import argparse
import csv
import cv2
import skvideo.io               # pip3 install sk-video
import json
import math
import numpy as np
import os

from props import PropertyNode
import props_json

import sys
sys.path.append('../scripts')
from lib import transformations

import camera
import horizon

# constants
d2r = math.pi / 180.0
r2d = 180.0 / math.pi

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
output_csv = filename + "_horiz.csv"
output_video = filename + "_horiz.mp4"
local_config = os.path.join(dirname, "camera.json")

camera = camera.VirtualCamera()
camera.load(args.camera, local_config, args.scale)
K = camera.get_K()
IK = camera.get_IK()
dist = camera.get_dist()
print('Camera:', camera.get_name())
print('K:\n', K)
print('dist:', dist)

cu = K[0,2]
cv = K[1,2]

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

csvfile = open(output_csv, 'w')
fieldnames = [ 'frame', 'video time',
               'camera roll (deg)', 'camera pitch (deg)',
               'roll rate (rad/sec)', 'pitch rate (rad/sec)' ]
csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
csv_writer.writeheader()

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

video_writer = skvideo.io.FFmpegWriter(output_video, inputdict=inputdict, outputdict=sane)

#frame = cv2.imread("/home/curt/Downloads/Frame1.png")
#frame_scale = cv2.resize(frame, (0,0), fx=scale, fy=scale,
#                         interpolation=cv2.INTER_AREA)
#cv2.imshow('scaled orig', frame_scale)
#frame_undist = cv2.undistort(frame_scale, K, np.array(dist))
#cv2.imshow("undist", frame_undist)
#cv2.waitKey()

counter = -1
last_roll = None
last_pitch = None
for frame in reader.nextFrame():
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    counter += 1
    
    if counter < skip_frames:
        if counter % 100 == 0:
            print("Skipping %d frames..." % counter)
        else:
            continue

    print("Frame %d" % counter)

    method = cv2.INTER_AREA
    frame_scale = cv2.resize(frame, (0,0), fx=scale, fy=scale,
                             interpolation=method)
    cv2.imshow('scaled orig', frame_scale)
    frame_undist = cv2.undistort(frame_scale, K, np.array(dist))

    # test horizon detection
    lines = horizon.horizon(frame_undist)
    if not lines is None:
        #best_line = horizon.track_best(lines)
        best_line = lines[0]
        roll, pitch = horizon.get_camera_attitude(best_line, IK, cu, cv)
        if last_roll is None or last_pitch is None:
            roll_rate = 0
            pitch_rate = 0
        else:
            roll_rate = (roll - last_roll) * fps * d2r
            pitch_rate = (pitch - last_pitch) * fps * d2r
        last_roll = roll
        last_pitch = pitch
        horizon.draw(frame_undist, best_line, IK, cu, cv)
        row = {'frame': counter,
               'video time': "%.4f" % (counter / fps),
               'camera roll (deg)': "%.2f" % roll,
               'camera pitch (deg)': "%.2f" % pitch,
               'roll rate (rad/sec)': "%.3f" % roll_rate,
               'pitch rate (rad/sec)': "%.3f" % pitch_rate }
        csv_writer.writerow(row)
    else:
        roll = None
        pitch = None
        roll_rate = 0.0
        pitch_rate = 0.0
        last_roll = None
        last_pitch = None

    cv2.imshow("horizon", frame_undist)
    if args.write:
        #write the frame as RGB not BGR
        video_writer.writeFrame(frame_undist[:,:,::-1])
    if 0xFF & cv2.waitKey(5) == 27:
        break

csvfile.close()
cv2.destroyAllWindows()

