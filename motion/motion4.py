#!/usr/bin/env python3

import argparse
import cv2
import skvideo.io               # pip3 install sk-video
import json
import numpy as np
import os
from tqdm import tqdm

from props import PropertyNode
import props_json

import camera
from motion import myFeatureFlow

parser = argparse.ArgumentParser(description='Track motion with homography transformation.')
parser.add_argument('video', help='video file')
parser.add_argument('--camera', help='select camera calibration file')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--skip-frames', type=int, default=0, help='skip n initial frames')
parser.add_argument('--write', action='store_true', help='write out video with keypoints shown')
args = parser.parse_args()

scale = args.scale
skip_frames = args.skip_frames

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

flow = myFeatureFlow(K)

slow = np.array( [] )
fast = np.array( [] )
kp_list_last = []
des_list_last = []
p1 = []
p2 = []
counter = -1
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
warp_flags = cv2.INTER_LANCZOS4
diff_factor = 255

pbar = tqdm(total=int(total_frames), smoothing=0.05)
for frame in reader.nextFrame():
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    counter += 1
    #if counter % 2 != 0:
    #    continue

    if counter < skip_frames:
        if counter % 100 == 0:
            print("Skipping %d frames..." % counter)
        else:
            continue

    # print "Frame %d" % counter

    method = cv2.INTER_AREA
    #method = cv2.INTER_LANCZOS4
    frame_scale = cv2.resize(frame, (0,0), fx=scale, fy=scale,
                             interpolation=method)
    cv2.imshow('scaled orig', frame_scale)

    frame_undist = cv2.undistort(frame_scale, K, np.array(dist))
    cv2.imshow("frame undist", frame_undist)

    M, newp1, newp2 = flow.update(frame_undist)

    if slow.shape[0] == 0 or fast.shape[0] == 0:
        slow = frame_undist.copy()
        fast = frame_undist.copy()
    else:
        mask = np.ones( frame_undist.shape[:2] ).astype('uint8')*255
        mask = cv2.warpPerspective(mask, np.linalg.inv(M), (frame_undist.shape[1], frame_undist.shape[0]), flags=warp_flags)
        mask = cv2.erode(mask, kernel5, iterations=3)
        print(np.count_nonzero(mask==255) + np.count_nonzero(mask==0))
        ret, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        print(np.count_nonzero(mask==255) + np.count_nonzero(mask==0))
        mask_inv = cv2.bitwise_not(mask)
        print(np.count_nonzero(mask_inv==255) + np.count_nonzero(mask_inv==0))
        #cv2.imshow("mask", mask)
        #cv2.imshow("mask_inv", mask_inv)
        print(frame_undist.shape, mask_inv.shape)
        a1 = cv2.bitwise_and(frame_undist, frame_undist, mask=mask_inv)
        #cv2.imshow("a1", a1)
        slow_proj = cv2.warpPerspective(slow, np.linalg.inv(M), (frame_undist.shape[1], frame_undist.shape[0]), flags=warp_flags)
        fast_proj = cv2.warpPerspective(fast, np.linalg.inv(M), (frame_undist.shape[1], frame_undist.shape[0]), flags=warp_flags)
        slow_a2 = cv2.bitwise_and(slow_proj, slow_proj, mask=mask)
        fast_a2 = cv2.bitwise_and(fast_proj, fast_proj, mask=mask)
        #cv2.imshow("a2", a2)
        slow_comp = cv2.add(a1, slow_a2)
        fast_comp = cv2.add(a1, fast_a2)
        slow = cv2.addWeighted(slow_comp, 0.95, frame_undist, 0.05, 0)
        fast = cv2.addWeighted(fast_comp, 0.4, frame_undist, 0.6, 0)
        #blend = cv2.resize(blend, (int(w*args.scale), int(h*args.scale)))
    cv2.imshow("zero frequency background", slow)
    cv2.imshow("fast average", fast)
    diff = cv2.absdiff(slow, fast)
    diff_max = np.max(diff)
    diff_factor = 0.95*diff_factor + 0.05*diff_max
    if diff_factor < diff_max:
        diff_factor = diff_max
    print("diff_factor:", diff_factor)
    diff_img = (255*diff.astype('float32')/diff_factor).astype('uint8')
    cv2.imshow("diff", diff_img)

    if True:
        for pt in newp1:
            cv2.circle(frame_scale, (int(pt[0]), int(pt[1])), 3, (0,255,0), 1, cv2.LINE_AA)
        for pt in newp2:
            cv2.circle(frame_scale, (int(pt[0]), int(pt[1])), 2, (0,0,255), 1, cv2.LINE_AA)

    cv2.imshow('bgr', frame_scale)

    # highlight = frame_scale.astype('float32') + 2*cv2.merge((diff, diff, diff))
    # cv2.imshow("highlight", (255*highlight.astype('float32')/np.max(highlight)).astype('uint8'))

    if args.write:
        # if rgb
        motion_writer.writeFrame(diff_img[:,:,::-1])
        bg_writer.writeFrame(slow[:,:,::-1])
        # if gray
        #motion_writer.writeFrame(diff_img)
        #bg_writer.writeFrame(slow)

    if 0xFF & cv2.waitKey(1) == 27:
        break
    pbar.update(1)
pbar.close()

cv2.destroyAllWindows()

