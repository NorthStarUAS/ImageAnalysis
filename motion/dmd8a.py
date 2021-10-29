#!/usr/bin/env python3

# dmd8a.py: copy of dmd8.py but use Maziar's streaming dmd algorith
# instead of pydmd

# dmd8.py: copy of dmd6.py (best background subtraction approach so far) but
# apply dmd (pydmd w/ sliding window) to the motion diff.

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

from streaming_dmd import StreamingDMD
#from streaming_dmd1 import StreamingDMD

import camera
from motion import myOpticalFlow
#from motion import myFarnebackFlow

parser = argparse.ArgumentParser(description='Track motion with homography transformation.')
parser.add_argument('video', help='video file')
parser.add_argument('--camera', help='select camera calibration file')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--fg-alpha', type=float, default=0.5, help='forground filter factor')
parser.add_argument('--bg-alpha', type=float, default=0.05, help='background filter factor')
parser.add_argument('--skip-frames', type=int, default=0, help='skip n initial frames')
parser.add_argument('--write', action='store_true', help='write out video with keypoints shown')
parser.add_argument('--write-quad', action='store_true', help='write out quad video')
parser.add_argument('--write-dmd', action='store_true', help='write out dmd video')
args = parser.parse_args()

scale = args.scale

# pathname work
abspath = os.path.abspath(args.video)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.video)
bg_video = filename + "_bg.mp4"
motion_video = filename + "_motion.mp4"
feat_video = filename + "_feat.mp4"
quad_video = filename + "_quad.mp4"
mode_video = filename + "_modes.mp4"
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
if args.write:
    motion_writer = skvideo.io.FFmpegWriter(motion_video, inputdict=inputdict,
                                            outputdict=sane)
    bg_writer = skvideo.io.FFmpegWriter(bg_video, inputdict=inputdict,
                                        outputdict=sane)
    feat_writer = skvideo.io.FFmpegWriter(feat_video, inputdict=inputdict,
                                          outputdict=sane)

if args.write_quad:
    quad_writer = skvideo.io.FFmpegWriter(quad_video, inputdict=inputdict,
                                          outputdict=sane)

if args.write_dmd:
    mode_writer = skvideo.io.FFmpegWriter(mode_video, inputdict=inputdict,
                                          outputdict=sane)

flow = myOpticalFlow()
#farneback = myFarnebackFlow()

bg_alpha = args.bg_alpha
if bg_alpha < 0.0: bg_alpha = 0.0
if bg_alpha > 1.0: bg_alpha = 1.0
fg_alpha = args.fg_alpha
if fg_alpha < 0.0: fg_alpha = 0.0
if fg_alpha > 1.0: fg_alpha = 1.0

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

# dmd options and structures
rows = 3
cols = 3
max_rank = ((rows*cols) * 2) - 2
print("max rank:", max_rank)
dmd_size = 200
window_size = 32
sdmd = StreamingDMD(max_rank=max_rank)
x = np.array( [] )
y = np.array( [] )

def draw_text(img, label, x, y, subscale=1.0, just="center"):
    font_scale = subscale * h / 700
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale, 1)
    if just == "center":
        locx = int(x - size[0][0]*0.5)
        locy = int(y + size[0][1]*1.5)
    elif just == "lower-right":
        locx = int(x - size[0][0])
        locy = int(y - size[0][1]*0.5)
    elif just == "upper-right":
        locx = int(x - size[0][0])
        locy = int(y + size[0][1])
    elif just == "right":
        locx = int(x - size[0][0])
        locy = int(y + size[0][1]*0.5)

    cv2.putText(img, label, (locx, locy),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                1, cv2.LINE_AA)

def draw_bar(img, min, max):
    (h, w) = img.shape[:2]
    min_str = "%.2f" % min
    max_str = "%.2f" % max
    font_scale = 0.8 * h / 700
    min_size = cv2.getTextSize(min_str, cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale, 1)
    max_size = cv2.getTextSize(max_str, cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale, 1)
    th = min_size[0][1]
    span = h - 2*th
    for i in range(span):
        img[i+th,(w-8):(w-2)] = 255 - int(i * 255 / span)
    draw_text(img, min_str, w-12, h-th, subscale=0.8, just="lower-right")
    draw_text(img, max_str, w-12, th, subscale=0.8, just="upper-right")
        

def draw_mode(label, mode, shape, factor=2):
    real = factor * np.abs(mode.real)
    min = np.min(real)
    max = np.max(real)
    range = max - min
    equalized = (real - min) * (255 / range)
    (h, w) = shape[:2]
    big = cv2.resize(equalized.reshape((dmd_size,dmd_size)).astype('uint8'), (w, h), interpolation=cv2.INTER_AREA)
    draw_bar(big, min, max)
    cv2.imshow(label, big)
    return big

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
        curr_filt = curr_proj.astype('float32') * (1 - fg_alpha) \
            + frame_undist.astype('float32') * fg_alpha
        cv2.imshow("prev_filt", prev_filt.astype('uint8'))
        cv2.imshow("curr_filt", curr_filt.astype('uint8'))
        diff = cv2.absdiff(prev_proj.astype('uint8'), curr_filt.astype('uint8'))
        bg_filt = bg_proj.astype('float32') * (1 - bg_alpha) \
            + frame_undist.astype('float32') * bg_alpha
        prev_filt = curr_proj.astype('float32') * (1 - fg_alpha) \
            + frame_undist.astype('float32') * fg_alpha
    diff_max = np.max(diff)
    diff_factor = 0.95*diff_factor + 0.05*diff_max
    if diff_factor < diff_max:
        diff_factor = diff_max
    print("diff_factor:", diff_factor)
    diff_img = (255*diff.astype('float32')/diff_factor).astype('uint8')
    cv2.imshow("diff", diff_img.astype('uint8'))
    cv2.imshow("background", bg_filt.astype('uint8'))

    # now run dmd on the diff image (already compensated for camera
    # motion)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    small = cv2.resize(gray, (dmd_size,dmd_size), interpolation=cv2.INTER_AREA).flatten()
    if not small.any():
        continue
    if not x.shape[0] or not y.shape[0]:
        x = small.copy().flatten()
    y = small.copy().flatten()
    sdmd.update(x.astype('float32'), y.astype('float32'))
    x = y.copy()
    sdmd_modes, sdmd_evals = sdmd.compute_modes()
    if len(sdmd_evals):
        #print(dmd.eigs)
        idx = np.argsort(np.abs(sdmd_evals-1))
        #idx = np.argsort(np.abs(dmd.eigs.imag))
        print(idx)
        print(sdmd_evals)

        (h, w) = frame_undist.shape[:2]
        grid = np.zeros( (h*rows, w*cols) ).astype('uint8')
        orig_gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        grid[0:h,0:w] = orig_gray
        draw_text(grid, "Original", w*0.5, 0)
        r = 0
        c = 1
        for i in range(0, max_rank, 2):
            if i >= len(idx):
                break
            #print(i)
            if c >= cols:
                r += 1
                c = 0
            #print("grid:", r, c, "i:", i)
            img = sdmd_modes[:,idx[i]].reshape(small.shape[:2])
            if i == 0:
                factor = 1
            else:
                factor = 2
            grid[r*h:(r+1)*h,c*w:(c+1)*w] = draw_mode("a", img, gray.shape, factor)
            #grid[r*h:(r+1)*h,c*w:(c+1)*w] = scaled
            eig = sdmd_evals[idx[i]]
            label = "Mode: %d (%.4f + %.4fj)" % (i, eig.real, eig.imag)
            draw_text(grid, label, (c+0.5)*w, r*h)
            c += 1
        draw_text(grid, "www.uav.aem.umn.edu", w*(rows-0.03), h*(cols-0.03), just="lower-right")
        cv2.imshow("grid", grid)
        if args.write_dmd:
            print("grid:", grid.shape)
            mode_writer.writeFrame(grid)

    if True:
        frame_feat = frame_undist.copy()
        for pt in curr_pts:
            cv2.circle(frame_feat, (int(pt[0][0]), int(pt[0][1])), 3, (0,255,0), 1, cv2.LINE_AA)
        for pt in prev_pts:
            cv2.circle(frame_feat, (int(pt[0][0]), int(pt[0][1])), 2, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow('features', frame_feat)

    if args.write:
        # if rgb
        motion_writer.writeFrame(diff_img[:,:,::-1])
        bg_writer.writeFrame(bg_filt[:,:,::-1])
        feat_writer.writeFrame(frame_feat[:,:,::-1])
        # if gray
        #motion_writer.writeFrame(diff_img)
        #bg_writer.writeFrame(prev_filt)

    if args.write_quad:
        def draw_text(img, label, x, y, subscale=1.0, just="center"):
            font_scale = subscale * h / 350
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, 1)
            if just == "center":
                locx = int(x - size[0][0]*0.5)
                locy = int(y + size[0][1]*1.5)
            elif just == "lower-right":
                locx = int(x - size[0][0])
                locy = int(y - size[0][1])
                
            cv2.putText(img, label, (locx, locy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        1, cv2.LINE_AA)
            
        quad = np.zeros( (h*2, w*2, 3) ).astype('uint8')
        quad[0:h,0:w,:] = frame_undist
        quad[h:,0:w,:] = frame_feat
        quad[0:h,w:,:] = diff_img
        quad[h:,w:,:] = bg_filt
        draw_text(quad, "Original", w*0.5, 0)
        draw_text(quad, "Feature Flow", w*0.5, h)
        draw_text(quad, "Motion Layer", w*1.5, 0)
        draw_text(quad, "Background Layer", w*1.5, h)
        draw_text(quad, "www.uav.aem.umn.edu", 1.97*w, 1.97*h, subscale=0.5, just="lower-right")

        cv2.imshow("quad", quad)
        quad_writer.writeFrame(quad[:,:,::-1])

    if 0xFF & cv2.waitKey(1) == 27:
        break

    pbar.update(1)
pbar.close()

cv2.destroyAllWindows()

