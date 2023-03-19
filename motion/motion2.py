#!/usr/bin/env python3

import argparse
import cv2
import skvideo.io               # pip3 install sk-video
import json
import numpy as np
import os
from tqdm import tqdm

import sys
sys.path.append('../scripts')

import camera

# constants
match_ratio = 0.75
max_features = 500
catchup = 0.02
affine_minpts = 7
tol = 1.0

parser = argparse.ArgumentParser(description='Track motion with homography transformation.')
parser.add_argument('video', help='video file')
parser.add_argument('--camera', help='select camera calibration file')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--skip-frames', type=int, default=0, help='skip n initial frames')
parser.add_argument('--equalize', action='store_true', help='disable image equalization')
parser.add_argument('--write', action='store_true', help='write out video with keypoints shown')
args = parser.parse_args()

scale = args.scale
skip_frames = args.skip_frames

# pathname work
abspath = os.path.abspath(args.video)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.video)
output_video = filename + "_keypts.mp4"
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
    video_writer = skvideo.io.FFmpegWriter(output_video, inputdict=inputdict, outputdict=sane)

def filterMatches(kp1, kp2, matches):
    mkp1, mkp2 = [], []
    idx_pairs = []
    used = np.zeros(len(kp2), np.bool_)
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * match_ratio:
            #print " dist[0] = %d  dist[1] = %d" % (m[0].distance, m[1].distance)
            m = m[0]
            # FIXME: ignore the bottom section of movie for feature detection
            #if kp1[m.queryIdx].pt[1] > h*0.75:
            #    continue
            if not used[m.trainIdx]:
                used[m.trainIdx] = True
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
                idx_pairs.append( (m.queryIdx, m.trainIdx) )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs, idx_pairs

def filterFeatures(p1, p2, K, method):
    inliers = 0
    total = len(p1)
    space = ""
    status = []
    M = None
    if len(p1) < 7:
        # not enough points
        return None, np.zeros(total), [], []
    if method == 'homography':
        M, status = cv2.findHomography(p1, p2, cv2.RANSAC, tol)
    elif method == 'fundamental':
        M, status = cv2.findFundamentalMat(p1, p2, cv2.RANSAC, tol)
    elif method == 'essential':
        M, status = cv2.findEssentialMat(p1, p2, K, cv2.LMEDS, prob=0.99999, threshold=tol)
    elif method == 'none':
        M = None
        status = np.ones(total)
    newp1 = []
    newp2 = []
    for i, flag in enumerate(status):
        if flag:
            newp1.append(p1[i])
            newp2.append(p2[i])
    inliers = np.sum(status)
    total = len(status)
    #print('%s%d / %d  inliers/matched' % (space, np.sum(status), len(status)))
    return M, status, np.float32(newp1), np.float32(newp2)

if True:
    # for ORB
    detector = cv2.ORB_create(max_features)
    extractor = detector
    norm = cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm)
else:
    # for SIFT
    max_features = 200
    detector = cv2.SIFT_create(nfeatures=max_features, nOctaveLayers=5)
    extractor = detector
    norm = cv2.NORM_L2
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    FLANN_INDEX_LSH    = 6
    flann_params = { 'algorithm': FLANN_INDEX_KDTREE,
                     'trees': 5 }
    matcher = cv2.FlannBasedMatcher(flann_params, {}) # bug : need to pass empty dict (#1329)

slow = np.array( [] )
kp_list_last = []
des_list_last = []
p1 = []
p2 = []
counter = -1
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

if True or args.equalize:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

pbar = tqdm(total=int(total_frames), smoothing=0.05)
for frame in reader.nextFrame():
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    counter += 1
    #if counter % 8 != 0:
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
    shape = frame_scale.shape
    tol = shape[1] / 200.0
    if tol < 1.0: tol = 1.0

    frame_undist = cv2.undistort(frame_scale, K, np.array(dist))
    cv2.imshow("frame undist", frame_undist)

    gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

    if True or args.equalize:
        gray = clahe.apply(gray)
        cv2.imshow("gray equalized", gray)

    kp_list = detector.detect(gray)
    kp_list, des_list = extractor.compute(gray, kp_list)

    # Fixme: make a command line option
    # possible values are "homography", "fundamental", "essential", "none"
    filter_method = "homography"

    if des_list_last is None or des_list is None or len(des_list_last) == 0 or len(des_list) == 0:
        kp_list_last = kp_list
        des_list_last = des_list
        continue

    #print(len(des_list_last), len(des_list))
    matches = matcher.knnMatch(des_list, trainDescriptors=des_list_last, k=2)
    p1, p2, kp_pairs, idx_pairs = filterMatches(kp_list, kp_list_last, matches)
    kp_list_last = kp_list
    des_list_last = des_list

    M, status, newp1, newp2 = filterFeatures(p1, p2, K, filter_method)
    if len(newp1) < 1:
        continue

    print("M:\n", M)

    if slow.shape[0] == 0:
        slow = frame_undist.copy()
    else:
        mask = np.ones( frame_undist.shape[:2] ).astype('uint8')*255
        mask = cv2.warpPerspective(mask, np.linalg.inv(M), (frame_undist.shape[1], frame_undist.shape[0]))
        mask = cv2.erode(mask, kernel3, iterations=3)
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
        slow_proj = cv2.warpPerspective(slow, np.linalg.inv(M), (frame_undist.shape[1], frame_undist.shape[0]))
        a2 = cv2.bitwise_and(slow_proj, slow_proj, mask=mask)
        #cv2.imshow("a2", a2)
        slow_comp = cv2.add(a1, a2)
        slow = cv2.addWeighted(slow_comp, 0.95, frame_undist, 0.05, 0)
        #blend = cv2.resize(blend, (int(w*args.scale), int(h*args.scale)))
    cv2.imshow("zero frequency background", slow)

    if True:
        for pt in newp1:
            cv2.circle(frame_undist, (int(pt[0]), int(pt[1])), 3, (0,255,0), 1, cv2.LINE_AA)
        for pt in newp2:
            cv2.circle(frame_undist, (int(pt[0]), int(pt[1])), 2, (0,0,255), 1, cv2.LINE_AA)

    cv2.imshow('bgr', frame_undist)

    if args.write:
        video_writer.writeFrame(frame_undist[:,:,::-1])
    if 0xFF & cv2.waitKey(1) == 27:
        break
    pbar.update(1)
pbar.close()

cv2.destroyAllWindows()

