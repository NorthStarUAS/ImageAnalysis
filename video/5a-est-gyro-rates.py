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

from props import PropertyNode
import props_json

import sys
sys.path.append('../scripts')
from lib import transformations

import camera

# constants
d2r = math.pi / 180.0
r2d = 180.0 / math.pi

match_ratio = 0.75
max_features = 500
catchup = 0.02
affine_minpts = 7
tol = 2.0

parser = argparse.ArgumentParser(description='Estimate gyro biases from movie.')
parser.add_argument('video', help='video file')
parser.add_argument('--camera', help='select camera calibration file')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--skip-frames', type=int, default=0, help='skip n initial frames')
parser.add_argument('--equalize', action='store_true', help='disable image equalization')
parser.add_argument('--draw-keypoints', action='store_true', help='draw keypoints on output')
parser.add_argument('--draw-masks', action='store_true', help='draw stabilization masks')
parser.add_argument('--write', action='store_true', help='write out video with keypoints shown')
args = parser.parse_args()

#file = args.video
scale = args.scale
skip_frames = args.skip_frames

# pathname work
abspath = os.path.abspath(args.video)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.video)
output_csv = filename + "_rates.csv"
output_video = filename + "_keypts" + ext
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


# find affine transform between matching keypoints in pixel
# coordinate space.  fullAffine=True means unconstrained to
# include best warp/shear.  fullAffine=False means limit the
# matrix to only best rotation, translation, and scale.
def findAffine(src, dst, fullAffine=False):
    #print("src:", src)
    #print("dst:", dst)
    if len(src) >= affine_minpts:
        # affine = cv2.estimateRigidTransform(np.array([src]), np.array([dst]), fullAffine)
        affine, status = \
            cv2.estimateAffinePartial2D(np.array([src]).astype(np.float32),
                                        np.array([dst]).astype(np.float32))
    else:
        affine = None
    #print str(affine)
    return affine

def decomposeAffine(affine):
    if affine is None:
        print("HEY: we should see affine=None here!")
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
    return p1, p2, kp_pairs, idx_pairs, mkp1

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
        M, status = cv2.findHomography(p1, p2, cv2.LMEDS, tol)
    elif method == 'fundamental':
        M, status = cv2.findFundamentalMat(p1, p2, cv2.LMEDS, tol)
    elif method == 'essential':
        M, status = cv2.findEssentialMat(p1, p2, K, cv2.LMEDS, threshold=tol)
    elif method == 'none':
        M = None
        status = np.ones(total)
    newp1 = []
    newp2 = []
    for i, flag in enumerate(status):
        if flag:
            newp1.append(p1[i])
            newp2.append(p2[i])
    p1 = np.float32(newp1)
    p2 = np.float32(newp2)
    inliers = np.sum(status)
    total = len(status)
    #print '%s%d / %d  inliers/matched' % (space, np.sum(status), len(status))
    return M, status, np.float32(newp1), np.float32(newp2)

# track persistant edges and create a mask from them (useful when
# portions of our own airframe are visible in the video)
edges_accum = None
edge_filt_time = 30             # sec
edge_filt_frames = edge_filt_time * fps
weight_a = (edge_filt_frames - 1) / edge_filt_frames
weight_b = 1 / edge_filt_frames
def make_edge_mask(gray):
    global edges_accum
    global edges_counter
    edges = cv2.Canny(gray, 50, 150)
    cv2.imshow("edges", edges)
    if edges_accum is None:
        edges_accum = edges.astype(np.float32) * 0
    else:
        edges_accum = weight_a * edges_accum + weight_b * edges.astype(np.float32)
    cv2.imshow("edges filter", edges_accum.astype('uint8'))
    max = np.max(edges_accum)
    thresh = int(round(max * 0.35))
    ret2, thresh1 = cv2.threshold(edges_accum.astype('uint8'), thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    thresh1 = cv2.dilate(thresh1, kernel, iterations=2)
    cv2.imshow('edge thresh1', thresh1)
    return thresh1

# only pass through keypoints if they aren't masked by the mask image
def apply_edge_mask(mask, kp_list):
    new_list = []
    for kp in kp_list:
        if mask[int(round(kp.pt[1])), int(round(kp.pt[0]))] == 0:
            new_list.append(kp)
    return new_list

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

accum = None
kp_list_last = []
des_list_last = []
p1 = []
p2 = []
mkp1 = []
counter = 0

rot_last = 0
tx_last = 0
ty_last = 0

if args.equalize:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

csvfile = open(output_csv, 'w')
fieldnames=['frame', 'video time', 'p (rad/sec)', 'q (rad/sec)', 'r (rad/sec)']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

pbar = tqdm(total=int(total_frames), smoothing=0.05)
for frame in reader.nextFrame():
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)

    filtered = []

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

    gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        
    if args.equalize:
        gray = clahe.apply(gray)
        cv2.imshow("gray equalized", gray)

    edge_mask = make_edge_mask(gray)
    
    kp_list = detector.detect(gray)
    kp_list = apply_edge_mask(edge_mask, kp_list)
    kp_list, des_list = extractor.compute(gray, kp_list)

    # Fixme: make a command line option
    # possible values are "homography", "fundamental", "essential", "none"
    filter_method = "homography"

    if not (des_list_last is None) and not (des_list is None) and len(des_list_last) and len(des_list):
        #print(len(des_list_last), len(des_list))
        matches = matcher.knnMatch(des_list, trainDescriptors=des_list_last, k=2)
        p1, p2, kp_pairs, idx_pairs, mkp1 = filterMatches(kp_list, kp_list_last, matches)

        M, status, newp1, newp2 = filterFeatures(p1, p2, K, filter_method)
        filtered = []
        for i, flag in enumerate(status):
            if flag:
                filtered.append(mkp1[i])
        if filter_method == "essential" and not M is None:
            print("M:", M)
            # we can attempt to extract frame rotation from the
            # essential matrix (but this seems currently very noisy or
            # something is wrong in my assumptions or usage.)
            (n, R, tvec, mask) = cv2.recoverPose(E=M,
                                                 points1=p1, points2=p2,
                                                 cameraMatrix=K)
            print("R:", R)
            (yaw, pitch, roll) = transformations.euler_from_matrix(R, 'rzyx')
            print("ypr: %.2f %.2f %.2f" % (yaw*r2d, pitch*r2d, roll*r2d))
    kp_list_last = kp_list
    des_list_last = des_list

    affine = findAffine(p2, p1, fullAffine=False)
    if affine is None:
        continue
    (rot, tx, ty, sx, sy) = decomposeAffine(affine)
    if abs(rot) > 6 or math.sqrt(tx*tx+ty*ty) > 10:
        (rot, tx, ty, sx, sy) = (0.0, 0.0, 0.0, 1.0, 1.0)
    #print affine
    #print (rot, tx, ty, sx, sy)

    translate_only = False
    rotate_translate_only = True
    if translate_only:
        rot = 0.0
        sx = 1.0
        sy = 1.0
    elif rotate_translate_only:
        sx = 1.0
        sy = 1.0

    # roll rate from affine rotation
    p = -rot * fps * d2r

    # pitch and yaw rates from affine translation projected through
    # camera calibration
    #print(cu, cv)
    #print("IK:", IK)
    #uv0 = np.array([cu,    cv,    1.0])
    uv1 = np.array([cu-tx, cv,    1.0])
    uv2 = np.array([cu,    cv+ty, 1.0])
    #proj0 = IK.dot(uv0)
    proj1 = IK.dot(uv1)
    proj2 = IK.dot(uv2)
    #print(proj1, proj2)
    dp1 = np.dot([0, 0, 1], proj1/np.linalg.norm(proj1))
    dp2 = np.dot([0, 0, 1], proj2/np.linalg.norm(proj2))
    #print("dp:", dp1, dp2)
    if uv1[1] > cu:
        r = -np.arccos(dp1) * fps
    else:
        r = np.arccos(dp1) * fps
    if uv2[1] < cv:
        q = -np.arccos(dp2) * fps
    else:
        q = np.arccos(dp2) * fps
    
    # divide tx, ty by args.scale to get a translation value
    # relative to the original movie size.
    row = { 'frame': counter,
            'video time': "%.4f" % (counter / fps),
            'p (rad/sec)': "%.4f" % p,
            'q (rad/sec)': "%.4f" % q,
            'r (rad/sec)': "%.4f" % r }
    writer.writerow(row)

    #print("affine motion: %d %.2f %.1f %.1f" % (counter, rot, tx, ty))
    #print("est gyro: %d %.3f %.3f %.3f" % (counter, p, q, r))

    if True:
        for kp in filtered:
            cv2.circle(frame_undist, (int(kp.pt[0]), int(kp.pt[1])), 3, (0,255,0), 1, cv2.LINE_AA)
        res1 = frame_undist

    else:
        res1 = frame_undist

    final = accum
    if args.draw_keypoints:
        new_filtered = []
        if affine_new != None:
            affine_T = affine_new.T
            for kp in filtered:
                a = np.array( [kp.pt[0], kp.pt[1], 1.0] )
                pt = np.dot(a, affine_T)
                new_kp = cv2.KeyPoint(pt[0], pt[1], kp.size)
                new_filtered.append(new_kp)
        final = cv2.drawKeypoints(accum, new_filtered, color=(0,255,0), flags=0)
   
    counter += 1
    
    cv2.imshow('bgr', res1)
    if args.write:
        video_writer.writeFrame(final[:,:,::-1])
    if 0xFF & cv2.waitKey(5) == 27:
        break
    pbar.update(1)
pbar.close()

cv2.destroyAllWindows()

