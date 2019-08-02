#!/usr/bin/python3

import argparse
import csv
import cv2
import skvideo.io
import json
import math
import numpy as np
import os
import sys

from props import PropertyNode
import props_json

sys.path.append('../scripts')
from lib import Render
from lib import transformations
r = Render.Render()

d2r = math.pi / 180.0
r2d = 180.0 / math.pi

match_ratio = 0.75
max_features = 500
smooth = 0.005
catchup = 0.02
affine_minpts = 7
tol = 2.0

parser = argparse.ArgumentParser(description='Estimate gyro biases from movie.')
parser.add_argument('--movie', required=True, help='movie file')
parser.add_argument('--camera', help='select camera calibration file')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--skip-frames', type=int, default=0, help='skip n initial frames')
parser.add_argument('--no-equalize', action='store_true', help='disable image equalization')
parser.add_argument('--draw-keypoints', action='store_true', help='draw keypoints on output')
parser.add_argument('--draw-masks', action='store_true', help='draw stabilization masks')
parser.add_argument('--write-smooth', action='store_true', help='write out the smoothed video')
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
output_avi = filename + "_smooth.avi"
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

metadata = skvideo.io.ffprobe(args.movie)
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

print("Opening ", args.movie)
reader = skvideo.io.FFmpegReader(args.movie, inputdict={}, outputdict={})

if args.write_smooth:
    #outfourcc = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
    outfourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #outfourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')
    #outfourcc = cv2.cv.CV_FOURCC('X', '2', '6', '4')
    #outfourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(output_avi, outfourcc, fps, (w, h))

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

def do_motion(src, dst):
    pass

def overlay(new_frame, base, motion_mask=None):
    newtmp = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    ret, newmask = cv2.threshold(newtmp, 0, 255, cv2.THRESH_BINARY_INV)

    blendsize = (3,3)
    kernel = np.ones(blendsize,'uint8')
    mask_dilate = cv2.dilate(newmask, kernel)
    #cv2.imshow('mask_dilate', mask_dilate)
    ret, mask_final = cv2.threshold(mask_dilate, 254, 255, cv2.THRESH_BINARY)
    if args.draw_masks:
        cv2.imshow('mask_final', mask_final)

    mask_inv = 255 - mask_final
    if motion_mask != None:
        motion_inv = 255 - motion_mask
        mask_inv = cv2.bitwise_and(mask_inv, motion_mask)
        cv2.imshow('mask_inv1', mask_inv)
        mask_final = cv2.bitwise_or(mask_final, motion_inv)
        cv2.imshow('mask_final1', mask_final)
    if args.draw_masks:
        cv2.imshow('mask_inv', mask_inv)

    mask_final_norm = mask_final / 255.0
    mask_inv_norm = mask_inv / 255.0

    base[:,:,0] = base[:,:,0] * mask_final_norm
    base[:,:,1] = base[:,:,1] * mask_final_norm
    base[:,:,2] = base[:,:,2] * mask_final_norm
    #cv2.imshow('base', base)

    new_frame[:,:,0] = new_frame[:,:,0] * mask_inv_norm
    new_frame[:,:,1] = new_frame[:,:,1] * mask_inv_norm
    new_frame[:,:,2] = new_frame[:,:,2] * mask_inv_norm

    accum = cv2.add(base, new_frame)
    #cv2.imshow('accum', accum)
    return accum

def motion1(new_frame, base):
    motion = cv2.absdiff(base, new_frame)
    gray = cv2.cvtColor(motion, cv2.COLOR_BGR2GRAY)
    cv2.imshow('motion', gray)
    ret, motion_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)

    blendsize = (3,3)
    kernel = np.ones(blendsize,'uint8')
    motion_mask = cv2.erode(motion_mask, kernel)

    # lots
    motion_mask /= 1.1429
    motion_mask += 16

    # medium
    #motion_mask /= 1.333
    #motion_mask += 32

    # minimal
    #motion_mask /= 2
    #motion_mask += 64

    cv2.imshow('motion1', motion_mask)
    return motion_mask

def motion2(new_frame, base):
    motion = cv2.absdiff(base, new_frame)
    gray = cv2.cvtColor(motion, cv2.COLOR_BGR2GRAY)
    cv2.imshow('motion', gray)
    motion_mask = 255 - gray
    motion_mask /= 2
    motion_mask += 2

    cv2.imshow('motion2', motion_mask)
    return motion_mask

last_frame = None
static_mask = None
def motion3(frame, counter):
    global last_frame
    global static_mask
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if last_frame is None:
        pass
    else:
        diff = cv2.absdiff(gray, last_frame)
        cv2.imshow('motion3', diff)
        if static_mask is None:
            static_mask = np.float32(diff)
        else:
            if counter > 1000:
                c = float(1000)
            else:
                c = float(counter)
            f = float(c - 1) / c
            static_mask = f*static_mask + (1.0 - f)*np.float32(diff)
        mask_uint8 = np.uint8(static_mask)
        cv2.imshow('mask3', mask_uint8)
        ret, newmask = cv2.threshold(mask_uint8, 2, 255, cv2.THRESH_BINARY)
        cv2.imshow('newmask', newmask)
    last_frame = gray

# average of frames (the stationary stuff should be the sharpest)
sum_frame = None
sum_counter = 0
def motion4(frame, counter):
    global sum_frame
    global sum_counter

    if sum_frame is None:
        sum_frame = np.float32(frame)
    else:
        sum_frame += np.float32(frame)
    sum_counter += 1

    avg_frame = sum_frame / float(sum_counter)
    cv2.imshow('motion4', np.uint8(avg_frame))

# low pass filter of frames
filt_frame = None
def motion5(frame, counter):
    global filt_frame

    if filt_frame is None:
        filt_frame = np.float32(frame)

    factor = 0.96
    filt_frame = factor * filt_frame + (1 - factor) * np.float32(frame)
    cv2.imshow('motion5', np.uint8(filt_frame))
    
# low pass filter of frames
filt1_frame = None
filt2_frame = None
filt_mask = None
def motion6(frame, counter):
    global filt1_frame
    global filt2_frame
    global filt_mask
    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame
    if filt1_frame is None: filt1_frame = np.float32(gray)
    if filt2_frame is None: filt2_frame = np.float32(gray)

    factor1 = 0.9
    factor2 = 0.995
    factor3 = 0.995
    
    filt1_frame = factor1 * filt1_frame + (1 - factor1) * np.float32(gray)
    filt2_frame = factor2 * filt2_frame + (1 - factor2) * np.float32(gray)
    mask = np.absolute(filt1_frame - filt2_frame)
    if filt_mask is None: filt_mask = mask
    filt_mask = factor3 * filt_mask + (1 - factor3) * mask
    
    cv2.imshow('motion6-1', np.uint8(filt1_frame))
    cv2.imshow('motion6-2', np.uint8(filt2_frame))
    cv2.imshow('motion6', np.uint8(filt_mask))

# this one doesn't work very well on my movies
fgbg = None
def motion7(frame, counter):
    global fgbg

    if fgbg is None:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    fgmask = fgbg.apply(frame)
    cv2.imshow('motion7', fgmask)    

# uses Knuth's 'real-time' avg/var algorithm
m = None
S = None
n = 0
def motion8(frame, counter):
    global m
    global S
    global n
    
    gray = np.float64(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if m is None: m = gray * 0
    if S is None: S = gray * 0

    prev_mean = m
    n += 1
    delta = gray - m
    m += delta / float(n)
    S += delta * (gray - prev_mean)
    cv2.imshow('motion8 mean', np.uint8(m))
    var = S / float(n)
    min = var.min()
    max = var.max()
    var1 = var / (max / 255.0)
    cv2.imshow('motion8 var', np.uint8(var1))
    # counter8 += 1
    # delta = gray - mean
    # mean += delta / float(counter8)
    # delta2 = gray - mean
    # M2 += np.multiply(delta, delta2)
    # if counter8 >= 2:
    #     var = M2 / float(counter8 - 1)
    #     min = var.min()
    #     max = var.max()
    #     print min, max
    #     var1 = np.uint8(var / (max / 255))
    #     cv2.imshow('motion8 mean', np.uint8(mean))
    #     cv2.imshow('motion8 var', np.uint8(var1))

# accumulate feature motion
accum9 = None
counter9 = None
def motion9(frame, src, dst):
    global accum9
    global counter9
    
    if accum9 is None or counter9 is None:
        print('creating accum/counter matrices')
        gray = np.float64(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        accum9 = gray * 0
        counter9 = gray + 0.01

    for i in range(len(src)):
        # print(src[i], dst[i])
        dx = dst[i][0] - src[i][0]
        dy = dst[i][1] - src[i][1]
        dist2 = dx*dx + dy*dy
        #accum9[ int(src[i][0]), int(src[i][1]) ] += dist2
        #accum9[ int(dst[i][0]), int(dst[i][1]) ] += dist2
        #counter9[ int(src[i][0]), int(src[i][1]) ] += 1
        #counter9[ int(dst[i][0]), int(dst[i][1]) ] += 1
        accum9[ int(src[i][1]), int(src[i][0]) ] += dist2
        accum9[ int(dst[i][1]), int(dst[i][0]) ] += dist2
        counter9[ int(src[i][1]), int(src[i][0]) ] += 1
        counter9[ int(dst[i][1]), int(dst[i][0]) ] += 1

    mean = accum9 / counter9
    min = mean.min()
    max = mean.max()
    print(min, max)
    var1 = mean / (max / 255.0)
    cv2.imshow('motion9', np.uint8(var1))
    
    
# for ORB
detector = cv2.ORB_create(max_features)
extractor = detector
norm = cv2.NORM_HAMMING
matcher = cv2.BFMatcher(norm)

# for Star
# detector = cv2.StarDetector(16, # maxSize
#                             20, # responseThreshold
#                             10, # lineThresholdProjected
#                             8,  # lineThresholdBinarized
#                             5, #  suppressNonmaxSize
#                             )
# extractor = cv2.DescriptorExtractor_create('ORB')
#norm = cv2.NORM_HAMMING
#matcher = cv2.BFMatcher(norm)

# for SIFT
#detector = cv2.SIFT(nfeatures=max_features, nOctaveLayers=5)
#extractor = detector
#norm = cv2.NORM_L2
#FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
#FLANN_INDEX_LSH    = 6
#flann_params = { 'algorithm': FLANN_INDEX_KDTREE,
#                 'trees': 5 }
#matcher = cv2.FlannBasedMatcher(flann_params, {}) # bug : need to pass empty dict (#1329)

accum = None
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

stop_count = 0

if not args.no_equalize:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

csvfile = open(output_csv, 'w')
fieldnames=['frame', 'time', 'rotation (deg)',
            'translation x (px)', 'translation y (px)']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

for frame in reader.nextFrame():
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    counter += 1

    filtered = []

    if counter < skip_frames:
        if counter % 1000 == 0:
            print("Skipping %d frames..." % counter)
        continue

    # print "Frame %d" % counter

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
    if not args.no_equalize:
        frame_undist = r.aeq_value(frame_undist)

    # test for building up an automatic mask
    # motion3(frame_undist, counter)

    # average frames (experiement)
    # motion6(frame_undist, counter)
    
    process_hsv = False
    if process_hsv:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2HSV)
        hue,s,v = cv2.split(hsv)
        cv2.imshow('hue', hue)
        cv2.imshow('saturation', s)
        cv2.imshow('value', v)
        gray = hue
    else:
        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        
    if not args.no_equalize:
        gray = clahe.apply(gray)

    kp_list = detector.detect(gray)
    kp_list, des_list = extractor.compute(gray, kp_list)

    # Fixme: make a command line option
    # possible values are 'homography', 'fundamental', 'essential', 'none'
    filter_method = 'homography'

    if not (des_list_last is None) and not (des_list is None) and len(des_list_last) and len(des_list):
        #print(len(des_list_last), len(des_list))
        matches = matcher.knnMatch(des_list, trainDescriptors=des_list_last, k=2)
        p1, p2, kp_pairs, idx_pairs, mkp1 = filterMatches(kp_list, kp_list_last, matches)

        # filtering by fundamental matrix would reject keypoints that
        # are not mathematically possible from one frame to the next
        # but in an anomaly detection scenerio at sea, we might only
        # have one match so this wouldn't work
        filter_fundamental = True
        if filter_fundamental:
            #print "before = ", len(p1)
            M, status, newp1, newp2 = filterFeatures(p1, p2, K, filter_method)
            filtered = []
            for i, flag in enumerate(status):
                if flag:
                    filtered.append(mkp1[i])
            #print "filtered = ", len(filtered)
        else:
            filtered = mkp1

    # motion9(frame_scale, p2, p1)
    
    affine = findAffine(p2, p1, fullAffine=False)
    (rot, tx, ty, sx, sy) = decomposeAffine(affine)
    if abs(rot) > 6:
        (rot, tx, ty, sx, sy) = (0.0, 0.0, 0.0, 1.0, 1.0)
    #print affine
    #print (rot, tx, ty, sx, sy)

    abs_rot += rot
    abs_rot *= 0.95
    abs_x += tx
    abs_x *= 0.95
    abs_y += ty
    abs_y *= 0.95
    
    print("motion: %d %.2f %.1f %.1f %.2f %.1f %.1f" % (counter, rot, tx, ty, abs_rot, abs_x, abs_y))

    translate_only = False
    rotate_translate_only = True
    if translate_only:
        rot = 0.0
        sx = 1.0
        sy = 1.0
    elif rotate_translate_only:
        sx = 1.0
        sy = 1.0

    # low pass filter our affine components
    keep = (1.0 - smooth)
    rot_avg = keep * rot_avg + smooth * rot
    tx_avg = keep * tx_avg + smooth * tx
    ty_avg = keep * ty_avg + smooth * ty
    sx_avg = keep * sx_avg + smooth * sx
    sy_avg = keep * sy_avg + smooth * sy
    print("%.2f %.2f %.2f %.4f %.4f" % (rot_avg, tx_avg, ty_avg, sx_avg, sy_avg))
    # divide tx, ty by args.scale to get a translation value
    # relative to the original movie size.
    row = {'frame': counter,
           'time': "%.4f" % (counter / fps),
           'rotation (deg)': "%.2f" % (-rot*fps*d2r),
           'translation x (px)': "%.1f" % (tx / args.scale),
           'translation y (px)': "%.1f" % (ty / args.scale)}
    writer.writerow(row)

    # try to catch bad cases
    #if math.fabs(rot) > 5.0 * math.fabs(rot_avg) and \
    #   math.fabs(tx) > 5.0 * math.fabs(tx_avg) and \
    #   math.fabs(ty) > 5.0 * math.fabs(ty_avg) and \
    #   math.fabs(sx - 1.0) > 5.0 * math.fabs(sx_avg - 1.0) and \
    #   math.fabs(sy - 1.0) > 5.0 * math.fabs(sy_avg - 1.0):
    #    rot = 0.0
    #    tx = 0.0
    #    ty = 0.0
    #    sx = 1.0
    #    sy = 1.0

    # total transforms needed
    rot_sum += rot
    tx_sum += tx
    ty_sum += ty
    sx_sum += (sx - 1.0)
    sy_sum += (sy - 1.0)

    # save left overs
    rot_sum -= rot_avg
    tx_sum -= tx_avg
    ty_sum -= ty_avg
    sx_sum = (sx_sum - sx_avg) + 1.0
    sy_sum = (sy_sum - sy_avg) + 1.0

    # catch up on left overs
    rot_catchup = catchup * rot_sum; rot_sum -= rot_catchup
    tx_catchup = catchup * tx_sum; tx_sum -= tx_catchup
    ty_catchup = catchup * ty_sum; ty_sum -= ty_catchup
    sx_catchup = catchup * (sx_sum - 1.0); sx_sum -= sx_catchup
    sy_catchup = catchup * (sy_sum - 1.0); sy_sum -= sy_catchup

    rot_delta = (rot_avg + rot_catchup) - rot_sum
    tx_delta = (tx_avg + tx_catchup) - tx_sum
    ty_delta = (ty_avg + ty_catchup) - ty_sum
    sx_delta = (sx_avg + sx_catchup - sx_sum) + 1.0
    sy_delta = (sy_avg + sy_catchup - sy_sum) + 1.0

    rot_rad = rot_delta * math.pi / 180.0
    costhe = math.cos(rot_rad)
    sinthe = math.sin(rot_rad)
    row1 = [ sx_delta * costhe, -sx_delta * sinthe, tx_delta ]
    row2 = [ sy_delta * sinthe,  sy_delta * costhe, ty_delta ]
    affine_new = np.array( [ row1, row2 ] )
    # print affine_new

    #rot_last = -(rot_delta - rot)
    #tx_last = -(tx_delta - tx)
    #ty_last = -(ty_delta - ty)
    #sx_last = -(sx_delta - sx) + 1.0
    #sy_last = -(sy_delta - sy) + 1.0

    rot_last = (rot_avg + rot_catchup)
    tx_last = (tx_avg + tx_catchup)
    ty_last = (ty_avg + ty_catchup)
    sx_last = (sx_avg - 1.0 + sx_catchup) + 1.0
    sy_last = (sy_avg - 1.0 + sy_catchup) + 1.0

    rot_rad = rot_last * math.pi / 180.0
    costhe = math.cos(rot_rad)
    sinthe = math.sin(rot_rad)
    row1 = [ sx_last * costhe, -sx_last * sinthe, tx_last ]
    row2 = [ sy_last * sinthe,  sy_last * costhe, ty_last ]
    affine_last = np.array( [ row1, row2 ] )
    # print affine_new

    rows, cols, depth = frame_undist.shape
    new_frame = cv2.warpAffine(frame_undist, affine_new, (cols,rows))

    if True:
        # FIXME
        # res1 = cv2.drawKeypoints(frame_undist, filtered, None, color=(0,255,0), flags=0)
        for kp in filtered:
            cv2.circle(frame_undist, (int(kp.pt[0]), int(kp.pt[1])), 3, (0,255,0), 1, cv2.LINE_AA)
        res1 = frame_undist

    else:
        res1 = frame_undist

    kp_list_last = kp_list
    des_list_last = des_list

    # composite with history
    if accum is None:
        accum = new_frame
    else:
        base = cv2.warpAffine(accum, affine_last, (cols, rows))
        do_motion_fade = False
        if do_motion_fade:
            motion_mask = motion2(new_frame, base)
        else:
            motion_mask = None
        accum = overlay(new_frame, base, motion_mask)

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
   
    cv2.imshow('bgr', res1)
    cv2.imshow('smooth', new_frame)
    cv2.imshow('final', final)
    #output.write(res1)
    if args.write_smooth:
        output.write(final)
    if 0xFF & cv2.waitKey(5) == 27:
        break

cv2.destroyAllWindows()

