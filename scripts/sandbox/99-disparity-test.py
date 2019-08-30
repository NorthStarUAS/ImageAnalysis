#!/usr/bin/python3

import argparse
import cv2
import math
import numpy as np

from lib import Image

parser = argparse.ArgumentParser(description='Align and combine sentera images.')
parser.add_argument('image1', help='image1 path')
parser.add_argument('image2', help='image1 path')
args = parser.parse_args()

i1 = cv2.imread(args.image1, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
i2 = cv2.imread(args.image2, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)

if i1 is None:
    print("Error loading:", args.image1)
    quit()
if i2 is None:
    print("Error loading:", args.image2)
    quit()
    
def imresize(src, height):
    ratio = src.shape[0] * 1.0/height
    width = int(src.shape[1] * 1.0/ratio)
    return cv2.resize(src, (width, height))

i1 = imresize(i1, 1000)
i2 = imresize(i2, 1000)

detector = cv2.xfeatures2d.SURF_create()
kp1 = detector.detect(i1)
kp2 = detector.detect(i2)
print("Keypoints:", len(kp1), len(kp2))

kp1, des1 = detector.compute(i1, kp1)
kp2, des2 = detector.compute(i2, kp2)
print("Descriptors:", len(des1), len(des2))

FLANN_INDEX_KDTREE = 1
flann_params = {
    'algorithm': FLANN_INDEX_KDTREE,
    'trees': 5
}
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(flann_params, search_params)
matches = matcher.knnMatch(des1,des2,k=2)
print("Raw matches:", len(matches))

# ratio test as per Lowe's paper
filt_matches = []
for i, m in enumerate(matches):
    if m[0].distance < 0.7*m[1].distance:
        filt_matches.append(m[0])
sh1 = i1.shape
sh2 = i2.shape
size1 = (sh1[1], sh1[0])
size2 = (sh2[1], sh2[0])
matchesGMS = cv2.xfeatures2d.matchGMS(size1, size2, kp1, kp2, filt_matches, withRotation=False, withScale=False)
print("GMS matches:", len(matchesGMS))

src = []
dst = []
for m in matchesGMS:
    src.append( kp1[m.queryIdx].pt )
    dst.append( kp2[m.trainIdx].pt )
affine, status = \
    cv2.estimateAffinePartial2D(np.array([src]).astype(np.float32),
                                np.array([dst]).astype(np.float32))

def decomposeAffine(affine):
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

(rot, tx, ty, sx, sy) = decomposeAffine(affine)
print(' ', rot, tx, ty, sx, sy)

def draw_inlier(src1, src2, kpt1, kpt2, inlier, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == 'ONLY_LINES':
        for i in range(len(inlier)):
            left = kpt1[inlier[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kpt2[inlier[i].trainIdx].pt, (src1
.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 
255, 255))

    elif drawing_type == 'LINES_AND_POINTS':
        for i in range(len(inlier)):
            left = kpt1[inlier[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kpt2[inlier[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))
        for i in range(len(inlier)):
            left = kpt1[inlier[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kpt2[inlier[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    cv2.imshow('show', output)
    cv2.waitKey()

draw_inlier(i1, i2, kp1, kp2, matchesGMS, 'ONLY_LINES')

i1_new = cv2.warpAffine(i1, affine, (i1.shape[1], i1.shape[0]))
blend = cv2.addWeighted(i1_new, 0.5, i2, 0.5, 0)

cv2.imshow('i1', i1)
cv2.imshow('i1_new', i1_new)
cv2.imshow('i2', i2)
cv2.imshow('blend', blend)

cv2.waitKey()

i1g = cv2.cvtColor(i1_new, cv2.COLOR_BGR2GRAY)
i2g = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

print(i1g.dtype, i2g.dtype)
print(i1.shape, i2.shape)
print(i1g.shape, i2g.shape)

if False:
    # cv2.StereoSGBM_MODE_SGBM      
    # cv2.StereoSGBM_MODE_HH
    # cv2.StereoSGBM_MODE_SGBM_3WAY 
    # cv2.StereoSGBM_MODE_HH4
    stereo = cv2.StereoSGBM_create(numDisparities=32, blockSize=3)
    stereo.setMinDisparity(-16)
    stereo.setMode(cv2.StereoSGBM_MODE_HH4)
    #stereo.setTextureThreshold(10)  # doesn't seem to do much?
    #stereo.setUniquenessRatio(1)    # smaller passes through more data/noise
    #stereo.setDisp12MaxDiff(-1)
    #stereo.setPreFilterType(0)      # default is 1 (xsobel)
    #stereo.setPreFilterSize(21)     # 9
    #stereo.setSmallerBlockSize(1)   # 0
    disparity = stereo.compute(i1g, i2g)
    print(disparity.dtype, disparity.shape, np.amax(disparity))
    min = np.amin(disparity)
    max = np.amax(disparity)
    spread = max - min
    scaled = (disparity.astype('float64') + min) * 255/spread
    cv2.imshow('disparity', scaled.astype('uint8'))
    cv2.waitKey()

#flow = cv2.calcOpticalFlowFarneback(i1g, i2g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow = cv2.calcOpticalFlowFarneback(i1g, i2g, None, 0.5, 5, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv = np.zeros_like(i1)
hsv[...,1] = 255
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imshow('frame2',bgr)
cv2.waitKey()
