#!/usr/bin/python3

import argparse
import cv2
import math
import numpy as np
import os
import pyexiv2                  # dnf install python3-exiv2 (py3exiv2)
from tqdm import tqdm

from props import root, getNode
import props_json

from lib import Camera
from lib import Image

parser = argparse.ArgumentParser(description='Align and combine sentera images.')
parser.add_argument('--flight', help='Base directory (parent of NIR/RGB directories.')
parser.add_argument('--image', help='image name')
parser.add_argument('image1', help='image1 path')
parser.add_argument('image2', help='image1 path')
args = parser.parse_args()

def detect_camera(image_path):
    camera = ""
    exif = pyexiv2.ImageMetadata(image_path)
    exif.read()
    if 'Exif.Image.Make' in exif:
        camera = exif['Exif.Image.Make'].value
    if 'Exif.Image.Model' in exif:
        camera += '_' + exif['Exif.Image.Model'].value
    if 'Exif.Photo.LensModel' in exif:
        camera += '_' + exif['Exif.Photo.LensModel'].value
    camera = camera.replace(' ', '_')
    return camera

if args.flight and args.image:
    image1 = os.path.join(args.flight, 'NIR', args.image)
    image2 = os.path.join(args.flight, 'RGB', args.image)
else:
    image1 = args.image1
    image2 = args.image2
    
cam1 = detect_camera(image1)
cam2 = detect_camera(image2)
print(cam1)
print(cam2)

cam1_node = getNode("/camera1", True)
cam2_node = getNode("/camera2", True)

if props_json.load(os.path.join("../cameras", cam1 + ".json"), cam1_node):
    print("successfully loaded cam1 config")
if props_json.load(os.path.join("../cameras", cam2 + ".json"), cam2_node):
    print("successfully loaded cam2 config")

tmp = []
for i in range(9):
    tmp.append( cam1_node.getFloatEnum('K', i) )
K1 = np.copy(np.array(tmp)).reshape(3,3)
print("K1:", K1)

tmp = []
for i in range(5):
    tmp.append( cam1_node.getFloatEnum('dist_coeffs', i) )
dist1 = np.array(tmp)
print("dist1:", dist1)

tmp = []
for i in range(9):
    tmp.append( cam2_node.getFloatEnum('K', i) )
K2 = np.copy(np.array(tmp)).reshape(3,3)
print("K2:", K2)

tmp = []
for i in range(5):
    tmp.append( cam2_node.getFloatEnum('dist_coeffs', i) )
dist2 = np.array(tmp)
print("dist2:", dist2)

i1 = cv2.imread(image1, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
i2 = cv2.imread(image2, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
if i1 is None:
    print("Error loading:", image1)
    quit()
if i2 is None:
    print("Error loading:", image2)
    quit()

i1 = cv2.undistort(i1, K1, dist1)
i2 = cv2.undistort(i2, K1, dist1)

def imresize(src, height):
    ratio = src.shape[0] * 1.0/height
    width = int(src.shape[1] * 1.0/ratio)
    return cv2.resize(src, (width, height))

i1 = imresize(i1, 768)
i2 = imresize(i2, 768)

detector = cv2.xfeatures2d.SIFT_create()
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
#search_params = dict(checks=50)
search_params = dict()
matcher = cv2.FlannBasedMatcher(flann_params, search_params)
matches = matcher.knnMatch(des1,des2,k=5)
print("Raw matches:", len(matches))

filt_matches = []
for i, m in enumerate(tqdm(matches)):
    min_index = 0
    min_value = 99999999999999999999999.9
    for j in range(len(m)):
        p1 = np.array(kp1[m[j].queryIdx].pt)
        p2 = np.array(kp2[m[j].trainIdx].pt)
        px_dist = np.linalg.norm(p1-p2)
        px_dist = 1
        a1 = np.array(kp1[m[j].queryIdx].angle)
        a2 = np.array(kp2[m[j].trainIdx].angle)
        angle = (a1-a2+180) % 360 - 180
        #print(a1, a2, angle)
        metric = m[j].distance * px_dist*px_dist * abs(angle)
        if metric < min_value:
            min_value = metric
            min_index = j
    #print(i, min_index, kp1[m[min_index].queryIdx].pt, kp2[m[min_index].trainIdx].pt, min_value)
    
    # ratio test as per Lowe's paper
    #if m[0].distance < 0.75*m[1].distance:
    #    filt_matches.append(m[0])
    
    if min_value < 850:
        print('dist:', m[min_index].distance)
        filt_matches.append(m[min_index])
print("Filtered matches:", len(filt_matches))

if False:
    print("Filtering with matchesGMS")
    sh1 = i1.shape
    sh2 = i2.shape
    size1 = (sh1[1], sh1[0])
    size2 = (sh2[1], sh2[0])
    matches_fit = cv2.xfeatures2d.matchGMS(size1, size2, kp1, kp2, filt_matches, withRotation=False, withScale=False, thresholdFactor=5.0)
    src = []
    dst = []
    for m in matches_fit:
        src.append( kp1[m.queryIdx].pt )
        dst.append( kp2[m.trainIdx].pt )
    affine, status = \
        cv2.estimateAffinePartial2D(np.array([src]).astype(np.float32),
                                    np.array([dst]).astype(np.float32))
    H, status = cv2.findHomography(np.array([src]).astype(np.float32),
                                   np.array([dst]).astype(np.float32),
                                   cv2.LMEDS)
if True:
    print("Filtering by findHomography")
    tol = 5.0
    src = []
    dst = []
    for m in filt_matches:
        src.append( kp1[m.queryIdx].pt )
        dst.append( kp2[m.trainIdx].pt )
    H, status = cv2.findHomography(np.array([src]).astype(np.float32),
                                   np.array([dst]).astype(np.float32),
                                   cv2.RANSAC,
                                   tol)
    matches_fit = []
    for i, m in enumerate(filt_matches):
        if status[i]:
            matches_fit.append(m)
            
print("Fitted matches:", len(matches_fit))
            
# src = []
# dst = []
# for m in matches_fit:
#     src.append( kp1[m.queryIdx].pt )
#     dst.append( kp2[m.trainIdx].pt )
# affine, status = \
#     cv2.estimateAffinePartial2D(np.array([src]).astype(np.float32),
#                                 np.array([dst]).astype(np.float32))
# H, status = cv2.findHomography(np.array([src]).astype(np.float32),
#                                         np.array([dst]).astype(np.float32),
#                                         cv2.LMEDS)

print("Homography:", H)
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

# (rot, tx, ty, sx, sy) = decomposeAffine(affine)
# print("Affine:")
# print("Rotation (deg):", rot)
# print("Translation (pixels):", tx, ty)
# print("Skew:", sx, sy)

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

draw_inlier(i1, i2, kp1, kp2, matches_fit, 'ONLY_LINES')

#i1_new = cv2.warpAffine(i1, affine, (i1.shape[1], i1.shape[0]))
i1_new = cv2.warpPerspective(i1, H, (i1.shape[1], i1.shape[0]))
blend = cv2.addWeighted(i1_new, 0.5, i2, 0.5, 0)

cv2.imshow('i1', i1)
cv2.imshow('i1_new', i1_new)
cv2.imshow('i2', i2)
cv2.imshow('blend', blend)

cv2.waitKey()

i1g = cv2.cvtColor(i1_new, cv2.COLOR_BGR2GRAY)
i2g = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

if False:
    print(i1g.dtype, i2g.dtype)
    print(i1.shape, i2.shape)
    print(i1g.shape, i2g.shape)

    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=31)
    disparity = stereo.compute(i1g, i2g)
    print(disparity.dtype, disparity.shape, np.amax(disparity))
    scaled = disparity.astype('float64')*256/np.amax(disparity)
    cv2.imshow('disparity', scaled.astype('uint8'))
    cv2.waitKey()

    flow = cv2.calcOpticalFlowFarneback(i1g, i2g, None, 0.5, 5, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros_like(i1)
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',bgr)
    cv2.waitKey()
