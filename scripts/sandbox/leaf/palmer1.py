#!/usr/bin/env python3

# Leaf detector
# 1. Find brighter greener areas (palmer leaves are green, face up and out
#    compete surrounding vegetation.  Threshold to make a "green mask"
# 2. Edge detect the green mask, dilate/erode to merge noisy thin/small areas.
#    low gradient areas are joined into 'black' blobs.
# 3. Invert the edge mask to make low gradient blobs now white.
# 4. Combine with green mask to hide none interesting blobs, result is a
#    leaf mask.
# 5. Find contours on leaf mask, then we can do analytics on those leaf shapes.

# using gradients to find blob/areas
#https://stackoverflow.com/questions/46036172/irregular-shape-detection-and-measurement-in-python-opencv

# /home/curt/Aerial Surveys/Palmer Amaranth/Houston Co, MN/mavic-33-20200812/DJI_0622_frames/img_0459.jpg

import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Leaf detector.')
parser.add_argument('image', help='image file')
args = parser.parse_args()

img = cv2.imread(args.image)
img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow("Original", img)
h, w = img.shape[:2]

kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# create a mask for only green stuff
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, sat, val = cv2.split(hsv)
# green hue = 60
green_mask = cv2.inRange(hsv, (40, 0, 200), (80, 255, 255))
green_mask = cv2.erode(green_mask, kernel3, iterations=2)
green_mask = cv2.dilate(green_mask, kernel3, iterations=2)
cv2.imshow("Green Mask", green_mask)

# edge detect on value channel
#edges = cv2.Canny(val, 50, 150)   # find more edges
edges = cv2.Canny(green_mask, 50, 150)   # find more edges
#edges = cv2.Canny(val, 200, 600) # find fewer edges
cv2.imshow("Edges", edges)

# merge noisy/close edges together (low gradiant areas become black blobs)
for i in range(5):
    edges = cv2.dilate(edges, kernel3, iterations=2)
    edges = cv2.erode(edges, kernel3, iterations=2)
cv2.imshow("Merge Edges", edges)

inv_edges = cv2.bitwise_not(edges)
cv2.imshow("Inverted Edges", inv_edges)

# clean noise
inv_edges = cv2.erode(inv_edges, kernel3, iterations=1)
inv_edges = cv2.dilate(inv_edges, kernel3, iterations=1)
cv2.imshow("Clean Noise Edges", inv_edges)

leaf_mask = cv2.bitwise_and(inv_edges, green_mask)
cv2.dilate(leaf_mask, kernel3, iterations=1)
cv2.imshow("Leaf Mask", leaf_mask)

leaves = cv2.bitwise_and(img, img, mask=leaf_mask)
cv2.imshow("Leaves", leaves)

cv2.waitKey()

contours, hierarchy = cv2.findContours(leaf_mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
contour_list = []
poly_list = []
for contour in contours:
    area = cv2.contourArea(contour)
    x,y,w,h = cv2.boundingRect(contour)
    rect_area = w*h
    extent = float(area)/rect_area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    approx = cv2.approxPolyDP(contour, 0.05*cv2.arcLength(contour,True), True)
    if len(approx) >= 3:
        poly_list.append(approx)
    if len(contour) >= 5:
        (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
        if MA > ma:
            circularity = ma / MA
        else:
            circularity = MA / ma
    else:
        circularity = 0
    if circularity > 0.5:
        if area > 100 and solidity > 0.75:
            contour_list.append(contour)
        elif area > 200:
            contour_list.append(contour)
cv2.drawContours(img, contour_list,  -1, (255,0,0), 2)
cv2.drawContours(img, poly_list,  -1, (255,0,255), 2)
cv2.imshow('Objects Detected',img)

# Set up the simple blob detector
params = cv2.SimpleBlobDetector_Params()
print(params)
params.filterByArea = True
params.filterByCircularity = False
params.filterByColor = False
params.filterByConvexity = False
params.filterByInertia = True
params.minInertiaRatio = 0.01
#params.blobColor = 255
#minThreshold = 1
#minparams.maxThreshold = 255
params.minArea = 50
params.maxArea = 100000000
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
print(leaf_mask.shape)
keypoints = detector.detect(leaf_mask)
#print(keypoints)
#print("----")
#print(dir(keypoints[0]))
#for kp in keypoints:
    #print("(%d, %d) size=%.1f resp=%.1f" % (kp.pt[0], kp.pt[1], kp.size, kp.response))
    #cv2.circle(leaf_mask, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size), (0, 0, 255))

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(leaf_mask, keypoints,
                                      np.array([]), (0,0,255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey()

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow('Morphological gradient', gradient)

# # Binarize gradient
# lowerb = np.array([0, 0, 0])
# upperb = np.array([15, 15, 15])
# binary = cv2.inRange(gradient, lowerb, upperb)
# cv2.imshow('Binarized gradient', binary)

# # Apply green mask
# binary = cv2.bitwise_and(binary, green_mask)
# cv2.imshow("masked off", binary)

# # Cleaning up mask
# foreground = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('Cleanup up crystal foreground mask', foreground)

# cv2.waitKey()
