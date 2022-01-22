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

# version 2 attempt to add in some SVC (support vector classification)
# principles

# /home/curt/Aerial Surveys/Palmer Amaranth/Houston Co, MN/mavic-33-20200812/DJI_0622_frames/img_0459.jpg

import argparse
import cv2
import numpy as np
import os

import classifier

parser = argparse.ArgumentParser(description='Leaf detector.')
parser.add_argument('image', help='image file')
args = parser.parse_args()

win = os.path.basename(args.image)
selected_contour = None

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
green_mask = cv2.inRange(hsv, (35, 0, 128), (85, 255, 255))
green_mask = cv2.erode(green_mask, kernel3, iterations=2)
green_mask = cv2.dilate(green_mask, kernel3, iterations=2)
cv2.imshow("Green Mask", green_mask)

# edge detect on value channel
#edges = cv2.Canny(val, 50, 150)   # find more edges (more sensitive)
edges = cv2.Canny(val, 100, 300) # find fewer edges (less sensitive)
#edges = cv2.Canny(green_mask, 50, 150)   # find more edges
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

if False:
    # hough circles, because ... just want to see what they look like
    print("before hough circles")
    circles = cv2. HoughCircles(leaf_mask, cv2.HOUGH_GRADIENT, 1, 20,
                                param1=50, param2=30, minRadius=5, maxRadius=50)
    print("after hough cirlces")
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        print(i, "of", len(circles[0,:]))
        # draw the outer circle
        cv2.circle(leaves, (i[0],i[1]), i[2], (0,255,0), 2)
        # draw the center of the circle
        cv2.circle(leaves, (i[0],i[1]), 2, (0,0,255), 3)
    cv2.imshow("leaves with circles", leaves)

if True:
    # watershed test/experiment

    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding step
    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    cv2.imshow("Shifted", shifted)
    
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
	                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)
    
cv2.waitKey()

print("finding contours...")
contours, hierarchy = cv2.findContours(leaf_mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

print("filtering contours...")
contour_list = []
poly_list = []
label_list = []
classifier_list = []
touched_list = []
for contour in contours:
    area = cv2.contourArea(contour)
    x,y,w,h = cv2.boundingRect(contour)
    rect_area = w*h
    extent = float(area)/rect_area
    hull = cv2.convexHull(contour)
    if len(hull) >= 3:
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
    else:
        solidity = 0
    poly = cv2.approxPolyDP(contour, 0.05*cv2.arcLength(contour,True), True)
    if len(poly) >= 3:
        poly_list.append(poly)
    if len(contour) >= 5:
        (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
        if MA > ma:
            circularity = ma / MA
        else:
            circularity = MA / ma
    else:
        circularity = 0
    if area > 100:
        contour_list.append(contour)
        classifier_list.append( [solidity, circularity, len(poly)] )
        if True:
            label_list.append(0)
        else:
            # hack guess at labels
            if circularity > 0.5:
                if solidity > 0.75:
                    label_list.append(1)
                elif area > 200:
                    label_list.append(1)
                else:
                    label_list.append(0)
            else:
                label_list.append(0)

def draw_prediction(image, contour_list, label_list, selected_contour=None):
    colors_hex = ['#ff6f0e', '#9467bd', '#1f77b4', '#d62728',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf', '#2ca02c' ]
    colors = []
    for c in colors_hex:
        r = int(c[1:3], 16)
        g = int(c[3:5], 16)
        b = int(c[5:7], 16)
        colors.append( (r, g, b) )

    print("drawing contours...")
    for i, contour in enumerate(contour_list):
        if i == selected_contour:
            color = (255, 255, 255)
        elif label_list[i] < len(colors):
            color = colors[label_list[i]]
        else:
            color = (64, 64, 64)
        cv2.drawContours(image, [contour], -1, color, 2)
        
    cv2.imshow(win, image)

def find_closest(x, y):
    print(x, y)
    min_dist = None
    min_index = None
    for i, contour in enumerate(contour_list):
        cx,cy,w,h = cv2.boundingRect(contour)
        dist = abs(cx+w*0.5 - x) + abs(cy+h*0.5 - y)
        #print(i, dist)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            min_index = i
    #print(min_index)
    return min_index

def onmouse(event, x, y, flags, params):
    global selected_contour
    if event == cv2.EVENT_LBUTTONDOWN:
        # show region detail
        i = find_closest(x, y)
        label_list[i] = not label_list[i]
        touched_list.append(i)
        draw_prediction(img, contour_list, label_list, selected_contour)

model = classifier.LeafClassifier("palmer1")
model.predict(label_list, classifier_list)
draw_prediction(img, contour_list, label_list, selected_contour)

print("win:", win)
cv2.setMouseCallback(win, onmouse)

while True:
    draw_prediction(img, contour_list, label_list, selected_contour)
    keyb = cv2.waitKey()
    # if keyb >= ord('0') and keyb <= ord('9'):
    #     if not selected_contour is None:
    #         label_list[selected_contour] = keyb - ord('0')
    #         selected_contour = None
    #         draw_prediction(img, contour_list, label_list, selected_contour)
    if keyb == ord('u'):
        # update model
        selected_contour = None
        labels = []
        classifiers = []
        for i in set(touched_list):
            labels.append(label_list[i])
            classifiers.append(classifier_list[i])
        model.update(labels, classifiers)
        model.predict(label_list, classifier_list)
        draw_prediction(img, contour_list, label_list, selected_contour)
    elif keyb == ord('q'):
        quit()
