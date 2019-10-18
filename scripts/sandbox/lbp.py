#!/usr/bin/python3

import argparse
import cv2
import math
import random
from skimage import feature        # pip3 install scikit-image
from sklearn.svm import LinearSVC  # pip3 install scikit-learn
import numpy as np
import matplotlib.pyplot as plt

texture_and_color = True
goal_step = 160                      # this is a tuning dial

def describe(gray, eps=1e-7):
    radius = 4
    numPoints = 8 * radius
    
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(gray, numPoints,
			               radius, method="uniform")
    cv2.imshow('gray', gray)
    cv2.imshow('lbp', lbp)
    cv2.waitKey()
    (hist, _) = np.histogram(lbp.ravel(),
			     bins=np.arange(0, numPoints + 3),
			     range=(0, numPoints + 2))
 
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    # return the histogram of Local Binary Patterns
    return hist

def normalize(img):
    min = np.min(img)
    max = np.max(img)
    print(min, max)
    img_norm = (img.astype('float') - min) / (max - min)
    return img_norm

parser = argparse.ArgumentParser(description='local binary patterns test.')
parser.add_argument('--image', required=True, help='image name')
parser.add_argument('--scale', type=float, default=0.4, help='scale image before processing')
args = parser.parse_args()

rgb = cv2.imread(args.image, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
(h, w) = rgb.shape[:2]
hcells = int(h / goal_step)
wcells = int(w / goal_step)
print(hcells, wcells)

gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
if True:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    # cv2 hue range: 0 - 179
    target_hue_value = 0          # red = 0
    t1 = np.mod((hue.astype('float') + 90), 180) - 90
    print('t1:', np.min(t1), np.max(t1))
    #cv2.imshow('t1', cv2.resize(t1, (int(w*args.scale), int(h*args.scale))))
    dist = np.abs(target_hue_value - t1)
    print('dist:', np.min(dist), np.max(dist))
    gray = (255 - dist * 256 / 90).astype('uint8')
    index = hue
elif True:
    g, b, r = cv2.split(rgb)
    g[g==0] = 1
    r[r==0] = 1
    ng = g.astype('float') / 255.0
    nr = r.astype('float') / 255.0
    index = (nr - ng) / (nr + ng)
    print("range:", np.min(index), np.max(index))
    #index[index<0.25] = -1.0
    index = ((0.5 * index + 0.5) * 255).astype('uint8')
cv2.imshow('index', cv2.resize(index, (int(w*args.scale), int(h*args.scale))))

radius = 3                      # this is a tuning dial
numPoints = 8 * radius
    
# compute the Local Binary Pattern representation
# of the image, and then use the LBP representation
# to build the histogram of patterns
lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")

scale = cv2.resize(rgb, (int(w*args.scale), int(h*args.scale)))
gscale = cv2.resize(gray, (int(w*args.scale), int(h*args.scale)))

def draw(image, r1, r2, c1, c2, color, width):
    cv2.rectangle(image,
                  (int(c1*args.scale), int(r1*args.scale)),
                  (int((c2)*args.scale)-1, int((r2)*args.scale)-1),
                  color=color, thickness=width)

def gen_classifier(lbp, index, r1, r2, c1, c2):
    lbp_region = lbp[r1:r2,c1:c2]
    (hist, _) = np.histogram(lbp_region.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    if texture_and_color:
        index_region = index[r1:r2,c1:c2]
        (index_hist, _) = np.histogram(index_region.ravel(),
                                       bins=64,
                                       range=(0, 255))
        #index_hist[0] = 0
        hist = np.concatenate((hist, index_hist), axis=None)
    return hist
 
def update_guess(image, rows, cols, scale, model):
    for j in range(len(rows)-1):
        for i in range(len(cols)-1):
            r1 = rows[j]
            r2 = rows[j+1]
            c1 = cols[i]
            c2 = cols[i+1]
            hist = gen_classifier(lbp, index, r1, r2, c1, c2)
            prediction = model.predict(hist.reshape(1, -1))
            if prediction == 'no':
                draw(scale, r1, r2, c1, c2, (0,255,0), 1)
            elif prediction == 'yes':
                draw(scale, r1, r2, c1, c2, (0,0,255), 1)

labels = []
data = []
# train a Linear SVM on the data
#model = LinearSVC(C=100.0, random_state=42)
model = LinearSVC(max_iter=5000000)

work_list = []
rows = np.linspace(0, h, hcells).astype('int')
cols = np.linspace(0, w, wcells).astype('int')
for j in range(len(rows)-1):
    for i in range(len(cols)-1):
        work_list.append( (int(rows[j]), int(rows[j+1]),
                           int(cols[i]), int(cols[i+1])) )
random.shuffle(work_list)

count = 0
for (r1, r2, c1, c2) in work_list:
    print(r1, r2, c1, c2)
    rgb_region = rgb[r1:r2,c1:c2]
    hist = gen_classifier(lbp, index, r1, r2, c1, c2)
    if False:
        # dist histogram
        plt.figure()
        y_pos = np.arange(len(hist))
        plt.bar(y_pos, hist, align='center', alpha=0.5)
        plt.xticks(y_pos, range(len(hist)))
        plt.ylabel('count')
        plt.title('classifier')
        plt.show()
    scale_copy = scale.copy()
    draw(scale_copy, r1, r2, c1, c2, (255,255,255), 2)
    cv2.imshow('gray', gscale)
    cv2.imshow('scale', scale_copy)
    cv2.imshow('region', cv2.resize(rgb_region, ( (r2-r1)*3, (c2-c1)*3) ))
    key = cv2.waitKey()
    if key == ord('y') or key == ord('Y'):
        labels.append('yes')
        data.append(hist)
    elif key == ord('n') or key == ord('N'):
        labels.append('no')
        data.append(hist)
    count += 1
    if count % 10 == 0:
        if len(set(labels)) >= 2:
            model.fit(data, labels)
            update_guess(gray, rows, cols, scale, model)
    
# get the histogram of LBP descriptors
hist = describe(gray)


if False:
    # dist histogram
    plt.figure()
    y_pos = np.arange(len(hist))
    plt.bar(y_pos, hist, align='center', alpha=0.5)
    plt.xticks(y_pos, range(len(hist)))
    plt.ylabel('count')
    plt.title('total distance histogram')

    plt.show()
