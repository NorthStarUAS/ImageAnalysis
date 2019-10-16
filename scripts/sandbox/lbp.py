#!/usr/bin/python3

import argparse
import cv2
import random
from skimage import feature        # pip3 install scikit-image
from sklearn.svm import LinearSVC  # pip3 install scikit-learn
import numpy as np
import matplotlib.pyplot as plt

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
if True:
    g, b, r = cv2.split(rgb)
    g[g==0] = 1
    r[r==0] = 1
    #ng = normalize(g)
    #nr = normalize(r)
    num = r.astype('float') - g.astype('float')
    den = r.astype('float') + g.astype('float')
    gray = (255 * normalize( num / den) ).astype('uint8')
else:
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

radius = 3                      # this is a tuning dial
numPoints = 8 * radius
    
# compute the Local Binary Pattern representation
# of the image, and then use the LBP representation
# to build the histogram of patterns
lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")

scale = cv2.resize(rgb, (int(w*args.scale), int(h*args.scale)))
gscale = cv2.resize(gray, (int(w*args.scale), int(h*args.scale)))

def draw(image, col, row, step, color, width):
    cv2.rectangle(image,
                  (int(col*args.scale), int(row*args.scale)),
                  (int((col+step)*args.scale)-1, int((row+step)*args.scale)-1),
                  color=color, thickness=width)

def update_guess(image, step, scale, model):
    row = 0
    (h, w) = image.shape[:2]
    while row + step < h:
        col = 0
        while col + step < w:
            lbp_region = lbp[row:row+step,col:col+step]
            (hist, _) = np.histogram(lbp_region.ravel(),
                                     bins=np.arange(0, numPoints + 3),
                                     range=(0, numPoints + 2))
            prediction = model.predict(hist.reshape(1, -1))
            if prediction == 'no':
                draw(scale, col, row, step, (0,255,0), 1)
            elif prediction == 'palmer':
                draw(scale, col, row, step, (0,0,255), 1)
            col += step
        row += step

labels = []
data = []
# train a Linear SVM on the data
#model = LinearSVC(C=100.0, random_state=42)
model = LinearSVC(max_iter=5000)

(h, w) = gray.shape[:2]
print(h, w)
row = 0
step = 192                      # this is a tuning dial
work_list = []
while row + step < h:
    col = 0
    while col + step < w:
        work_list.append( (row, col) )
        col += step
    row += step
random.shuffle(work_list)

for (row, col) in work_list:
    print(row, col)
    lbp_region = lbp[row:row+step,col:col+step]
    rgb_region = rgb[row:row+step,col:col+step]
    (hist, _) = np.histogram(lbp_region.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    scale_copy = scale.copy()
    draw(scale_copy, col, row, step, (255,255,255), 2)
    cv2.imshow('gray', gscale)
    cv2.imshow('scale', scale_copy)
    cv2.imshow('region', cv2.resize(rgb_region, (step*2, step*2)))
    key = cv2.waitKey()
    if key == ord('y') or key == ord('Y'):
        labels.append('palmer')
        data.append(hist)
        if len(set(labels)) >= 2:
            model.fit(data, labels)
            update_guess(gray, step, scale, model)
    elif key == ord('n') or key == ord('N'):
        labels.append('no')
        data.append(hist)
        if len(set(labels)) >= 2:
            model.fit(data, labels)
            update_guess(gray, step, scale, model)
    
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
