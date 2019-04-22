#!/usr/bin/python3

import argparse
import cv2
import math
import numpy as np
import os.path
from progress.bar import Bar

from props import getNode

from lib import ProjectMgr

# for all the images in the project image_dir, detect features using the
# specified method and parameters
#
# Suggests censure/star has good stability between images (highest
# likelihood of finding a match in the target features set:
# http://computer-vision-talks.com/articles/2011-01-04-comparison-of-the-opencv-feature-detection-algorithms/
#
# Suggests censure/star works better than sift in outdoor natural
# environments: http://www.ai.sri.com/~agrawal/isrr.pdf
#
# Basic description of censure/star algorithm: http://www.researchgate.net/publication/221304099_CenSurE_Center_Surround_Extremas_for_Realtime_Feature_Detection_and_Matching

parser = argparse.ArgumentParser(description='I want to vignette.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--scale', type=float, default=0.2, help='working scale')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)

# load existing images info which could include things like camera pose
proj.load_images_info()

# def imresize(src, height):
#     ratio = src.shape[0] * 1.0/height
#     width = int(src.shape[1] * 1.0/ratio)
#     return cv2.resize(src, (width, height))

vignette_file = os.path.join(proj.analysis_dir, 'vignette.jpg')
if not os.path.exists(vignette_file):
    # compute the 'average' of all the images in the set (more images is better)
    sum = None
    vmask = None
    count = 0
    for image in proj.image_list:
        rgb = image.load_rgb()
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if args.scale < 1.0:
            gray = cv2.resize(gray, None, fx=args.scale, fy=args.scale)
        if sum is None:
            sum = np.zeros(gray.shape, np.float32)
        sum += gray
        count += 1
        vmask = (sum / count).astype('uint8')
        cv2.imshow('vmask', vmask)
        print(image.name, np.amin(vmask), '-', np.amax(vmask))
        if 0xFF & cv2.waitKey(5) == 27:
            break
    # save our work
    cv2.imwrite(vignette_file, vmask)

vmask = cv2.imread(vignette_file, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
cv2.imshow('vmask', vmask)

h, w = vmask.shape[:2]
print("shape:", h, w)

cy = h/2
cx = w/2
vals = []
for x in range(w):
    print(x)
    for y in range(h):
        dx = x - cx
        dy = y - cy
        r = math.sqrt(dx*dx + dy*dy) / args.scale
        v = vmask[y,x]
        vals.append( [r, v] )

data = np.array(vals, dtype=np.float32)
fit, res, _, _, _ = np.polyfit( data[:,1], data[:,0], 4, full=True )
print(fit)

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def f4(x, a, b, c):
    return a*x*x*x*x + b*x*x + c

popt, pcov = curve_fit(f4, data[:,0], data[:,1])
print("fit coefficients:", popt)

def dither(x):
    i = int(x)
    r = x - int(x)
    if np.random.rand() < r:
        i += 1
    return i
    
# generate the ideal vignette mask based on polynomial fit
for x in range(w):
    print(x)
    for y in range(h):
        dx = x - cx
        dy = y - cy
        r = math.sqrt(dx*dx + dy*dy) / args.scale
        vmask[y,x] = dither(f4(r, *popt))
cv2.imshow('vmask_fit', vmask)
cv2.waitKey(0)

plt.plot(data[:,0], data[:,1], 'b-', label='data')
plt.plot(data[:,0], f4(data[:,0], *popt), 'r-',
         label='fit: a=%f, b=%f, c=%f' % tuple(popt))
plt.xlabel('radius')
plt.ylabel('value')
plt.legend()
plt.show()
