#!/usr/bin/env python3

import argparse
import cv2
from math import sqrt
import numpy as np
import os.path
from tqdm import tqdm
import random

from props import getNode

from lib import camera
from lib import project

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
parser.add_argument('project', help='project directory')
parser.add_argument('--scale', type=float, default=0.2, help='preview scale')
parser.add_argument('--nofit', action='store_true', help='skip fitting the ideal function and just process the averate as the mask')
args = parser.parse_args()

proj = project.ProjectMgr(args.project)

# load existing images info which could include things like camera pose
proj.load_images_info()

# camera paramters
K = camera.get_K(optimized=True)
cu = K[0,2]
cv = K[1,2]
print("Project cu = %.2f  cv = %.2f:" % (cu, cv) )

vignette_avg_file = os.path.join(proj.analysis_dir,
                                 'models', 'vignette-avg.jpg')
vignette_mask_file = os.path.join(proj.analysis_dir,
                                  'models', 'vignette-mask.jpg')
if not os.path.exists(vignette_avg_file):
    # compute the 'average' of all the images in the set (more images is better)
    sum = None
    vmask = None
    count = 0
    il = list(proj.image_list)
    random.shuffle(il)
    for image in tqdm(il):
        rgb = image.load_rgb()
        if args.scale < 1.0:
            #rgb = cv2.resize(rgb, None, fx=args.scale, fy=args.scale)
            pass
        if sum is None:
            sum = np.zeros(rgb.shape, np.float32)
        sum += rgb
        count += 1
        vmask = (sum / count).astype('uint8')
        preview = cv2.resize(vmask, None, fx=args.scale, fy=args.scale)
        cv2.imshow('vmask', preview)
        cv2.waitKey(5)
        #print("blending:", image.name)
    # save our work
    vmask = (sum / count).astype('uint8')
    cv2.imshow('vmask', vmask)
    cv2.imwrite(vignette_avg_file, vmask)

vmask = cv2.imread(vignette_avg_file, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
cv2.imshow('vmask', vmask)

h, w = vmask.shape[:2]
print("shape:", h, w)

if not args.nofit:
    scale = 1.0
    #scale = args.scale
    cy = cv * scale
    cx = cu * scale
    vals = []
    print("Sampling vignette average image:")
    for x in tqdm(range(w)):
        for y in range(h):
            dx = x - cx
            dy = y - cy
            rad = sqrt(dx*dx + dy*dy) / scale
            b = vmask[y,x,0]
            g = vmask[y,x,1]
            r = vmask[y,x,2]
            vals.append( [rad, b, g, r] )

    data = np.array(vals, dtype=np.float32)

    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    def f4(x, a, b, c):
        return a*x*x*x*x + b*x*x + c

    print("computing curve fit, may take some time.")
    bopt, pcov = curve_fit(f4, data[:,0], data[:,1])
    print("blue fit coefficients:", bopt)
    gopt, pcov = curve_fit(f4, data[:,0], data[:,2])
    print("green fit coefficients:", gopt)
    ropt, pcov = curve_fit(f4, data[:,0], data[:,3])
    print("red fit coefficients:", ropt)

    plt.plot(data[:,0], data[:,3], 'bx', label='data')
    plt.plot(data[:,0], f4(data[:,0], *ropt), 'r-',
             label='fit: a=%f, b=%f, c=%f' % tuple(ropt))
    plt.xlabel('radius')
    plt.ylabel('value')
    plt.legend()
    plt.show()

    def dither(x):
        i = int(x)
        r = x - int(x)
        if np.random.rand() < r:
            i += 1
        return i

    # generate the ideal vignette mask based on polynomial fit
    w, h = camera.get_image_params()
    print("original shape:", h, w)
    vmask = np.zeros((h, w, 3), np.uint8)

    print("Generating best fit vignette mask:")
    for x in tqdm(range(w)):
        for y in range(h):
            dx = x - cu
            dy = y - cv
            rad = sqrt(dx*dx + dy*dy)
            vmask[y,x,0] = dither(f4(rad, *bopt))
            vmask[y,x,1] = dither(f4(rad, *gopt))
            vmask[y,x,2] = dither(f4(rad, *ropt))

b, g, r = cv2.split(vmask)
b = 255 - b
g = 255 - g
r = 255 - r
b -= np.amin(b)
g -= np.amin(g)
r -= np.amin(r)
vmask = cv2.merge((b, g, r))
cv2.imwrite(vignette_mask_file, vmask)
#cv2.imshow('vmask_fit', vmask)
#cv2.waitKey(0)

