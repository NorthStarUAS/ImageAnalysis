#!/usr/bin/python3

# create a cumulative (weighted) sum of the match/neighbor histograms
# for each iamge and use that to rerender the image.  Attempting to
# balance the look of the images and account for different
# lighting/angles/white balance, etc.

import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

from props import getNode

from lib import camera
from lib import project
from lib import smart
from lib import srtm
from lib import transformations

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('project', help='project directory')
args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()
# proj.load_features(descriptors=False)
proj.load_match_pairs()

r2d = 180 / math.pi
d2r = math.pi / 180

# lookup ned reference
ref_node = getNode("/config/ned_reference", True)
ref = [ ref_node.getFloat('lat_deg'),
        ref_node.getFloat('lon_deg'),
        ref_node.getFloat('alt_m') ]
# setup SRTM ground interpolator
srtm.initialize( ref, 6000, 6000, 30 )

smart.load(proj.analysis_dir)

# histogram matching from:
# https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
def hist_match(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

# compute the histogram for the image
def get_histogram_rgb(image):
    print(image.name)
    rgb = image.load_rgb()
    scaled = cv2.resize(rgb, (0,0), fx=0.25, fy=0.25)
    g, b, r = cv2.split(scaled)
    
    g_hist = np.bincount(g.ravel(), minlength=256)
    b_hist = np.bincount(b.ravel(), minlength=256)
    r_hist = np.bincount(r.ravel(), minlength=256)
    bins = np.arange(256)
    print(g_hist.shape, g_hist)
    return (g_hist.astype("float"),
            b_hist.astype("float"),
            r_hist.astype("float"))

print("Load and scale images (rgb):")
for i, i1 in enumerate(proj.image_list):
    i1.hist = get_histogram_rgb(i1)
    
print("Computing histogram templates:")
for i, i1 in enumerate(proj.image_list):
    src_g = None
    src_b = None
    src_r = None
    src_weights = 0.0
    
    ned1, ypr1, quat1 = i1.get_camera_pose()
    for i2_name in i1.match_list:
        i2 = proj.findImageByName(i2_name)
        ned2, ypr2, quat2 = i2.get_camera_pose()
        diff = np.array(ned2) - np.array(ned1)
        dist_m = np.linalg.norm( diff )
        if dist_m <= 1:
            weight = 1
        else:
            weight = 1 / dist_m
        print(i1.name, i2.name, dist_m, weight)
        if src_g is None:
            src_g = i2.hist[0] * weight
            src_b = i2.hist[1] * weight
            src_r = i2.hist[2] * weight
        else:
            src_g += i2.hist[0] * weight
            src_b += i2.hist[1] * weight
            src_r += i2.hist[2] * weight
        src_weights += weight

    # include ourselves at some relative weight to the surrounding pairs
    weight = 0.25 * src_weights
    if src_g is None:
        src_g = i1.hist[0] * weight
        src_b = i1.hist[1] * weight
        src_r = i1.hist[2] * weight
    else:
        src_g += i2.hist[0] * weight
        src_b += i2.hist[1] * weight
        src_r += i2.hist[2] * weight
    src_weights += weight

    # normalize
    src_g = src_g / src_weights
    src_b = src_b / src_weights
    src_r = src_r / src_weights

    # cumulative sums (normalized)
    g_quantiles = np.cumsum(src_g)
    b_quantiles = np.cumsum(src_b)
    r_quantiles = np.cumsum(src_r)
    g_quantiles /= g_quantiles[-1]
    b_quantiles /= b_quantiles[-1]
    r_quantiles /= r_quantiles[-1]
    i1.template = (g_quantiles, b_quantiles, r_quantiles)

    # plt.figure()
    # plt.plot(np.arange(256), g_quantiles, 'g')
    # plt.plot(np.arange(256), b_quantiles, 'b')
    # plt.plot(np.arange(256), r_quantiles, 'r')
    # plt.show()
    
    rgb = i1.load_rgb()
    scaled = cv2.resize(rgb, (0,0), fx=0.25, fy=0.25)
    g, b, r = cv2.split(scaled)

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    src_g_quantiles = np.cumsum(i1.hist[0])
    src_b_quantiles = np.cumsum(i1.hist[1])
    src_r_quantiles = np.cumsum(i1.hist[2])
    src_g_quantiles /= src_g_quantiles[-1]
    src_b_quantiles /= src_b_quantiles[-1]
    src_r_quantiles /= src_r_quantiles[-1]
    
    interp_g_values = np.interp(src_g_quantiles, g_quantiles, np.arange(256))
    interp_b_values = np.interp(src_b_quantiles, b_quantiles, np.arange(256))
    interp_r_values = np.interp(src_r_quantiles, r_quantiles, np.arange(256))

    g = interp_g_values[g].reshape(g.shape).astype('uint8')
    b = interp_b_values[b].reshape(b.shape).astype('uint8')
    r = interp_r_values[r].reshape(r.shape).astype('uint8')

    result = cv2.merge( (g, b, r) )
    cv2.imshow('scaled', scaled)
    cv2.imshow('result', result)
    cv2.waitKey()
    

