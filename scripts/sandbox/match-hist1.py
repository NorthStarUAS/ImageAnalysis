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

from lib import histogram
from lib import project

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('project', help='project directory')
args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()
# proj.load_features(descriptors=False)
proj.load_match_pairs()

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

histogram.make_histograms(proj.image_list)
histogram.make_templates(proj.image_list, dist_cutoff=40, self_weight=0.1)

histograms = histogram.histograms
templates = histogram.templates
for image in proj.image_list:
    rgb = image.load_rgb()
    scaled = cv2.resize(rgb, (0,0), fx=0.25, fy=0.25)
    g, b, r = cv2.split(scaled)

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    src_g_quantiles = np.cumsum(histograms[image.name][0])
    src_b_quantiles = np.cumsum(histograms[image.name][1])
    src_r_quantiles = np.cumsum(histograms[image.name][2])
    src_g_quantiles /= src_g_quantiles[-1]
    src_b_quantiles /= src_b_quantiles[-1]
    src_r_quantiles /= src_r_quantiles[-1]
    
    interp_g_values = np.interp(src_g_quantiles, templates[image.name][0], np.arange(256))
    interp_b_values = np.interp(src_b_quantiles, templates[image.name][1], np.arange(256))
    interp_r_values = np.interp(src_r_quantiles, templates[image.name][2], np.arange(256))

    g = interp_g_values[g].reshape(g.shape).astype('uint8')
    b = interp_b_values[b].reshape(b.shape).astype('uint8')
    r = interp_r_values[r].reshape(r.shape).astype('uint8')

    result = cv2.merge( (g, b, r) )
    cv2.imshow('scaled', scaled)
    cv2.imshow('result', result)
    cv2.waitKey()

