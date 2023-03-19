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

parser = argparse.ArgumentParser(description='Preprocessing for simple weighted histogram equalizationr.')
parser.add_argument('project', help='project directory')
args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()

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

if not histogram.load(proj.analysis_dir):
    histogram.make_histograms(proj.image_list)
    
histogram.make_templates(proj.image_list, dist_cutoff=50, self_weight=1.0)
histogram.save(proj.analysis_dir)
    
histograms = histogram.histograms
templates = histogram.templates
for image in proj.image_list:
    rgb = image.load_rgb()
    scaled = cv2.resize(rgb, (0,0), fx=0.25, fy=0.25)
    result = histogram.match_neighbors(scaled, image.name)
    cv2.imshow('scaled', scaled)
    cv2.imshow('result', result)
    cv2.waitKey()

