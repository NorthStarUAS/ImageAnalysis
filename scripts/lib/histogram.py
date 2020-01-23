# much of this approach borrows heavily from the example here: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
#
# Thank you ali_m whoever you are!

import cv2
import numpy as np
import os
import pickle

histograms = {}
templates = {}

# compute the g, b, and r histograms for the image (a scaling
# parameter can be set to improve performance at a tiny loss of
# resolution)
def get_histogram_rgb(image, scale=0.25):
    print(image.name)
    rgb = image.load_rgb()
    scaled = cv2.resize(rgb, (0,0), fx=scale, fy=scale)
    g, b, r = cv2.split(scaled)
    
    g_hist = np.bincount(g.ravel(), minlength=256)
    b_hist = np.bincount(b.ravel(), minlength=256)
    r_hist = np.bincount(r.ravel(), minlength=256)
    bins = np.arange(256)
    return (g_hist.astype("float32"),
            b_hist.astype("float32"),
            r_hist.astype("float32"))

def make_histograms(image_list):
    print("Generating individual histograms...")
    for image in image_list:
        histograms[image.name] = get_histogram_rgb(image)
 
# compute the histogram templates as a distance weighted average of
# the surrounding image histograms
def make_templates(image_list, dist_cutoff=40, self_weight=0.1):
    print("Computing histogram templates:")
    for i, i1 in enumerate(image_list):
        print(i1.name)
        src_g = None
        src_b = None
        src_r = None
        src_weights = 0.0

        ned1, ypr1, quat1 = i1.get_camera_pose()
        for j, i2 in enumerate(image_list):
            if i == j:
                continue
            ned2, ypr2, quat2 = i2.get_camera_pose()
            diff = np.array(ned2) - np.array(ned1)
            dist_m = np.linalg.norm( diff )
            if dist_m > dist_cutoff:
                continue
            if dist_m <= 1:
                weight = 1
            else:
                weight = 1 / dist_m
            #print(i1.name, i2.name, dist_m, weight)
            if src_g is None:
                src_g = histograms[i2.name][0] * weight
                src_b = histograms[i2.name][1] * weight
                src_r = histograms[i2.name][2] * weight
            else:
                src_g += histograms[i2.name][0] * weight
                src_b += histograms[i2.name][1] * weight
                src_r += histograms[i2.name][2] * weight
            src_weights += weight

        # include ourselves at some relative weight to the surrounding pairs
        weight = self_weight * src_weights
        if src_g is None:
            src_g = histograms[i1.name][0] * weight
            src_b = histograms[i1.name][1] * weight
            src_r = histograms[i1.name][2] * weight
        else:
            src_g += histograms[i1.name][0] * weight
            src_b += histograms[i1.name][1] * weight
            src_r += histograms[i1.name][2] * weight
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
        templates[i1.name] = (g_quantiles, b_quantiles, r_quantiles)

def match_neighbors(rgb, image_name):
    g, b, r = cv2.split(rgb)

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    src_g_quantiles = np.cumsum(histograms[image_name][0])
    src_b_quantiles = np.cumsum(histograms[image_name][1])
    src_r_quantiles = np.cumsum(histograms[image_name][2])
    src_g_quantiles /= src_g_quantiles[-1]
    src_b_quantiles /= src_b_quantiles[-1]
    src_r_quantiles /= src_r_quantiles[-1]
    
    interp_g_values = np.interp(src_g_quantiles, templates[image_name][0], np.arange(256))
    interp_b_values = np.interp(src_b_quantiles, templates[image_name][1], np.arange(256))
    interp_r_values = np.interp(src_r_quantiles, templates[image_name][2], np.arange(256))

    g = interp_g_values[g].reshape(g.shape).astype('uint8')
    b = interp_b_values[b].reshape(b.shape).astype('uint8')
    r = interp_r_values[r].reshape(r.shape).astype('uint8')

    result = cv2.merge( (g, b, r) )
    return result

def load(analysis_dir):
    global histograms
    global templates
    hist_file = os.path.join(analysis_dir, "histogram")
    if os.path.isfile(hist_file):
        (histograms, templates) = pickle.load(open(hist_file, "rb"))
        return True
    else:
        return False

def save(analysis_dir):
    hist_file = os.path.join(analysis_dir, "histogram")
    pickle.dump((histograms, templates), open(hist_file, "wb"))
    
