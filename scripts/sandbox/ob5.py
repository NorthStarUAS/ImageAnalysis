#!/usr/bin/python3

import argparse
import cv2
import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import classifier

texture_and_color = False
# goal_step = 160                      # this is a tuning dial

parser = argparse.ArgumentParser(description='local binary patterns test.')
parser.add_argument('--image', required=True, help='image name')
parser.add_argument('--scale', type=float, default=0.4, help='scale image before processing')
# parser.add_argument('--model', help='saved learning model name')
args = parser.parse_args()

rgb = cv2.imread(args.image, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
(h, w) = rgb.shape[:2]

# texture based classifier
model = classifier.Classifier()
model.init_model(basename="ob5")
#model.compute_lbp(rgb, radius=3)
model.compute_redness(rgb)
model.compute_grid(grid_size=128)
cv2.imshow('model', cv2.resize(model.index.astype('uint8'), (int(w*args.scale), int(h*args.scale))))
#model.update_prediction()

# cv2.imshow('index', cv2.resize(model.index, (int(w*args.scale), int(h*args.scale))))
scale_orig = cv2.resize(rgb, (int(w*args.scale), int(h*args.scale)))
scale = scale_orig.copy()
gscale = cv2.cvtColor(scale, cv2.COLOR_BGR2GRAY)

def draw(image, r1, r2, c1, c2, color, width):
    cv2.rectangle(image,
                  (int(c1*args.scale), int(r1*args.scale)),
                  (int((c2)*args.scale)-1, int((r2)*args.scale)-1),
                  color=color, thickness=width)

def draw_prediction(image, cells, selected_cell, show_mode, alpha=0.25):
    cutoff = 0.05
    #colors_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors_hex = ['#2ca02c', '#ff6f0e', '#9467bd', '#1f77b4', '#d62728', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = []
    for c in colors_hex:
        r = int(c[1:3], 16)
        g = int(c[3:5], 16)
        b = int(c[5:7], 16)
        colors.append( (r, g, b) )
    overlay = image.copy()
    for key in cells:
        cell = cells[key]
        (r1, r2, c1, c2) = cell["region"]
        if show_mode == "user" and cell["user"] != None:
            color = colors[cell["user"]]
            draw(overlay, r1, r2, c1, c2, color, cv2.FILLED)
        elif show_mode == "model" and cell["prediction"] != None:
            index = cell["prediction"]
            if index >= 0 and abs(cell["score"]) >= cutoff:
                color = colors[index]
                draw(overlay, r1, r2, c1, c2, color, cv2.FILLED)
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    if show_mode != "none" and show_mode != "user":
        overlay = result.copy()
        for key in cells:
            cell = cells[key]
            (r1, r2, c1, c2) = cell["region"]
            if cell["user"] != None:
                color = colors[cell["user"]]
                draw(overlay, r1, r2, c1, c2, color, 2)
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    if selected_cell != None:
        (r1, r2, c1, c2) = cells[selected_cell]["region"]
        draw(result, r1, r2, c1, c2, (255,255,255), 2)
    return result

selected_cell = None
show_modes = ["none", "user", "model"]
show_mode = "none"
show_index = 0

win = 'scale'
scale = draw_prediction(scale_orig, model.cells, selected_cell, show_mode)
cv2.imshow(win, scale)

def onmouse(event, x, y, flags, param):
    global selected_cell
    if event == cv2.EVENT_LBUTTONDOWN:
        # show region detail
        key = model.find_key(int(x/args.scale), int(y/args.scale))
        selected_cell = key
        (r1, r2, c1, c2) = model.cells[key]["region"]
        rgb_region = rgb[r1:r2,c1:c2]
        cv2.imshow('region', cv2.resize(rgb_region, ( (r2-r1)*3, (c2-c1)*3) ))
        scale = draw_prediction(scale_orig, model.cells,
                                selected_cell, show_mode)
        cv2.imshow(win, scale)
    elif event == cv2.EVENT_RBUTTONDOWN:
        key = model.find_key(int(x/args.scale), int(y/args.scale))
        #if cells[key]["user"] == None:
        #    cells[key]["user"] = "yes"
        #elif cells[key]["user"] == "yes":
        #    cells[key]["user"] = "no"
        #else:
        #    cells[key]["user"] = None
        scale = draw_prediction(scale_orig, model.cells,
                                selected_cell, show_mode)
        cv2.imshow(win, scale)

cv2.setMouseCallback(win, onmouse)

# work list
work_list = list(model.cells.keys())
random.shuffle(work_list)

index = 0
while index < len(work_list):
    key = work_list[index]
    selected_cell = key
    scale = draw_prediction(scale_orig, model.cells, selected_cell, show_mode)
    (r1, r2, c1, c2) = model.cells[key]["region"]
    print(r1, r2, c1, c2)
    rgb_region = rgb[r1:r2,c1:c2]
    #cv2.imshow('gray', gscale)
    cv2.imshow('scale', scale)
    cv2.imshow('region', cv2.resize(rgb_region, ( (r2-r1)*3, (c2-c1)*3) ))
    keyb = cv2.waitKey()
    if keyb >= ord('0') and keyb <= ord('9'):
        model.cells[selected_cell]["user"] = keyb - ord('0')
        if key == selected_cell:
            index += 1
    elif keyb == ord(' '):
        # pass this cell
        index += 1
    elif keyb == ord('g'):
        show_index = (show_index + 1) % len(show_modes)
        show_mode = show_modes[show_index]
        print("Show:", show_mode)
    elif keyb == ord('f'):
        model.update_model()
        model.update_prediction()
    elif keyb == ord('q'):
        quit()

# if False:
#     # dist histogram
#     plt.figure()
#     y_pos = np.arange(len(hist))
#     plt.bar(y_pos, hist, align='center', alpha=0.5)
#     plt.xticks(y_pos, range(len(hist)))
#     plt.ylabel('count')
#     plt.title('total distance histogram')

#     plt.show()
