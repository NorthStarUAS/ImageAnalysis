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
parser.add_argument('--model', help='saved learning model name')
args = parser.parse_args()

rgb = cv2.imread(args.image, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)

tmodel = classifier.Classifier()
if args.model:
    tmodel.init_model(args.model + ".fit", args.model + ".data")
else:
    texture_mode = init_model()
tmodel.compute_lbp(rgb)
tmodel.compute_grid()

# gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
# if False:
#     lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     # cv2 hue range: 0 - 179
#     # target_hue_value = 0          # red = 0
#     # t1 = np.mod((hue.astype('float') + 90), 180)
#     # print('t1:', np.min(t1), np.max(t1))
#     # #cv2.imshow('t1', cv2.resize(t1, (int(w*args.scale), int(h*args.scale))))
#     # dist = np.abs(90 - t1)
#     # print('dist:', np.min(dist), np.max(dist))
#     # t2 = 255 - (dist*dist) * (255 / 90)
#     # t2[t2<0] = 0
#     # weight = (hue.astype('float')/255) * (sat.astype('float')/255)
#     # index = (t2 * weight).astype('uint8')
#     index = a
# elif False:
#     hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
#     hue, sat, val = cv2.split(hsv)
#     # cv2 hue range: 0 - 179
#     target_hue_value = 0          # red = 0
#     t1 = np.mod((hue.astype('float') + 90), 180)
#     print('t1:', np.min(t1), np.max(t1))
#     #cv2.imshow('t1', cv2.resize(t1, (int(w*args.scale), int(h*args.scale))))
#     dist = np.abs(90 - t1)
#     print('dist:', np.min(dist), np.max(dist))
#     t2 = 255 - (dist*dist) * (255 / 90)
#     t2[t2<0] = 0
#     weight = (hue.astype('float')/255) * (sat.astype('float')/255)
#     index = (t2 * weight).astype('uint8')
#     #index = hue
# elif False:
#     # very dark pixels can map out noisily
#     g, b, r = cv2.split(rgb)
#     g[g==0] = 1
#     r[r==0] = 1
#     ng = g.astype('float') / 255.0
#     nr = r.astype('float') / 255.0
#     index = (nr - ng) / (nr + ng)
#     print("range:", np.min(index), np.max(index))
#     #index[index<0.5] = -1.0
#     index = ((0.5 * index + 0.5) * 255).astype('uint8')
# elif True:
#     # very dark pixels can map out noisily
#     g, b, r = cv2.split(rgb)
#     g[g==0] = 1                 # protect against divide by zero
#     ratio = (r / g).astype('float') * 0.25
#     # knock out the low end
#     lum = gray.astype('float') / 255
#     lumf = lum / 0.15
#     lumf[lumf>1] = 1
#     ratio *= lumf
#     #ratio[ratio<0.5] = 0
#     ratio[ratio>1] = 1
#     gray = (ratio*255).astype('uint8')
#     index = gray
#     print("range:", np.min(index), np.max(index))

(h, w) = tmodel.index.shape[:2]
# cv2.imshow('index', cv2.resize(tmodel.index, (int(w*args.scale), int(h*args.scale))))
scale_orig = cv2.resize(rgb, (int(w*args.scale), int(h*args.scale)))
scale = scale_orig.copy()
gscale = cv2.cvtColor(scale, cv2.COLOR_BGR2GRAY)

# def gen_classifier(lbp, index, r1, r2, c1, c2):
#     lbp_region = lbp[r1:r2,c1:c2]
#     (hist, _) = np.histogram(lbp_region.ravel(),
#                              bins=np.arange(0, numPoints + 3),
#                              range=(0, numPoints + 2))
#     if texture_and_color:
#         index_region = index[r1:r2,c1:c2]
#         (index_hist, _) = np.histogram(index_region.ravel(),
#                                        bins=64,
#                                        range=(0, 255))
#         #index_hist[0] = 0
#         hist = np.concatenate((hist, index_hist), axis=None)
#     if False:
#         # dist histogram
#         plt.figure()
#         y_pos = np.arange(len(hist))
#         plt.bar(y_pos, hist, align='center', alpha=0.5)
#         plt.xticks(y_pos, range(len(hist)))
#         plt.ylabel('count')
#         plt.title('classifier')
#         plt.show()
#     return hist

def draw(image, r1, r2, c1, c2, color, width):
    cv2.rectangle(image,
                  (int(c1*args.scale), int(r1*args.scale)),
                  (int((c2)*args.scale)-1, int((r2)*args.scale)-1),
                  color=color, thickness=width)

def draw_prediction(image, cell_list, selected_cell, show_grid, alpha=0.25):
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
    for key in cell_list:
        cell = cell_list[key]
        (r1, r2, c1, c2) = cell["region"]
        if show_grid == "user" and cell["user"] != None:
            index = ord(cell["user"]) - ord('0')
            color = colors[index]
            draw(overlay, r1, r2, c1, c2, color, cv2.FILLED)
        elif show_grid == "prediction" and cell["prediction"] != None:
            index = ord(cell["prediction"][0]) - ord('0')
            if index > 0 and abs(cell["score"][0]) > 0.25:
                color = colors[index]
                draw(overlay, r1, r2, c1, c2, color, cv2.FILLED)
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    if show_grid == "prediction":
        overlay = result.copy()
        for key in cell_list:
            cell = cell_list[key]
            (r1, r2, c1, c2) = cell["region"]
            if cell["user"] != None:
                color = colors[ord(cell["user"]) - ord('0')]
                draw(overlay, r1, r2, c1, c2, color, 2)
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    if selected_cell != None:
        (r1, r2, c1, c2) = cell_list[selected_cell]["region"]
        draw(result, r1, r2, c1, c2, (255,255,255), 2)
    return result

selected_cell = None
show_grid = "user"

win = 'scale'
scale = draw_prediction(scale_orig, tmodel.cells, selected_cell, show_grid)
cv2.imshow(win, scale)

def onmouse(event, x, y, flags, param):
    global selected_cell
    if event == cv2.EVENT_LBUTTONDOWN:
        # show region detail
        key = tmodel.find_key(int(x/args.scale), int(y/args.scale))
        selected_cell = key
        (r1, r2, c1, c2) = tmodel.cells[key]["region"]
        rgb_region = rgb[r1:r2,c1:c2]
        cv2.imshow('region', cv2.resize(rgb_region, ( (r2-r1)*3, (c2-c1)*3) ))
        scale = draw_prediction(scale_orig, tmodel.cells, selected_cell, show_grid)
        cv2.imshow(win, scale)
    elif event == cv2.EVENT_RBUTTONDOWN:
        key = tmodel.find_key(int(x/args.scale), int(y/args.scale))
        #if cell_list[key]["user"] == None:
        #    cell_list[key]["user"] = "yes"
        #elif cell_list[key]["user"] == "yes":
        #    cell_list[key]["user"] = "no"
        #else:
        #    cell_list[key]["user"] = None
        scale = draw_prediction(scale_orig, tmodel.cells, selected_cell, show_grid)
        cv2.imshow(win, scale)

cv2.setMouseCallback(win, onmouse)

# work list
work_list = list(tmodel.cells.keys())
random.shuffle(work_list)

index = 0
while index < len(work_list):
    key = work_list[index]
    selected_cell = key
    scale = draw_prediction(scale_orig, tmodel.cells, selected_cell, show_grid)
    (r1, r2, c1, c2) = tmodel.cells[key]["region"]
    print(r1, r2, c1, c2)
    rgb_region = rgb[r1:r2,c1:c2]
    cv2.imshow('gray', gscale)
    cv2.imshow('scale', scale)
    cv2.imshow('region', cv2.resize(rgb_region, ( (r2-r1)*3, (c2-c1)*3) ))
    keyb = cv2.waitKey()
    if keyb >= ord('0') and keyb <= ord('9'):
        tmodel.cells[selected_cell]["user"] = chr(keyb)
        if key == selected_cell:
            index += 1
    elif keyb == ord(' '):
        # pass this cell
        index += 1
    elif keyb == ord('g'):
        if show_grid == "user":
            show_grid = "prediction"
        elif show_grid == "prediction":
            show_grid = "none"
        elif show_grid == "none":
            show_grid = "user"
    elif keyb == ord('f'):
        tmodel.update_model()
        tmodel.update_prediction()
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
