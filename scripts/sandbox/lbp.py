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

# def normalize(img):
#     min = np.min(img)
#     max = np.max(img)
#     print(min, max)
#     img_norm = (img.astype('float') - min) / (max - min)
#     return img_norm

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
    #gray = (255 - dist * 256 / 90).astype('uint8')
    index = hue
elif False:
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

scale_orig = cv2.resize(rgb, (int(w*args.scale), int(h*args.scale)))
scale = scale_orig.copy()
gscale = cv2.resize(gray, (int(w*args.scale), int(h*args.scale)))

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
    if False:
        # dist histogram
        plt.figure()
        y_pos = np.arange(len(hist))
        plt.bar(y_pos, hist, align='center', alpha=0.5)
        plt.xticks(y_pos, range(len(hist)))
        plt.ylabel('count')
        plt.title('classifier')
        plt.show()
    return hist

def update_model(cell_list, model):
    labels = []
    data = []
    for key in cell_list:
        cell = cell_list[key]
        if cell["user"] != None:
            labels.append(cell["user"])
            data.append(cell["classifier"])
    if len(set(labels)) >= 2:
        print("Updating model fit...")
        model.fit(data, labels)
        print("Done.")
 
def update_prediction(cell_list, model):
    for key in cell_list:
        cell = cell_list[key]
        #(r1, r2, c1, c2) = cell["region"]
        #hist = gen_classifier(lbp, index, r1, r2, c1, c2)
        cell["prediction"] = model.predict(cell["classifier"].reshape(1, -1))

def draw(image, r1, r2, c1, c2, color, width):
    cv2.rectangle(image,
                  (int(c1*args.scale), int(r1*args.scale)),
                  (int((c2)*args.scale)-1, int((r2)*args.scale)-1),
                  color=color, thickness=width)

def draw_prediction(image, cell_list, selected_cell, show_grid, alpha=0.25):
    overlay = image.copy()
    for key in cell_list:
        cell = cell_list[key]
        (r1, r2, c1, c2) = cell["region"]
        if cell["user"] == "no":
            draw(overlay, r1, r2, c1, c2, (0,255,0), cv2.FILLED)
        elif cell["user"] == "yes":
            draw(overlay, r1, r2, c1, c2, (0,0,255), cv2.FILLED)
        elif cell["prediction"] == "no" and show_grid:
            draw(overlay, r1, r2, c1, c2, (0,255,0), 2)
        elif cell["prediction"] == "yes" and show_grid:
            draw(overlay, r1, r2, c1, c2, (0,0,255), 2)
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    if selected_cell != None:
        (r1, r2, c1, c2) = cell_list[selected_cell]["region"]
        draw(result, r1, r2, c1, c2, (255,255,255), 2)
    return result

# train a Linear SVM on the data
model = LinearSVC(max_iter=5000000)

# generate grid
rows = np.linspace(0, h, hcells).astype('int')
cols = np.linspace(0, w, wcells).astype('int')

# cell list
cell_list = {}
for j in range(len(rows)-1):
    for i in range(len(cols)-1):
        key = "%d,%d,%d,%d" % (int(rows[j]), int(rows[j+1]),
                               int(cols[i]), int(cols[i+1])) 
        cell_list[key] = { "region": (int(rows[j]), int(rows[j+1]),
                                      int(cols[i]), int(cols[i+1])),
                           "classifier": None,
                           "user": None,
                           "prediction": "no" }

# compute the classifier
for key in cell_list.keys():
    (r1, r2, c1, c2) = cell_list[key]["region"]
    cell_list[key]["classifier"] = gen_classifier(lbp, index, r1, r2, c1, c2)

selected_cell = None
show_grid = True

scale = draw_prediction(scale_orig, cell_list, selected_cell, show_grid)

win = 'scale'
cv2.imshow(win, scale)

def onmouse(event, x, y, flags, param):
    global selected_cell
    if event == cv2.EVENT_LBUTTONDOWN:
        i = np.searchsorted(cols, int(x/args.scale), side='right') - 1
        j = np.searchsorted(rows, int(y/args.scale), side='right') - 1
        # print("  cell:", (int(rows[j]), int(rows[j+1]), int(cols[i]), int(cols[i+1])))
        key = "%d,%d,%d,%d" % (int(rows[j]), int(rows[j+1]),
                               int(cols[i]), int(cols[i+1])) 
        if cell_list[key]["user"] == None:
            cell_list[key]["user"] = "yes"
        elif cell_list[key]["user"] == "yes":
            cell_list[key]["user"] = "no"
        else:
            cell_list[key]["user"] = None
        scale = draw_prediction(scale_orig, cell_list, selected_cell, show_grid)
        cv2.imshow(win, scale)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # show region detail
        i = np.searchsorted(cols, int(x/args.scale), side='right') - 1
        j = np.searchsorted(rows, int(y/args.scale), side='right') - 1
        key = "%d,%d,%d,%d" % (int(rows[j]), int(rows[j+1]),
                               int(cols[i]), int(cols[i+1]))
        selected_cell = key
        (r1, r2, c1, c2) = cell_list[key]["region"]
        rgb_region = rgb[r1:r2,c1:c2]
        cv2.imshow('region', cv2.resize(rgb_region, ( (r2-r1)*3, (c2-c1)*3) ))
        scale = draw_prediction(scale_orig, cell_list, selected_cell, show_grid)
        cv2.imshow(win, scale)

cv2.setMouseCallback(win, onmouse)

# work list
work_list = list(cell_list.keys())
random.shuffle(work_list)

index = 0
while index < len(work_list):
    key = work_list[index]
    selected_cell = key
    scale = draw_prediction(scale_orig, cell_list, selected_cell, show_grid)
    (r1, r2, c1, c2) = cell_list[key]["region"]
    print(r1, r2, c1, c2)
    rgb_region = rgb[r1:r2,c1:c2]
    cv2.imshow('gray', gscale)
    cv2.imshow('scale', scale)
    cv2.imshow('region', cv2.resize(rgb_region, ( (r2-r1)*3, (c2-c1)*3) ))
    keyb = cv2.waitKey()
    if keyb == ord('y') or keyb == ord('Y'):
        cell_list[key]["user"] = "yes"
        index += 1
        #labels.append('yes')
        #data.append(hist)
    elif keyb == ord('n') or keyb == ord('N'):
        cell_list[key]["user"] = "no"
        index += 1
        #labels.append('no')
        #data.append(hist)
    elif keyb == ord(' '):
        # pass this cell
        index += 1
    elif keyb == ord('g'):
        show_grid = not show_grid
    elif keyb == ord('f'):
        update_model(cell_list, model)
        update_prediction(cell_list, model)

if False:
    # dist histogram
    plt.figure()
    y_pos = np.arange(len(hist))
    plt.bar(y_pos, hist, align='center', alpha=0.5)
    plt.xticks(y_pos, range(len(hist)))
    plt.ylabel('count')
    plt.title('total distance histogram')

    plt.show()
