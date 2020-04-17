#!/usr/bin/python3

import argparse
import csv
import cv2
import fnmatch
import json
import numpy as np
import os

import navpy
from props import getNode       # aura-props

from lib import camera
from lib import project

parser = argparse.ArgumentParser(description='Chop up an image for zooniverse.')
parser.add_argument('project', help='project directory')
parser.add_argument('--output-dir', required=True, help='should (must?) be empty')
parser.add_argument('--divs', default=4, type=int, help="sub divisions in both directions")
args = parser.parse_args()

if not os.path.isdir(args.project):
    print("project needs to be a directory of images")
    quit()

if os.path.isdir(args.output_dir):
    if len(os.listdir(args.output_dir)):
        print("ideally the output directory wouldn't already contain stuff")
        quit()
else:
    os.makedirs(args.output_dir)

output_list = []

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
for file in sorted(os.listdir(args.project)):
    if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
        print("file:", file)
        base, ext = os.path.splitext(file)
        print(file, base, ext)
        path = os.path.join(args.project, file)
        rgb = cv2.imread(path, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        aeq = clahe.apply(val)
        hsv = cv2.merge((hue,sat,aeq))
        # convert back to rgb
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        h, w = rgb.shape[:2]
        xstep = w / float(args.divs)
        ystep = h / float(args.divs)
        print(h, w, xstep, ystep)
        xdivs = np.linspace(0, w, args.divs+1, endpoint=True).tolist()
        ydivs = np.linspace(0, h, args.divs+1, endpoint=True).tolist()
        print(xdivs)
        for j in range(args.divs):
            for i in range(args.divs):
                chop_name = base + "_%d%d" % (i, j) + ".JPG"
                output_list.append( chop_name )
                output = os.path.join(args.output_dir, chop_name)
                print(" ", output)
                x1 = int(xdivs[i])
                x2 = int(xdivs[i+1])
                y1 = int(ydivs[j])
                y2 = int(ydivs[j+1])
                print("%d:%d %d:%d" % (x1, x2, y1, y2))
                clip = rgb[y1:y2, x1:x2]
                cv2.imwrite(output, clip)

# write out simple csv version
filename = os.path.join(args.output_dir, "manifest.csv")
with open(filename, 'w') as f:
    fieldnames = ['filename']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for chop_file in output_list:
        writer.writerow({ "filename": chop_file } )

quit()
