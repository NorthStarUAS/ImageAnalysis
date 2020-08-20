#!/usr/bin/env python3

import argparse
import cv2
import fnmatch
import os
import pyexiv2                  # dnf install python3-exiv2 (py3exiv2)
import sys

from props import getNode       # from aura-props package
import props_json               # from aura-props package

from lib import camera
from lib import project

parser = argparse.ArgumentParser(description='New camera configuration.')
parser.add_argument('project', help='project directory')
parser.add_argument('--config', default='../cameras', help='camera config directory')
parser.add_argument('--ccd-width', type=float)
parser.add_argument('--force', action='store_true', help='force overwrite of an existing config file')

args = parser.parse_args()

proj = project.ProjectMgr(args.project)

image_dir = args.project
image_file = None
for file in os.listdir(image_dir):
    if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG') or fnmatch.fnmatch(file, '*.tif') or fnmatch.fnmatch(file, '*.TIF'):
        image_file = os.path.join(image_dir, file)
        print("Found a suitable image:", image_file)
        break
if not image_file:
    print("No suitable *.JPG or *.jpg file found in:", image_dir)
    quit()

# auto detect camera from image meta data
camera_name, make, model, lens_model = proj.detect_camera()
camera_file = os.path.join("..", "cameras", camera_name + ".json")
print("Camera:", camera_name, camera_file)

exif = pyexiv2.ImageMetadata(image_file)
exif.read()
#for key in exif:
#    print(key)

if 'Exif.Photo.FocalLength' in exif:
    focal_len_mm = exif['Exif.Photo.FocalLength'].value
else:
    focal_len_mm = 4.0
width = 0
height = 0
if 'Exif.Photo.PixelXDimension' in exif:
    width = float(exif['Exif.Photo.PixelXDimension'].value)
elif 'Exif.Image.ImageWidth' in exif:
    width = float(exif['Exif.Image.ImageWidth'].value)
if 'Exif.Photo.PixelYDimension' in exif:
    height = float(exif['Exif.Photo.PixelYDimension'].value)
elif 'Exif.Image.ImageLength' in exif:
    height = float(exif['Exif.Image.ImageLength'].value)
if not width or not height:
    print("cannot determine image dimensions, aborting...")
    quit()

print(height, width)

# sanity check against actual image size versus meta data size
# test load the image
img = cv2.imread(image_file, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
print("image shape:", img.shape[:2])

if height != img.shape[0] or width != img.shape[1]:
    print("disagreement between exif size and actual image size")
    print("Using actual image size")
    (height, width) = img.shape[:2]
    
#base_name.replace(' ', '_')
#print('base:', base_name)

aspect_ratio = width / height
print('aspect ratio:', aspect_ratio)

if 'Xmp.drone-dji.CalibratedFocalLength' in exif:
    fx = float(exif['Xmp.drone-dji.CalibratedFocalLength'].value)
    fy = fx
    ccd_width = focal_len_mm * width / fx
    ccd_height = focal_len_mm * height / fy
elif not args.ccd_width == None:
    ccd_width = args.ccd_width
    ccd_height = args.ccd_width / aspect_ratio
    # in this system we force fx == fy, but here fy computed separately
    # just for fun.
    fx = focal_len_mm * width / args.ccd_width
    fy = focal_len_mm * height / ccd_height
else:
    print("Cannot autodetect calibrated focal length, please specify a ccd-width")
    quit()
print('ccd: %.3f x %.3f' % (ccd_width, ccd_height))
print('fx fy = %.2f %.2f' % (fx, fy))

cu = width * 0.5
cv = height * 0.5
print('cu cv = %.2f %.2f' % (cu, cv))

camera.set_defaults()
camera.set_meta(make, model, lens_model)
camera.set_lens_params(ccd_width, ccd_height, focal_len_mm)
camera.set_K(fx, fy, cu, cv)
camera.set_image_params(width, height)

if os.path.exists(camera_file):
    print("Camera config file already exists:", camera_file)
    if args.force:
        print("Overwriting ...")
    else:
        print("Use [ --force ] to overwrite ...")
        quit()

print("Saving:", camera_file)
cam_node = getNode('/config/camera', True)
props_json.save(camera_file, cam_node)
