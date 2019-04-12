#!/usr/bin/python3

import argparse
import fnmatch
import os
import pyexiv2                  # dnf install python3-exiv2 (py3exiv2)
import sys

from props import getNode       # from aura-props package
import props_json               # from aura-props package

from lib import Camera

parser = argparse.ArgumentParser(description='New camera configuration.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--config', default='../cameras', help='camera config directory')
parser.add_argument('--ccd-width', type=float)
parser.add_argument('--force', action='store_true', help='force overwrite of an existing config file')

args = parser.parse_args()

image_dir = args.project
image_file = None
for file in os.listdir(image_dir):
    if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
        image_file = os.path.join(image_dir, file)
        print("Found a suitable image:", image_file)
        break
if not image_file:
    print("No suitable *.JPG or *.jpg file found in:", image_dir)
    quit()

exif = pyexiv2.ImageMetadata(image_file)
exif.read()

make = exif['Exif.Image.Make'].value
model = exif['Exif.Image.Model'].value
base_name = make + "_" + model
if 'Exif.Photo.LensModel' in exif:
    lens_model = exif['Exif.Photo.LensModel'].value
    base_name += "_" + lens_model
else:
    lens_model = 'unknown'
if 'Exif.Photo.FocalLength' in exif:
    focal_len_mm = exif['Exif.Photo.FocalLength'].value
else:
    focal_len_mm = 4.0
width = float(exif['Exif.Photo.PixelXDimension'].value)
height = float(exif['Exif.Photo.PixelYDimension'].value)

base_name.replace(' ', '_')
print('base:', base_name)

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

cam = Camera.Camera()
cam.set_defaults()
cam.set_meta(make, model, lens_model)
cam.set_lens_params(ccd_width, ccd_height, focal_len_mm)
cam.set_K(fx, fy, cu, cv)
cam.set_image_params(width, height)

file_name = os.path.join(args.config, base_name + '.json')
if os.path.exists(file_name):
    print("Camera config file already exists:", file_name)
    if args.force:
        print("Overwriting ...")
    else:
        print("Use [ --force ] to overwrite ...")
        quit()

print("Saving:", file_name)
cam_node = getNode('/config/camera', True)
props_json.save(file_name, cam_node)
