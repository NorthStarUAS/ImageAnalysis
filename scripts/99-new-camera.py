#!/usr/bin/python3

import argparse
import os
import pyexiv2                  # dnf install python3-exiv2 (py3exiv2)
import sys

from props import getNode       # from aura-props package
import props_json               # from aura-props package

sys.path.append('../lib')
import Camera

parser = argparse.ArgumentParser(description='New camera configuration.')
parser.add_argument('--config', default='../cameras', help='camera config directory')
parser.add_argument('--ccd-width', required=True, type=float)
parser.add_argument('--image', required=True, help='sample image from this camera with exif tags')
parser.add_argument('--force', action='store_true', help='force overwrite of an existing config file')

args = parser.parse_args()

exif = pyexiv2.ImageMetadata(args.image)
exif.read()

make = exif['Exif.Image.Make'].value
model = exif['Exif.Image.Model'].value
lens_model = exif['Exif.Photo.LensModel'].value
focal_len_mm = exif['Exif.Photo.FocalLength'].value
width = exif['Exif.Photo.PixelXDimension'].value
height = exif['Exif.Photo.PixelYDimension'].value

base_name = (make + "_" + model + "_" + lens_model).replace(' ', '_')
print('base:', base_name)

aspect_ratio = float(width) / float(height)
print('aspect ratio:', aspect_ratio)

ccd_height = args.ccd_width / aspect_ratio
print('ccd: %.3f x %.3f' % (args.ccd_width, ccd_height))

# in this system we force fx == fy, but here fy computed separately
# just for fun.
fx = focal_len_mm * float(width) / args.ccd_width
fy = focal_len_mm * float(height) / ccd_height
print('fx fy = %.2f %.2f' % (fx, fy))

cu = width * 0.5
cv = height * 0.5
print('cu cv = %.2f %.2f' % (cu, cv))

cam = Camera.Camera()
cam.set_defaults()
cam.set_meta(make, model, lens_model)
cam.set_lens_params(args.ccd_width, ccd_height, focal_len_mm)
cam.set_K(fx, fy, cu, cv)
cam.set_image_params(width, height)

file_name = os.path.join(args.config, base_name + '.json')
if os.path.exists(file_name):
    print('Camera config file already exists:', file_name)
    if args.force:
        print('Overwriting ...')
    else:
        print('Aborting ...')
        quit()

print('Saving:', file_name)
cam_node = getNode('/config/camera', True)
props_json.save(file_name, cam_node)
