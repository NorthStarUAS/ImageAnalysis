#!/usr/bin/python3

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

parser = argparse.ArgumentParser(description='Show capture data of images.')
parser.add_argument('project', help='project directory')
parser.add_argument('--config', default='../cameras', help='camera config directory')

args = parser.parse_args()

proj = project.ProjectMgr(args.project)

image_dir = args.project
image_file = None
for file in sorted(os.listdir(image_dir)):
    if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
        image_file = os.path.join(image_dir, file)
        exif = pyexiv2.ImageMetadata(image_file)
        exif.read()
        print(file, exif['Exif.Image.DateTime'].value)
