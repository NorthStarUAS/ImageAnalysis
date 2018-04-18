#!/usr/bin/python3

import argparse
import fnmatch
import os.path
import sys

sys.path.append('../lib')
import ProjectMgr
import Image

# initialize a new project workspace

parser = argparse.ArgumentParser(description='Create an empty project.')
parser.add_argument('--project', required=True, help='project work directory')
parser.add_argument('--images', required=True, help='image source directory')

args = parser.parse_args()

# create an empty project
proj = ProjectMgr.ProjectMgr(args.project, create=True)
proj.set_images_source(args.images)
proj.save()

# create the initial images list
meta_dir = os.path.join(args.project, 'Images')
proj.image_list = []
for name in os.listdir(args.images):
    if fnmatch.fnmatch(name, '*.jpg') or fnmatch.fnmatch(name, '*.JPG'):
        image = Image.Image(args.images, meta_dir, name)
proj.save_images_info()
