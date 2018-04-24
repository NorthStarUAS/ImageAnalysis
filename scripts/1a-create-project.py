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

# test if images.json exists
#if os.path.isfile( os.path.join(args.project, 'images.json') ):
#    print('Notice: found an existing images.json file, so pre-loading it.')
#    proj.load_images_info()

proj.load_images_info()
