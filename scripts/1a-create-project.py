#!/usr/bin/python3

import argparse

from lib import ProjectMgr

# initialize a new project workspace

parser = argparse.ArgumentParser(description='Create an empty project.')
parser.add_argument('--project', required=True, help='project work directory')
parser.add_argument('--image-dirs', required=True, nargs='+', help='image source directory')

args = parser.parse_args()
print(args)

# create an empty project
proj = ProjectMgr.ProjectMgr(args.project, create=True)
proj.set_image_sources(args.image_dirs)
proj.save()

proj.load_images_info()
