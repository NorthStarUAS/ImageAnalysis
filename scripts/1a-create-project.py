#!/usr/bin/python3

import argparse
import sys
sys.path.append('../lib')
import ProjectMgr

# create an empty project

parser = argparse.ArgumentParser(description='Create an empty project.')
parser.add_argument('--project', required=True, help='project work directory')
parser.add_argument('--images', required=True, help='image source directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project, create=True)
proj.set_images_source(args.images)
proj.save()
