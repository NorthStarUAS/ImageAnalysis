#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import os.path

sys.path.append('../lib')
import ProjectMgr

# create an empty project

parser = argparse.ArgumentParser(description='Create an empty project.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.save()
