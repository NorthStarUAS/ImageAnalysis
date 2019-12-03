#!/usr/bin/python3

import argparse
import os

from lib import project

# initialize a new project workspace

parser = argparse.ArgumentParser(description='Create an empty project.')
parser.add_argument('--project', required=True, help='Directory with a set of aerial images.')

args = parser.parse_args()

# test if images directory exists
if not os.path.isdir(args.project):
    print("Images directory doesn't exist:", args.project)
    quit()

# create an empty project
proj = project.ProjectMgr(args.project, create=True)

# and save what we have so far ...
proj.save()

