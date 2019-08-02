#!/usr/bin/python3

import os, sys, argparse

from lib import ProjectMgr

# initialize a new project workspace
def new_project(project_dir):
    # test if images directory exists
    if not os.path.isdir(project_dir):
        print("Images directory doesn't exist:", args.project)
        quit()

    # create an empty project
    proj = ProjectMgr.ProjectMgr(project_dir, create=True)

    # and save what we have so far ...
    proj.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an empty project.')
    parser.add_argument('--project', required=True, help='Directory with a set of aerial images.')

    args = parser.parse_args()
    new_project(args.project)