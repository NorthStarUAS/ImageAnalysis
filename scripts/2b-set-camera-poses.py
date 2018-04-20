#!/usr/bin/python3

import argparse

import sys
sys.path.append('../lib')
import Pose
import ProjectMgr

# for all the images in the project image_dir, compute the camera poses from
# the aircraft pose (and camera mounting transform)

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()

Pose.compute_camera_poses(proj)

proj.save_images_info()
    
