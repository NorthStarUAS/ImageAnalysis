#!/usr/bin/python3

import argparse

from props import getNode

import sys
sys.path.append('../lib')
import Pose
import ProjectMgr

# for all the images in the project image_dir, detect features using the
# specified method and parameters

parser = argparse.ArgumentParser(description='Set the aircraft poses from flight data.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--meta', help='use the specified image-metadata.txt file (lat,lon,alt,yaw,pitch,roll)')
parser.add_argument('--pix4d', help='use the specified pix4d csv file (lat,lon,alt,roll,pitch,yaw)')
parser.add_argument('--max-angle', type=float, default=25.0, help='max pitch or roll angle for image inclusion')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
print("Loading image info...")
proj.load_images_info()

pose_set = False
if args.meta != None:
    Pose.setAircraftPoses(proj, args.meta, order='ypr', max_angle=args.max_angle)
    pose_set = True
elif args.pix4d != None:
    Pose.setAircraftPoses(proj, args.pix4d, order='rpy', max_angle=args.max_angle)
    pose_set = True

if not pose_set:
    print("Error: no flight data specified or problem with flight data")
    print("No poses computed")
    quit()

# compute the project's NED reference location (based on average of
# aircraft poses)
proj.compute_ned_reference_lla()
ned_node = getNode('/config/ned_reference', True)
print("NED reference location:")
ned_node.pretty_print("  ")

# set the camera poses (fixed offset from aircraft pose) Camera pose
# location is specfied in ned, so do this after computing the ned
# reference point for this project.
Pose.compute_camera_poses(proj)

# save the poses
proj.save_images_info()

# save change to ned reference
proj.save()
    
