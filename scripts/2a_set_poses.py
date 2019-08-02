#!/usr/bin/python3

import argparse
import os

from props import getNode

from lib import Pose
from lib import ProjectMgr

# for all the images in the project image_dir, detect features using the
# specified method and parameters

def set_pose(project_dir, max_angle = 25.0 ):
    proj = ProjectMgr.ProjectMgr(project_dir)
    print("Loading image info...")
    proj.load_images_info()

    # simplifying assumption
    image_dir = project_dir
        
    pix4d_file = os.path.join(image_dir, 'pix4d.csv')
    meta_file = os.path.join(image_dir, 'image-metadata.txt')
    if os.path.exists(pix4d_file):
        Pose.setAircraftPoses(proj, pix4d_file, order='rpy',
                            max_angle=max_angle)
    elif os.path.exists(meta_file):
        Pose.setAircraftPoses(proj, meta_file, order='ypr',
                            max_angle=max_angle)
    else:
        print("Error: no pose file found in image directory:", image_dir)
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set the aircraft poses from flight data.')
    parser.add_argument('--project', required=True, help='project directory')
    #parser.add_argument('--meta', help='use the specified image-metadata.txt file (lat,lon,alt,yaw,pitch,roll)')
    #parser.add_argument('--pix4d', help='use the specified pix4d csv file (lat,lon,alt,roll,pitch,yaw)')
    parser.add_argument('--max-angle', type=float, default=25.0, help='max pitch or roll angle for image inclusion')

    args = parser.parse_args()