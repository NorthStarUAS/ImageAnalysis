#!/usr/bin/python3

import argparse
import cv2
import json
import numpy as np
import os

import navpy
from props import getNode       # aura-props

from lib import camera
from lib import project

parser = argparse.ArgumentParser(description='Generate cropped preview images from annotation points.')
parser.add_argument('project', help='project directory')
args = parser.parse_args()

id_prefix = "Marker "

file = os.path.join(args.project, 'annotations.json')
if os.path.exists(file):
    print('Loading annotations:', file)
    f = open(file, 'r')
    root = json.load(f)
    if type(root) is dict:
        if 'id_prefix' in root:
            id_prefix = root['id_prefix']
        if 'markers' in root:
            lla_list = root['markers']
    f.close()
else:
    print('No annotations file found.')
    quit()
    
proj = project.ProjectMgr(args.project)
proj.load_images_info()

# lookup ned reference
ref_node = getNode("/config/ned_reference", True)
ned_ref = [ ref_node.getFloat('lat_deg'),
            ref_node.getFloat('lon_deg'),
            ref_node.getFloat('alt_m') ]
print(ned_ref)

K = camera.get_K()
dist_coeffs = camera.get_dist_coeffs()

for m in lla_list:
    feat_ned = navpy.lla2ned(m['lat_deg'], m['lon_deg'], m['alt_m'],
                             ned_ref[0], ned_ref[1], ned_ref[2])
    print(m, feat_ned)

    # quick hack to find closest image
    best_dist = None
    best_image = None
    for i in proj.image_list:
        image_ned, ypr, quat = i.get_camera_pose(opt=True)
        dist = np.linalg.norm( np.array(feat_ned) - np.array(image_ned) )
        if best_dist == None or dist < best_dist:
            best_dist = dist
            best_image = i
            # print("  best_dist:", best_dist)
    if best_image != None:
        # project the feature ned coordinate into the uv space of the
        # closest image
        print(" ", best_image.name, best_dist)
        rvec, tvec = best_image.get_proj()
        reproj_points, jac = cv2.projectPoints(np.array([feat_ned]),
                                               rvec, tvec,
                                               K, dist_coeffs)
        reproj_list = reproj_points.reshape(-1,2).tolist()
        kp = reproj_list[0]
        print(kp)
        
        rgb = best_image.load_rgb()
        h, w = rgb.shape[:2]
        size = 512
        cx = int(round(kp[0]))
        cy = int(round(kp[1]))
        if cx < size:
            xshift = size - cx
            cx = size
        elif cx > (w - size):
            xshift = (w - size) - cx
            cx = w - size
        else:
            xshift = 0
        if cy < size:
            yshift = size - cy
            cy = size
        elif cy > (h - size):
            yshift = (h - size) - cy
            cy = h - size
        else:
            yshift = 0
        print('size:', w, h, 'shift:', xshift, yshift)
        crop = rgb[cy-size:cy+size, cx-size:cx+size]
        label = "%s%03d" % (id_prefix, m['id'])
        cv2.imshow(best_image.name + ' ' + label, crop)
        cv2.waitKey()
