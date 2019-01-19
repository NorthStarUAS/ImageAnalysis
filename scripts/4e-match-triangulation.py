#!/usr/bin/python3

import numpy as np
import os
import sys

import argparse
import pickle

from props import getNode

sys.path.append('../lib')
import LineSolver
import ProjectMgr
import SRTM

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--area', required=True, help='sub area directory')
parser.add_argument('--method', default='srtm', choices=['srtm', 'triangulate'])
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_area_info(args.area)
#proj.load_features(descriptors=False)
#proj.undistort_keypoints()
area_dir = os.path.join(args.project, args.area)

source = 'matches_grouped'
print("Loading source matches:", source)
matches = pickle.load( open( os.path.join(area_dir, source), 'rb' ) )

K = proj.cam.get_K()
IK = np.linalg.inv(K)

if args.method == 'srtm':
    # lookup ned reference
    ref_node = getNode("/config/ned_reference", True)
    ref = [ ref_node.getFloat('lat_deg'),
            ref_node.getFloat('lon_deg'),
            ref_node.getFloat('alt_m') ]

    # setup SRTM ground interpolator
    sss = SRTM.NEDGround( ref, 3000, 3000, 30 )

    # for each image lookup the SRTM elevation under the camera
    for image in proj.image_list:
        ned, ypr, quat = image.get_camera_pose()
        image.base_elev = sss.interp([ned[0], ned[1]])[0]
        #print(image.name, image.base_elev)
    
    for match in matches:
        sum = np.zeros(3)
        for m in match[1:]:
            image = proj.image_list[m[0]]
            cam2body = image.get_cam2body()
            body2ned = image.get_body2ned()
            ned, ypr, quat = image.get_camera_pose()
            uv_list = [ m[1] ] # just one uv element
            vec_list = proj.projectVectors(IK, body2ned, cam2body, uv_list)
            v = vec_list[0]
            if v[2] > 0.0:
                d_proj = -(ned[2] + image.base_elev)
                factor = d_proj / v[2]
                n_proj = v[0] * factor
                e_proj = v[1] * factor
                p = [ ned[0] + n_proj, ned[1] + e_proj, ned[2] + d_proj ]
                #print('  ', p)
                sum += np.array(p)
            else:
                print('vector projected above horizon.')
        match[0] = (sum/len(match[1:])).tolist()
elif args.method == 'triangulate':
    for match in matches:
        #print(match)
        points = []
        vectors = []
        for m in match[1:]:
            #print(m)
            image = proj.image_list[m[0]]
            cam2body = image.get_cam2body()
            body2ned = image.get_body2ned()
            ned, ypr, quat = image.get_camera_pose()
            uv_list = [ m[1] ] # just one uv element
            vec_list = proj.projectVectors(IK, body2ned, cam2body, uv_list)
            points.append( ned )
            vectors.append( vec_list[0] )
            #print(' ', image.name)
            #print(' ', uv_list)
            #print('  ', vec_list)
        #print('points:', points)
        #print('vectors:', vectors)
        p = LineSolver.ls_lines_intersection(points, vectors, transpose=True).tolist()
        #print('result:',  p, p[0])
        #print(match[0], '>>>', end=" ")
        match[0] = [ p[0][0], p[1][0], p[2][0] ]
        #print(match[0])
    
print("Writing:", source)
pickle.dump(matches, open(os.path.join(area_dir, source), "wb"))
