#!/usr/bin/python3

import argparse
import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm

from props import getNode

from lib import Groups
from lib import LineSolver
from lib import ProjectMgr
from lib import SRTM

def match_trig(project_dir, match_trig_options):

    group = match_trig_options[0]
    method = match_trig_options[1]
    proj = ProjectMgr.ProjectMgr(project_dir)
    proj.load_images_info()

    source = 'matches_grouped'
    print("Loading source matches:", source)
    matches = pickle.load( open( os.path.join(proj.analysis_dir, source), 'rb' ) )

    # load the group connections within the image set
    groups = Groups.load(proj.analysis_dir)
    print('Group sizes:', end=" ")
    for group in groups:
        print(len(group), end=" ")
    print()

    if method == 'triangulate':
        K = proj.cam.get_K(optimized=True)
        dist_coeffs = np.array(proj.cam.get_dist_coeffs(optimized=True))
    else:
        K = proj.cam.get_K(optimized=False)
    IK = np.linalg.inv(K)

    do_sanity_check = False

    # assume global K and distcoeff set earlier
    def undistort(uv_orig):
        # convert the point into the proper format for opencv
        uv_raw = np.zeros((1,1,2), dtype=np.float32)
        uv_raw[0][0] = (uv_orig[0], uv_orig[1])
        # do the actual undistort
        uv_new = cv2.undistortPoints(uv_raw, K, dist_coeffs, P=K)
        # print(uv_orig, type(uv_new), uv_new)
        return uv_new[0][0]
        
    if method == 'srtm':
        # lookup ned reference
        ref_node = getNode("/config/ned_reference", True)
        ref = [ ref_node.getFloat('lat_deg'),
                ref_node.getFloat('lon_deg'),
                ref_node.getFloat('alt_m') ]

        # setup SRTM ground interpolator
        sss = SRTM.NEDGround( ref, 3000, 3000, 30 )

        # for each image lookup the SRTM elevation under the camera
        print("Looking up SRTM base elevation for each image location...")
        for image in proj.image_list:
            ned, ypr, quat = image.get_camera_pose()
            image.base_elev = sss.interp([ned[0], ned[1]])[0]
            # print(image.name, image.base_elev)

        print("Estimating initial projection for each feature...")
        bad_count = 0
        bad_indices = []
        for i, match in enumerate(tqdm(matches)):
            sum = np.zeros(3)
            array = []              # fixme: temp/debug
            for m in match[2:]:
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
                    # print('  ', p)
                    sum += np.array(p)
                    array.append(p)
                else:
                    print('vector projected above horizon.')
            match[0] = (sum/len(match[2:])).tolist()
            # print(match[0])
            if do_sanity_check:
                # crude sanity check
                ok = True
                for p in array:
                    dist = np.linalg.norm(np.array(match[0]) - np.array(p))
                    if dist > 100:
                        ok = False
                if not ok:
                    bad_count += 1
                    bad_indices.append(i)
                    print('match:', i, match[0])
                    for p in array:
                        dist = np.linalg.norm(np.array(match[0]) - np.array(p))
                        print(' ', dist, p)
        if do_sanity_check:
            print('bad count:', bad_count)
            print('deleting bad matches...')
            bad_indices.reverse()
            for i in bad_indices:
                del matches[i]
    elif method == 'triangulate':
        for i, match in enumerate(matches):
            if match[1] == group: # used in current group
                # print(match)
                points = []
                vectors = []
                for m in match[2:]:
                    if proj.image_list[m[0]].name in groups[group]:
                        # print(m)
                        image = proj.image_list[m[0]]
                        cam2body = image.get_cam2body()
                        body2ned = image.get_body2ned()
                        ned, ypr, quat = image.get_camera_pose(opt=True)
                        uv_list = [ undistort(m[1]) ] # just one uv element
                        vec_list = proj.projectVectors(IK, body2ned, cam2body, uv_list)
                        points.append( ned )
                        vectors.append( vec_list[0] )
                        # print(' ', image.name)
                        # print(' ', uv_list)
                        # print('  ', vec_list)
                if len(points) >= 2:
                    # print('points:', points)
                    # print('vectors:', vectors)
                    p = LineSolver.ls_lines_intersection(points, vectors, transpose=True).tolist()
                    # print('result:',  p, p[0])
                    print(i, match[0], '>>>', end=" ")
                    match[0] = [ p[0][0], p[1][0], p[2][0] ]
                    if p[2][0] > 0:
                        print("WHOA!")
                    print(match[0])
    
    print("Writing:", source)
    pickle.dump(matches, open(os.path.join(proj.analysis_dir, source), "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keypoint projection.')
    parser.add_argument('--project', required=True, help='project directory')
    parser.add_argument('--group', type=int, default=0, help='group number')
    parser.add_argument('--method', default='srtm', choices=['srtm', 'triangulate'])
    args = parser.parse_args()

    match_trig_options = [args.group, args.method]
    
    match_trig(args.project, match_trig_options)