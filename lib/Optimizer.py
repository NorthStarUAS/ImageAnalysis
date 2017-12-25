#!/usr/bin/python

# Prep the point and camera data and run it through the scipy least
# squares optimizer.

import cv2
import numpy as np
import os
# import re

import sys
sys.path.append('../lib')
import transformations


# This is a python class that optimizes the estimate camera and 3d
# point fits by minimizing the mean reprojection error.

class Optimizer():
    def __init__(self, root):
        self.root = root
        self.camera_map_fwd = {}
        self.camera_map_rev = {}
        self.feat_map_fwd = {}
        self.feat_map_rev = {}

    # write the camera (motion) parameters, feature (structure)
    # parameters, and calibration (K) to files in the project
    # directory.
    def prepair_data(self, image_list, placed_images, matches_list, K,
                     use_sba=False):
        if placed_images == None:
            placed_images = set()
            # if no placed images specified, mark them all as placed
            for i in range(len(image_list)):
                placed_images.add(i)
                
        # construct the camera index remapping
        self.camera_map_fwd = {}
        self.camera_map_rev = {}
        for i, index in enumerate(placed_images):
            self.camera_map_fwd[i] = index
            self.camera_map_rev[index] = i
        print(self.camera_map_fwd)
        print(self.camera_map_rev)
        
        # initialize the feature index remapping
        self.feat_map_fwd = {}
        self.feat_map_rev = {}
        
        # iterate through the matches dictionary to produce a list of matches
        feat_used = 0
        f = open( self.root + '/sba-points.txt', 'w' )
        for i, match in enumerate(matches_list):
            ned = np.array(match[0])
            #print type(ned), ned.size, ned
            count = 0
            for p in match[1:]:
                if p[0] in placed_images:
                    count += 1
            if ned.size == 3 and count >= 2:
                self.feat_map_fwd[i] = feat_used
                self.feat_map_rev[feat_used] = i
                feat_used += 1
                s = "%.4f %.4f %.4f  " % (ned[0], ned[1], ned[2])
                f.write(s)
                s = "%d  " % (count)
                f.write(s)
                for p in match[1:]:
                    if p[0] in placed_images:
                        local_index = self.camera_map_rev[p[0]]
                        # kp = image_list[p[0]].kp_list[p[1]].pt # distorted
                        kp = image_list[p[0]].uv_list[p[1]]      # undistorted
                        s = "%d %.2f %.2f " % (local_index, kp[0], kp[1])
                        f.write(s)
                f.write('\n')
        f.close()

        # count number of 3d points and observations
        n_points = 0
        n_observations = 0
        for i, match in enumerate(matches_list):
            # count the number of referenced observations
            used = False
            for p in match[1:]:
                if p[0] in placed_images:
                    n_observations += 1
                    used = True
            if used:
                n_points += 1

        # assemble 3d point estimates and build indexing maps
        points_3d = np.empty(n_points * 3)
        point_idx = 0
        feat_used = 0
        for i, match in enumerate(matches_list):
            ned = np.array(match[0])
            used = False
            for p in match[1:]:
                if p[0] in placed_images:
                    used = True
            if used:
                self.feat_map_fwd[i] = feat_used
                self.feat_map_rev[feat_used] = i
                feat_used += 1
                points_3d[point_idx] = ned[0]
                points_3d[point_idx+1] = ned[1]
                points_3d[point_idx+2] = ned[2]
                point_idx += 3
                
        # assemble observations (image index, feature index, u, v)
        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))
        obs_idx = 0
        for i, match in enumerate(matches_list):
            used = False
            for p in match[1:]:
                if p[0] in placed_images:
                    cam_index = self.camera_map_rev[p[0]]
                    feat_index = self.feat_map_fwd[i]
                    uv = image_list[p[0]].uv_list[p[1]]
                    camera_indices[obs_idx] = cam_index
                    point_indices[obs_idx] = feat_index
                    points_2d[obs_idx] = uv
                    obs_idx += 1

        # assemble the initial camera estimates
        n_cameras = len(placed_images)
        camera_params = np.empty(n_cameras * 6)
        cam_idx = 0
        for index in placed_images:
            image = image_list[index]
            rvec, tvec = image.get_proj_sba()
            camera_params[cam_idx*6:cam_idx*6+6] = np.append(rvec, tvec)
            cam_idx += 1

        return camera_params, points_3d, camera_indices, point_indices, points_2d

    def run(self, mode=''):
        command = []

        #result = subprocess.check_output( command )
        # bufsize=1 is line buffered
        #process = subprocess.Popen( command, stdout=subprocess.PIPE)

        state = ''
        mre_start = 0.0         # mre = mean reprojection error
        mre_final = 0.0         # mre = mean reprojection error
        iterations = 0
        time_msec = 0.0
        cameras = []
        features = []
        error_images = set()

        result = process.stdout.readline()
        print(result)
        while result:
            for line in result.split('\n'):
                #print "line: ", line
                if re.search('mean reprojection error', line):
                    print(line)
                    value = float(re.sub('mean reprojection error', '', line))
                    if mre_start == 0.0:
                        mre_start = value
                    else:
                        mre_final = value
                elif re.search('damping term', line):
                    print(line )
                elif re.search('iterations=', line):
                    print(line)
                    iterations = int(re.sub('iterations=', '', line))
                elif re.search('Elapsed time:', line):
                    print(line)
                    tokens = line.split()
                    time_msec = float(tokens[4])
                elif re.search('Motion parameters:', line):
                    state = 'motion'
                elif re.search('Structure parameters:', line):
                    state = 'structure'
                elif re.search('the estimated projection of point', line):
                    print(line)
                    tokens = line.split()
                    cam_index = int(tokens[12])
                    image_index = self.camera_map_fwd[cam_index]
                    print('sba cam: {} image index: {}'.format(cam_index, image_index))
                    error_images.add(image_index)
                else:
                    tokens = line.split()
                    if state == 'motion' and len(tokens) > 0:
                        # print "camera:", np.array(tokens, dtype=float)
                        cameras.append( np.array(tokens, dtype=float) )
                    elif state == 'structure' and len(tokens) == 3:
                        # print "feature:", np.array(tokens, dtype=float)
                        features.append( np.array(tokens, dtype=float) )
                    elif len(line):
                        print(line)
            # read next line
            result = process.stdout.readline()
            
        print("Starting mean reprojection error: {}".format(mre_start))
        print("Final mean reprojection error: {}".format(mre_final))
        print("Iterations: {}".format(iterations))
        print("Elapsed time = {} sec ({} msec)".format(time_msec/1000,
                                                       time_msec))
        return cameras, features, self.camera_map_fwd, self.feat_map_rev, error_images
