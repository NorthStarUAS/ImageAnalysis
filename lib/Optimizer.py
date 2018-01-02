#!/usr/bin/python3

# Prep the point and camera data and run it through the scipy least
# squares optimizer.
#
# This version does not try to solve for any camera parameters, just
# the camera pose and 3d feature locations.

import cv2
import numpy as np
import os
# import re

import sys
sys.path.append('../lib')
import transformations

def project(points, cam_M):
    """Convert 3-D points to 2-D by projecting onto images."""
    #print('points: {}'.format(points.shape))
    #print('ones: {}'.format( np.ones((points.shape[0], 1)) ))
    nedh = np.hstack((points, np.ones((points.shape[0], 1))))
    #print('nedh: {}'.format(nedh))
    #uvh = np.einsum("...ij,...i", cam_M, nedh)
    #uvh = cam_M.dot( nedh )
    uvh = np.zeros( (points.shape[0], 3) )
    for i in range(points.shape[0]):
        uvh[i] = cam_M[i].dot(nedh[i])
    #print('uvh: {}'.format(uvh[:,2:3].shape))
    uvh = uvh / uvh[:,2:3]
    #print('uvh: {}'.format(uvh))
    uv = uvh[:, 0:2]
    #print('uv: {}'.format(uv))
    return uv

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals.
    
    `params` contains camera parameters, 3-D coordinates, and camera calibration parameters.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    cam_M = np.zeros((camera_params.shape[0], 3, 4))
    for i, cam in enumerate(camera_params):
        R, jac = cv2.Rodrigues(cam[:3])
        PROJ = np.concatenate((R, cam[3:6].reshape(3,1)), axis=1)
        M = K.dot( PROJ )
        cam_M[i] = M
    # print('cam_M: {}'.format(cam_M))
    points_3d = params[n_cameras * 6:n_cameras * 6 + n_points * 3].reshape((n_points, 3))
    #print("calib:")
    tmp = cam_M[camera_indices]
    #print('tmp.shape {}'.format(tmp.shape))
    points_proj = project(points_3d[point_indices],
                          cam_M[camera_indices])
    # mre
    error = (points_proj - points_2d).ravel()
    mre = np.mean(np.abs(error))
    print("mre = {}".format(mre))
    return (points_proj - points_2d).ravel()

from scipy.sparse import lil_matrix

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    print('sparcity matrix is {} x {}'.format(m, n))

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    #for s in range(3):
    #    A[2 * i, n_cameras * 6 + n_points * 3 + s] = 1
    #    A[2 * i + 1, n_cameras * 6 + n_points * 3 + s] = 1

    print('A non-zero elements = {}'.format(A.nnz))
    
    return A

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
    def run(self, image_list, placed_images, matches_list, K, use_sba=False):
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
            rvec, tvec = image.get_proj()
            camera_params[cam_idx*6:cam_idx*6+6] = np.append(rvec, tvec)
            cam_idx += 1

        #return camera_params, points_3d, camera_indices, point_indices, points_2d

        # def run(self, mode=''):
        import matplotlib.pyplot as plt
        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
        print('x0:')
        print(x0.shape)
        print(x0)
        f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d, K)
        plt.plot(f0)
        mre_start = np.mean(np.abs(f0))

        A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

        import time
        from scipy.optimize import least_squares

        t0 = time.time()
        res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac',
                            ftol=1e-3, method='trf',
                            args=(n_cameras, n_points, camera_indices,
                                  point_indices, points_2d, K))
        t1 = time.time()
        print("Optimization took {0:.0f} seconds".format(t1 - t0))
        # print(res['x'])
        print(res)
        plt.plot(res.fun)
        plt.show()
        
        camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = res.x[n_cameras * 6:n_cameras * 6 + n_points * 3].reshape((n_points, 3))

        # command = []
        #result = subprocess.check_output( command )
        # bufsize=1 is line buffered
        #process = subprocess.Popen( command, stdout=subprocess.PIPE)

        state = ''
        mre_final = np.mean(np.abs(res.fun))
        iterations = res.njev
        time_sec = t1 - t0

        print("Starting mean reprojection error: {}".format(mre_start))
        print("Final mean reprojection error: {}".format(mre_final))
        print("Iterations: {}".format(iterations))
        print("Elapsed time = {} sec".format(time_sec))
        return camera_params, points_3d, self.camera_map_fwd, self.feat_map_rev
