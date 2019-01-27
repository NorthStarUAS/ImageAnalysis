#!/usr/bin/python3

# Prep the point and camera data and run it through the scipy least
# squares optimizer.
#
# This optimizer explores using cv2 native functions to do per-image
# reprojection, and then extract out the errors from that.

import os
import time

import cv2
import math
# import matplotlib.pyplot as plt
# from matplotlib import cm
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

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
        self.last_mre = None
        self.graph = None
        self.graph_counter = 0
        #self.optimize_calib = 'global' # global camera optimization
        self.optimize_calib = 'none' # no camera calibration optimization
        self.min_chain_length = 3
        self.with_bounds = False
        self.ncp = 6

    # plot range
    def my_plot_range(self, data, stats=False):
        if stats:
            avg = np.mean(data)
            std = np.std(data)
            min = math.floor((avg-3*std) / 10) * 10
            max = math.ceil((avg+3*std) / 10) * 10
        else:
            min = math.floor(np.amin(data) / 10) * 10
            max = math.ceil(np.amax(data) / 10) * 10
        return min, max
    
    # for lack of a better function name, input rvec, tvec, and return
    # corresponding ypr and ned values
    def rvectvec2yprned(self, rvec, tvec):
        cam2body = np.array( [[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]], dtype=float )
        Rned2cam, jac = cv2.Rodrigues(rvec)
        Rned2body = cam2body.dot(Rned2cam)
        Rbody2ned = np.matrix(Rned2body).T
        ypr = transformations.euler_from_matrix(Rbody2ned, 'rzyx')
        pos = -np.matrix(Rned2cam).T * np.matrix(tvec).T
        ned = np.squeeze(np.asarray(pos.T[0]))
        return ypr, ned

    # compute the sparsity matrix (dependency relationships between
    # observations and parameters the optimizer can manipulate.)
    # Because of the extreme number of parameters and observations, a
    # sparse matrix is required to run in finite time for all but the
    # smallest data sets.
    def bundle_adjustment_sparsity(self, n_cameras, n_points,
                                   camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * self.ncp + n_points * 3
        if self.optimize_calib == 'global':
            n += 8  # three K params (fx == fy) + five distortion params
        A = lil_matrix((m, n), dtype=int)
        print('sparsity matrix is %d x %d' % (m, n))

        i = np.arange(camera_indices.size)
        for s in range(self.ncp):
            A[2 * i, camera_indices * self.ncp + s] = 1
            A[2 * i + 1, camera_indices * self.ncp + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * self.ncp + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * self.ncp + point_indices * 3 + s] = 1

        if self.optimize_calib == 'global':
            for s in range(0,3): # K
                A[2 * i, n_cameras * self.ncp + n_points * 3 + s] = 1
                A[2 * i + 1, n_cameras * self.ncp + n_points * 3 + s] = 1
            for s in range(3,8): # dist coeffs
                A[2 * i, n_cameras * self.ncp + n_points * 3 + s] = 1
                A[2 * i + 1, n_cameras * self.ncp + n_points * 3 + s] = 1

        print('A-matrix non-zero elements:', A.nnz)
        return A

    # compute an array of residuals (one for each observation)
    # params contains camera parameters, 3-D coordinates, and
    # camera calibration parameters.
    def fun(self, params, n_cameras, n_points, by_camera_point_indices, by_camera_points_2d):
        error = None
        # extract the parameters
        camera_params = params[:n_cameras * self.ncp].reshape((n_cameras, self.ncp))
        
        points_3d = params[n_cameras * self.ncp:n_cameras * self.ncp + n_points * 3].reshape((n_points, 3))
        
        if self.optimize_calib == 'global':
            # assemble K and distCoeffs from the optimizer param list
            camera_calib = params[n_cameras * self.ncp + n_points * 3:]
            K = np.identity(3)
            K[0,0] = camera_calib[0]
            K[1,1] = camera_calib[0]
            K[0,2] = camera_calib[1]
            K[1,2] = camera_calib[2]
            distCoeffs = camera_calib[3:]
        else:
            # use a fixed K and distCoeffs
            K = self.K
            distCoeffs = self.distCoeffs

        #fixme: global calibration optimization, but force distortion
        #paramters to stay fixed to those originally given
        #distCoeffs = self.distCoeffs

        sum = 0
        # cams_3d = np.zeros((n_cameras, 3)) # for plotting
        for i, cam in enumerate(camera_params):
            rvec = cam[:3]
            tvec = cam[3:6]
            # ypr, ned = self.rvectvec2yprned(rvec, tvec)
            # cams_3d[i] = ned # for plotting
            if len(by_camera_point_indices[i]) == 0:
                continue
            proj_points, jac = cv2.projectPoints(points_3d[by_camera_point_indices[i]], rvec, tvec, K, distCoeffs)
            sum += len(proj_points.ravel())
            if error is None:
                error = (by_camera_points_2d[i] - proj_points).ravel()
            else:
                error = np.append(error, (by_camera_points_2d[i] - proj_points).ravel())

        # provide some runtime feedback for the operator
        mre = np.mean(np.abs(error))
        if self.last_mre is None or 1.0 - mre/self.last_mre > 0.001:
            # mre has improved by more than 0.1%
            self.last_mre = mre
            print('mre: %.3f std: %.3f max: %.2f' % (mre, np.std(error), np.amax(np.abs(error))) )
            if self.optimize_calib == 'global':
                print("K:\n", K)
                print("distCoeffs: %.3f %.3f %.3f %.3f %.3f" %
                      (distCoeffs[0], distCoeffs[1], distCoeffs[2],
                       distCoeffs[3], distCoeffs[4]))
            # if not self.graph is None:
            #     points = points_3d
            #     #points = cams_3d
            #     self.graph.set_offsets(points[:,[1,0]])
            #     self.graph.set_array(-points[:,2])
            #     xmin, xmax = self.my_plot_range(points[:,1])
            #     ymin, ymax = self.my_plot_range(points[:,0])
            #     plt.xlim(xmin, xmax)
            #     plt.ylim(ymin, ymax)
            #     cmin, cmax = self.my_plot_range(-points[:,2], stats=True)
            #     plt.clim(cmin, cmax)
            #     plt.gcf().set_size_inches(16,9,forward=True)
            #     plt.draw()
            #     if False:
            #         # animate the optimizer progress as a movie
            #         # ex: ffmpeg -f image2 -r 2 -s 1280x720 -i optimizer-%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p optimizer.mp4
            #         plt_name = 'optimizer-%03d.png' % self.graph_counter
            #         out_file = os.path.join(self.root, plt_name)
            #         plt.savefig(out_file, dpi=80)
            #         self.graph_counter += 1
            #     plt.pause(0.01)
        return error

    # assemble the structures and remapping indices required for
    # optimizing a group of images/features
    def setup(self, proj, group, matches_list, optimized=False):
        print('Setting up optimizer data structures...')
        # if placed_images == None:
        #     placed_images = []
        #     # if no placed images specified, mark them all as placed
        #     for i in range(len(proj.image_list)):
        #         placed_images.append(i)
        placed_images = set()
        for name in group:
            i = proj.findIndexByName(name)
            placed_images.add(i)            
        print('Number of placed images:', len(placed_images))
        
        # construct the camera index remapping
        self.camera_map_fwd = {}
        self.camera_map_rev = {}
        for i, index in enumerate(placed_images):
            self.camera_map_fwd[i] = index
            self.camera_map_rev[index] = i
        #print(self.camera_map_fwd)
        #print(self.camera_map_rev)
        
        # initialize the feature index remapping
        self.feat_map_fwd = {}
        self.feat_map_rev = {}

        self.K = proj.cam.get_K(optimized)
        self.distCoeffs = np.array(proj.cam.get_dist_coeffs(optimized))
        
        # assemble the initial camera estimates
        self.n_cameras = len(placed_images)
        self.camera_params = np.empty(self.n_cameras * self.ncp)
        for cam_idx, global_index in enumerate(placed_images):
            image = proj.image_list[global_index]
            rvec, tvec = image.get_proj(optimized)
            self.camera_params[cam_idx*self.ncp:cam_idx*self.ncp+self.ncp] = np.append(rvec, tvec)

        # count number of 3d points and observations
        self.n_points = 0
        n_observations = 0
        for i, match in enumerate(matches_list):
            # count the number of referenced observations
            if match[1]:        # in use
                count = 0
                for m in match[2:]:
                    if m[0] in placed_images:
                        count += 1
                if count >= self.min_chain_length:
                    n_observations += count
                    self.n_points += 1

        # assemble 3d point estimates and build indexing maps
        self.points_3d = np.empty(self.n_points * 3)
        point_idx = 0
        feat_used = 0
        for i, match in enumerate(matches_list):
            if match[1]:        # in use
                count = 0
                for m in match[2:]:
                    if m[0] in placed_images:
                        count += 1
                if count >= self.min_chain_length:
                    self.feat_map_fwd[i] = feat_used
                    self.feat_map_rev[feat_used] = i
                    feat_used += 1
                    ned = np.array(match[0])
                    if np.any(np.isnan(ned)):
                        print(i, ned)
                    self.points_3d[point_idx] = ned[0]
                    self.points_3d[point_idx+1] = ned[1]
                    self.points_3d[point_idx+2] = ned[2]
                    point_idx += 3
                
        # assemble observations (image index, feature index, u, v)
        self.by_camera_point_indices = [ [] for i in range(self.n_cameras) ]
        self.by_camera_points_2d = [ [] for i in range(self.n_cameras) ]
        #print('by_camera:', by_camera)
        #points_2d = np.empty((n_observations, 2))
        #obs_idx = 0
        for i, match in enumerate(matches_list):
            if match[1]:        # in use
                count = 0
                for m in match[2:]:
                    if m[0] in placed_images:
                        count += 1
                if count >= self.min_chain_length:
                    for m in match[2:]:
                        if m[0] in placed_images:
                            cam_index = self.camera_map_rev[m[0]]
                            feat_index = self.feat_map_fwd[i]
                            kp = m[1] # orig/distorted
                            #kp = proj.image_list[m[0]].uv_list[m[1]] # undistorted
                            self.by_camera_point_indices[cam_index].append(feat_index)
                            self.by_camera_points_2d[cam_index].append(kp)

        # convert to numpy native structures
        for i in range(self.n_cameras):
            size = len(self.by_camera_point_indices[i])
            self.by_camera_point_indices[i] = np.array(self.by_camera_point_indices[i])
            self.by_camera_points_2d[i] = np.asarray([self.by_camera_points_2d[i]]).reshape(size, 1, 2)

        # generate the camera and point indices (for mapping the
        # sparse jacobian entries which define which observations
        # depend on which parameters.)
        self.camera_indices = np.empty(n_observations, dtype=int)
        self.point_indices = np.empty(n_observations, dtype=int)
        obs_idx = 0
        for i in range(self.n_cameras):
            for j in range(len(self.by_camera_point_indices[i])):
                self.camera_indices[obs_idx] = i
                self.point_indices[obs_idx] = self.by_camera_point_indices[i][j]
                obs_idx += 1
        print("num observations:", obs_idx)

    # assemble the structures and remapping indices required for
    # optimizing a group of images/features, call the optimizer, and
    # save the result.
    def run(self):
        if self.optimize_calib == 'global':
            x0 = np.hstack((self.camera_params.ravel(), self.points_3d.ravel(),
                            self.K[0,0], self.K[0,2], self.K[1,2],
                            self.distCoeffs))
        else:
            x0 = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))
        f0 = self.fun(x0, self.n_cameras, self.n_points,
                      self.by_camera_point_indices, self.by_camera_points_2d)
        mre_start = np.mean(np.abs(f0))

        A = self.bundle_adjustment_sparsity(self.n_cameras, self.n_points,
                                            self.camera_indices,
                                            self.point_indices)

        if self.with_bounds:
            # quick test of bounds ... allow camera parameters to go free,
            # but limit 3d points =to +/- 100m of initial guess
            lower = []
            upper = []
            tol = 100.0
            for i in range(self.n_cameras):
                # unlimit the camera params
                for j in range(self.ncp):
                    if j == 6:
                        # bound focal length
                        lower.append(self.K[0,0]*0.95)
                        upper.append(self.K[0,0]*1.05)
                    else:
                        lower.append( -np.inf )
                        upper.append( np.inf )
            for i in range(self.n_points * 3):
                #lower.append( points_3d[i] - tol )
                #upper.append( points_3d[i] + tol )
                # what if we let point locations float without constraint?
                lower.append( -np.inf )
                upper.append( np.inf )
            if self.optimize_calib == 'global':
                tol = 0.01
                # bound focal length
                lower.append(self.K[0,0]*(1-tol))
                upper.append(self.K[0,0]*(1+tol))
                cu = self.K[0,2]
                cv = self.K[1,2]
                lower.append(cu*(1-tol))
                upper.append(cu*(1+tol))
                lower.append(cv*(1-tol))
                upper.append(cv*(1+tol))
                # unlimit distortion params
                for i in range(5):
                    lower.append( -np.inf )
                    upper.append( np.inf )
            bounds = [lower, upper]
        else:
            bounds = (-np.inf, np.inf)
        # plt.figure(figsize=(16,9))
        # plt.ion()
        # mypts = self.points_3d.reshape((self.n_points, 3))
        # self.graph = plt.scatter(mypts[:,1], mypts[:,0], 100, -mypts[:,2], cmap=cm.jet)
        # plt.colorbar()
        # plt.draw()
        # plt.pause(0.01)
        
        t0 = time.time()
        ftol = 0.01             # default is 1e-8
        ftol = 1e-4
        # bounds=bounds,
        res = least_squares(self.fun, x0,
                            jac_sparsity=A,
                            verbose=2,
                            method='trf',
                            loss='linear',
                            ftol=ftol,
                            x_scale='jac',
                            args=(self.n_cameras, self.n_points,
                                  self.by_camera_point_indices,
                                  self.by_camera_points_2d))
        t1 = time.time()
        print("Optimization took {0:.0f} seconds".format(t1 - t0))
        # print(res['x'])
        print(res)
        
        self.camera_params = res.x[:self.n_cameras * self.ncp].reshape((self.n_cameras, self.ncp))
        self.points_3d = res.x[self.n_cameras * self.ncp:self.n_cameras * self.ncp + self.n_points * 3].reshape((self.n_points, 3))
        if self.optimize_calib == 'global':
            camera_calib = res.x[self.n_cameras * self.ncp + self.n_points * 3:]
            fx = camera_calib[0]
            fy = camera_calib[0]
            cu = camera_calib[1]
            cv = camera_calib[2]
            distCoeffs_opt = camera_calib[3:]
        else:
            fx = self.K[0,0]
            fy = self.K[1,1]
            cu = self.K[0,2]
            cv = self.K[1,2]
            distCoeffs_opt = self.distCoeffs
        
        mre_final = np.mean(np.abs(res.fun))
        iterations = res.njev
        time_sec = t1 - t0

        print("Starting mean reprojection error: %.2f" % mre_start)
        print("Final mean reprojection error: %.2f" % mre_final)
        print("Iterations:", iterations)
        print("Elapsed time = %.1f sec" % time_sec)
        if self.optimize_calib == 'global':
            print("Final camera calib:\n", camera_calib)

        # final plot
        # plt.plot(res.fun)
        # plt.ioff()
        # plt.show()

        return ( self.camera_params, self.points_3d,
                 self.camera_map_fwd, self.feat_map_rev,
                 fx, fy, cu, cv, distCoeffs_opt )
