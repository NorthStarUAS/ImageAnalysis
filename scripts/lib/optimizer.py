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

from .logger import log, qlog
from . import transformations

d2r = math.pi / 180.0
r2d = 180.0 / math.pi

# return a 3d affine tranformation between current camera locations
# and original camera locations.
def get_recenter_affine(src_list, dst_list):
    log('get_recenter_affine():')
    src = [[], [], [], []]      # current camera locations
    dst = [[], [], [], []]      # original camera locations
    for i in range(len(src_list)):
        src_ned = src_list[i]
        src[0].append(src_ned[0])
        src[1].append(src_ned[1])
        src[2].append(src_ned[2])
        src[3].append(1.0)
        dst_ned = dst_list[i]
        dst[0].append(dst_ned[0])
        dst[1].append(dst_ned[1])
        dst[2].append(dst_ned[2])
        dst[3].append(1.0)
        # print("{} <-- {}".format(dst_ned, src_ned))
    A = transformations.superimposition_matrix(src, dst, scale=True)
    log("A:\n", A)
    return A

# transform a point list given an affine transform matrix
def transform_points( A, pts_list ):
    src = [[], [], [], []]
    for p in pts_list:
        src[0].append(p[0])
        src[1].append(p[1])
        src[2].append(p[2])
        src[3].append(1.0)
    dst = A.dot( np.array(src) )
    result = []
    for i in range(len(pts_list)):
        result.append( [ float(dst[0][i]),
                         float(dst[1][i]),
                         float(dst[2][i]) ] )
    return result

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
        #self.graph_counter = 0
        #self.optimize_calib = 'global' # global camera optimization
        self.optimize_calib = 'none' # no camera calibration optimization
        #self.ftol = 1e-2              # stop condition - extra coarse
        #self.ftol = 1e-3              # stop condition - quicker
        self.ftol = 1e-4              # stop condition - better
        self.min_chain_len = 2        # use whatever matches are defind upstream
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
        log('sparsity matrix is %d x %d' % (m, n))

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

        log('A-matrix non-zero elements:', A.nnz)
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
        by_cam = []             # for debugging data set problems
        for i, cam in enumerate(camera_params):
            rvec = cam[:3]
            tvec = cam[3:6]
            # ypr, ned = self.rvectvec2yprned(rvec, tvec)
            # cams_3d[i] = ned # for plotting
            if len(by_camera_point_indices[i]) == 0:
                continue
            proj_points, jac = cv2.projectPoints(points_3d[by_camera_point_indices[i]], rvec, tvec, K, distCoeffs)
            sum += len(proj_points.ravel())
            cam_error = (by_camera_points_2d[i] - proj_points).ravel()
            by_cam.append( [np.mean(np.abs(cam_error)),
                            np.amax(np.abs(cam_error)),
                            self.camera_map_fwd[i] ] )
            if error is None:
                error = cam_error
            else:
                error = np.append(error, cam_error)

        mre = np.mean(np.abs(error))
        std = np.std(error)

        # debug
        count_std = 0
        count_bad = 0
        for e in error.tolist():
            if e > mre + 3 * std:
                count_std += 1
            if e > 10000:
                count_bad += 1
        # print( 'std: %.2f %d/%d > 3*std (max: %.2f)' % (std, count_std, error.shape[0], np.amax(error)) )
        # by_cam = sorted(by_cam, key=lambda fields: fields[0], reverse=True)
        # for line in by_cam:
        #     if line[0] > mre + 2*std:
        #         print("  %s -- mean: %.3f max: %.3f" % (line[2], line[0], line[1]))
        
        # provide some runtime feedback for the operator
        if self.last_mre is None or 1.0 - mre/self.last_mre > 0.001:
            # mre has improved by more than 0.1%
            self.last_mre = mre
            log('mre: %.3f std: %.3f max: %.2f' % (mre, np.std(error), np.amax(np.abs(error))) )
            if self.optimize_calib == 'global':
                log("K:\n", K)
                log("distCoeffs: %.3f %.3f %.3f %.3f %.3f" %
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
    def setup(self, proj, groups, group_index, matches_list, optimized=False,
              cam_calib=False):
        log('Setting up optimizer data structures...')
        if cam_calib:
            self.optimize_calib = 'global' # global camera optimization
        else:
            self.optimize_calib = 'none' # no camera calibration optimization

        # if placed_images == None:
        #     placed_images = []
        #     # if no placed images specified, mark them all as placed
        #     for i in range(len(proj.image_list)):
        #         placed_images.append(i)
        placed_images = set()
        for name in groups[group_index]:
            i = proj.findIndexByName(name)
            placed_images.add(i)            
        log('Number of placed images:', len(placed_images))
        
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
            if match[1] == group_index: # used by the current group
                count = 0
                for m in match[2:]:
                    if m[0] in placed_images:
                        count += 1
                if count >= self.min_chain_len:
                    n_observations += count
                    self.n_points += 1

        # assemble 3d point estimates and build indexing maps
        self.points_3d = np.empty(self.n_points * 3)
        point_idx = 0
        feat_used = 0
        for i, match in enumerate(matches_list):
            if match[1] == group_index: # used by the current group
                count = 0
                for m in match[2:]:
                    if m[0] in placed_images:
                        count += 1
                if count >= self.min_chain_len:
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
            if match[1] == group_index: # used by the current group
                count = 0
                for m in match[2:]:
                    if m[0] in placed_images:
                        count += 1
                if count >= self.min_chain_len:
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
        log("num observations:", obs_idx)

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
                    if j == 5:
                        # bound the altitude of camera (pretend we
                        # trust dji to +/- 1m)
                        lower.append( self.camera_params[i*self.ncp + j] - 1 )
                        upper.append( self.camera_params[i*self.ncp + j] + 1 )
                    elif j == 6:
                        pass 
                        # bound focal length
                        #lower.append(self.K[0,0]*0.95)
                        #upper.append(self.K[0,0]*1.05)
                    else:
                        lower.append( -np.inf )
                        upper.append( np.inf )
            for i in range(self.n_points * 3):
                #lower.append( points_3d[i] - tol )
                #upper.append( points_3d[i] + tol )
                # let point locations float without constraint
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
        # bounds=bounds,
        res = least_squares(self.fun, x0,
                            jac_sparsity=A,
                            verbose=2,
                            method='trf',
                            loss='linear',
                            ftol=self.ftol,
                            x_scale='jac',
                            bounds=bounds,
                            args=(self.n_cameras, self.n_points,
                                  self.by_camera_point_indices,
                                  self.by_camera_points_2d))
        t1 = time.time()
        log("Optimization took %.1f seconds" % (t1 - t0))
        # print(res['x'])
        log("res:", res)
        
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

        log("Starting mean reprojection error: %.2f" % mre_start)
        log("Final mean reprojection error: %.2f" % mre_final)
        log("Iterations:", iterations)
        log("Elapsed time = %.1f sec" % time_sec)
        if self.optimize_calib == 'global':
            log("Final camera calib:\n", camera_calib)

        # final plot
        # plt.plot(res.fun)
        # plt.ioff()
        # plt.show()

        return ( self.camera_params, self.points_3d,
                 self.camera_map_fwd, self.feat_map_rev,
                 fx, fy, cu, cv, distCoeffs_opt )

    def update_camera_poses(self, proj):
        log('Updated the optimized camera poses.')
        
        # mark all the optimized poses as invalid
        for image in proj.image_list:
            opt_cam_node = image.node.getChild('camera_pose_opt', True)
            opt_cam_node.setBool('valid', False)

        for i, cam in enumerate(self.camera_params):
            image_index = self.camera_map_fwd[i]
            image = proj.image_list[image_index]
            ned_orig, ypr_orig, quat_orig = image.get_camera_pose()
            # print('optimized cam:', cam)
            rvec = cam[0:3]
            tvec = cam[3:6]
            Rned2cam, jac = cv2.Rodrigues(rvec)
            cam2body = image.get_cam2body()
            Rned2body = cam2body.dot(Rned2cam)
            Rbody2ned = np.matrix(Rned2body).T
            (yaw, pitch, roll) = transformations.euler_from_matrix(Rbody2ned, 'rzyx')
            #print "orig ypr =", image.camera_pose['ypr']
            #print "new ypr =", [yaw/d2r, pitch/d2r, roll/d2r]
            pos = -np.matrix(Rned2cam).T * np.matrix(tvec).T
            newned = pos.T[0].tolist()[0]
            log(image.name, ned_orig, '->', newned, 'dist:', np.linalg.norm(np.array(ned_orig) - np.array(newned)))
            image.set_camera_pose( newned, yaw*r2d, pitch*r2d, roll*r2d, opt=True )
            image.placed = True
        proj.save_images_info()

    # compare original camera locations with optimized camera
    # locations and derive a transform matrix to 'best fit' the new
    # camera locations over the original ... trusting the original
    # group gps solution as our best absolute truth for positioning
    # the system in world coordinates.  (each separately optimized
    # group needs a separate/unique fit)
    def refit(self, proj, matches, groups, group_index):
        matches_opt = list(matches) # shallow copy
        group = groups[group_index]
        log('refitting group size:', len(group))
        src_list = []
        dst_list = []
        # only consider images that are in the current   group
        for name in group:
            image = proj.findImageByName(name)
            ned, ypr, quat = image.get_camera_pose(opt=True)
            src_list.append(ned)
            ned, ypr, quat = image.get_camera_pose()
            dst_list.append(ned)
        A = get_recenter_affine(src_list, dst_list)

        # extract the rotation matrix (R) from the affine transform
        scale, shear, angles, trans, persp = transformations.decompose_matrix(A)
        log('  scale:', scale)
        log('  shear:', shear)
        log('  angles:', angles)
        log('  translate:', trans)
        log('  perspective:', persp)
        R = transformations.euler_matrix(*angles)
        log("R:\n{}".format(R))

        # fixme (just group):

        # update the optimized camera locations based on best fit
        camera_list = []
        # load optimized poses
        for image in proj.image_list:
            if image.name in group:
                ned, ypr, quat = image.get_camera_pose(opt=True)
            else:
                # this is just fodder to match size/index of the lists
                ned, ypr, quat = image.get_camera_pose()
            camera_list.append( ned )

        # refit
        new_cams = transform_points(A, camera_list)

        # update position
        for i, image in enumerate(proj.image_list):
            if not image.name in group:
                continue
            ned, [y, p, r], quat = image.get_camera_pose(opt=True)
            image.set_camera_pose(new_cams[i], y, p, r, opt=True)
        proj.save_images_info()

        if True:
            # update optimized pose orientation.
            dist_report = []
            for i, image in enumerate(proj.image_list):
                if not image.name in group:
                    continue
                ned_orig, ypr_orig, quat_orig = image.get_camera_pose()
                ned, ypr, quat = image.get_camera_pose(opt=True)
                Rbody2ned = image.get_body2ned(opt=True)
                # update the orientation with the same transform to keep
                # everything in proper consistent alignment

                newRbody2ned = R[:3,:3].dot(Rbody2ned)
                (yaw, pitch, roll) = transformations.euler_from_matrix(newRbody2ned, 'rzyx')
                image.set_camera_pose(new_cams[i], yaw*r2d, pitch*r2d, roll*r2d,
                                      opt=True)
                dist = np.linalg.norm( np.array(ned_orig) - np.array(new_cams[i]))
                qlog("image:", image.name)
                qlog("  orig pos:", ned_orig)
                qlog("  fit pos:", new_cams[i])
                qlog("  dist moved:", dist)
                dist_report.append( (dist, image.name) )
            proj.save_images_info()

            dist_report = sorted(dist_report,
                                 key=lambda fields: fields[0],
                                 reverse=False)
            log("Image movement sorted lowest to highest:")
            for report in dist_report:
                log(report[1], "dist:", report[0])

        # tranform the optimized point locations using the same best
        # fit transform for the camera locations.
        new_feats = transform_points(A, self.points_3d)

        # update any of the transformed feature locations that have
        # membership in the currently processing group back to the
        # master match structure.  Note we process groups in order of
        # little to big so if a match is in more than one group it
        # follows the larger group.
        for i, feat in enumerate(new_feats):
            match_index = self.feat_map_rev[i]
            match = matches_opt[match_index]
            in_group = False
            for m in match[2:]:
                if proj.image_list[m[0]].name in group:
                    in_group = True
                    break
            if in_group:
                #print(" before:", match)
                match[0] = feat
                #print(" after:", match)
