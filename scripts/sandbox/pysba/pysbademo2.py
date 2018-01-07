#!/usr/bin/python3

# reference:
#
# http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
#

# this script is a modification of the original demo script to
# optimize only 6 individual camera paramters (rotation and
# translation).  focal len, k1, and k2 are 3 additional parameters
# that are optimized universally.  The intension/assumption is that
# all images in a set are captured with the same camera and we want to
# also refine the camera calibration estimate as part of the overall
# optimization.
#
# additional modifications to help understand the coordinate system
# conventions used here and in 'bal'.

from __future__ import print_function

import urllib
import bz2
import os
import numpy as np

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")
import cv2

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
#FILE_NAME = "problem-138-19878-pre.txt.bz2"
#FILE_NAME = "problem-1064-113655-pre.txt.bz2"
URL = BASE_URL + FILE_NAME

FILE_NAME = '../../rw87/bundler.txt.bz2'
K_FILE = '../../rw87/sba-calib.txt'

if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)
    
def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))
        f = np.mean(camera_params[:, 6])
        k1 = np.mean(camera_params[:, 7])
        k2 = np.mean(camera_params[:, 8])
        print('focal len = {}'.format(f))
        print('k1 = {}'.format(k1))
        print('k2 = {}'.format(k2))
        camera_params = camera_params[:,0:6]
        
        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))
    K = []
    with open(K_FILE, 'r') as file:
        for i in range(3):
            line = file.readline()
            K.append(list(map(float, line.split())))
    K = np.array(K)
    print('K: {}'.format(K))
    
    return camera_params, points_3d, camera_indices, point_indices, points_2d, f, k1, k2, K

camera_params, points_3d, camera_indices, point_indices, points_2d, f, k1, k2, K = read_bal_data(FILE_NAME)

n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]

n = 6 * n_cameras + 3 * n_points + 3
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params, calib_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    f = calib_params[0]
    k1 = calib_params[1]
    k2 = calib_params[2]
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def project2(points, cam_M, calib_params):
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

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
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
    calib_params = params[n_cameras * 6 + n_points * 3:]
    #print("calib:")
    #print(calib_params.shape)
    #print(calib_params)
    tmp = cam_M[camera_indices]
    #print('tmp.shape {}'.format(tmp.shape))
    points_proj = project2(points_3d[point_indices],
                           cam_M[camera_indices],
                           calib_params)
    # mre
    error = (points_proj - points_2d).ravel()
    mre = np.mean(np.abs(error))
    print("mre = {}".format(mre))
    return (points_proj - points_2d).ravel()

from scipy.sparse import lil_matrix

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3 + 3
    A = lil_matrix((m, n), dtype=int)
    print('sparcity matrix is {} x {}'.format(m, n))

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + n_points * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + n_points * 3 + s] = 1

    print('A non-zero elements = {}'.format(A.nnz))
    
    return A

import matplotlib.pyplot as plt
x0 = np.hstack((camera_params.ravel(), points_3d.ravel(), f, k1, k2))
print('x0:')
print(x0.shape)
print(x0)
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
plt.plot(f0)

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

import time
from scipy.optimize import least_squares

t0 = time.time()
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac',
                    ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices,
                          points_2d))
t1 = time.time()
print("Optimization took {0:.0f} seconds".format(t1 - t0))
print(res['x'])
plt.plot(res.fun)

plt.show()

