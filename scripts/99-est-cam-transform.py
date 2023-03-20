#!/usr/bin/env python3

# after optimizing camera poses and feature locations, let's estimate
# the average transform between the original aircraft pose (estimated
# from flight data) and the optimized camera pose.  This gives us some
# sort of average estimate of the rigid transform from aircraft body
# to camera frame of reference.

import argparse
from math import pi
import numpy as np
import numpy.matlib as npm

from props import getNode
import transformations

from lib import groups
from lib import project

r2d = 180.0 / pi
d2r = pi / 180.0

parser = argparse.ArgumentParser(description='Set the aircraft poses from flight data.')
parser.add_argument('project', help='project directory')

args = parser.parse_args()

proj = project.ProjectMgr(args.project)
print("Loading image info...")
proj.load_images_info()

group_list = groups.load(proj.analysis_dir)
print(group_list)

# compute an average transform between original camera attitude estimate
# and optimized camera attitude estimate
quats = []
for i, image in enumerate(proj.image_list):
    if image.name in group_list[0]:
        print(image.name)
        ned, ypr, q0 = image.get_camera_pose(opt=False)
        ned, ypr, q1 = image.get_camera_pose(opt=True)
        # rx = q1 * conj(q0)
        conj_q0 = transformations.quaternion_conjugate(q0)
        rx = transformations.quaternion_multiply(q1, conj_q0)
        rx /= transformations.vector_norm(rx)
        print(' ', rx)
        (yaw_rad, pitch_rad, roll_rad) = transformations.euler_from_quaternion(rx, 'rzyx')
        print('euler (ypr):', yaw_rad*r2d, pitch_rad*r2d, roll_rad*r2d)
        quats.append(rx)

# https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py
#
# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation

def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    print('averaging # quats:', M)
    A = npm.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)


q = averageQuaternions(np.array(quats))
q /= transformations.vector_norm(q)
print('average transform:', q)
(yaw_rad, pitch_rad, roll_rad) = transformations.euler_from_quaternion(q, 'rzyx')
print('euler transform (ypr):', yaw_rad*r2d, pitch_rad*r2d, roll_rad*r2d)

mount_node = getNode('/config/camera/mount', True)
camera_yaw = mount_node.getFloat('yaw_deg')
camera_pitch = mount_node.getFloat('pitch_deg')
camera_roll = mount_node.getFloat('roll_deg')
print('camera:', camera_yaw, camera_pitch, camera_roll)
body2cam = transformations.quaternion_from_euler(camera_yaw * d2r,
                                                 camera_pitch * d2r,
                                                 camera_roll * d2r,
                                                 'rzyx')
print('body2cam:', body2cam)
(yaw_rad, pitch_rad, roll_rad) = transformations.euler_from_quaternion(body2cam, 'rzyx')
print('euler body2cam (ypr):', yaw_rad*r2d, pitch_rad*r2d, roll_rad*r2d)

cam2body = transformations.quaternion_inverse(body2cam)
print('cam2body:', cam2body)
(yaw_rad, pitch_rad, roll_rad) = transformations.euler_from_quaternion(cam2body, 'rzyx')
print('euler cam2body (ypr):', yaw_rad*r2d, pitch_rad*r2d, roll_rad*r2d)

tot = transformations.quaternion_multiply(q, body2cam)
(yaw_rad, pitch_rad, roll_rad) = transformations.euler_from_quaternion(tot, 'rzyx')
print(tot)
print('euler (ypr):', yaw_rad*r2d, pitch_rad*r2d, roll_rad*r2d)

tot_inv = transformations.quaternion_inverse(tot)

def wrap_pi(val):
    while val < pi:
        val += 2*pi
    while val > pi:
        val -= 2*pi
    return val

# test correcting the original aircraft attitude from optimized camera attitude
for i, image in enumerate(proj.image_list):
    if image.name in group_list[0]:
        lla, ypr, q_aircraft = image.get_aircraft_pose()
        ned_init, ypr, q_cam_initial = image.get_camera_pose(opt=False)
        ned_opt, ypr, q_cam_opt = image.get_camera_pose(opt=True)
        q_aircraft_corrected = transformations.quaternion_multiply(q_cam_opt, tot_inv)
        (orig_yaw_rad, orig_pitch_rad, orig_roll_rad) = transformations.euler_from_quaternion(q_aircraft, 'rzyx')
        #print('  euler aircraft (ypr):', orig_yaw_rad*r2d, orig_pitch_rad*r2d, orig_roll_rad*r2d)
        (corr_yaw_rad, corr_pitch_rad, corr_roll_rad) = transformations.euler_from_quaternion(q_aircraft_corrected, 'rzyx')
        #print('  euler correctd (ypr):', corr_yaw_rad*r2d, corr_pitch_rad*r2d, corr_roll_rad*r2d)
        timestamp = image.node.getFloat('flight_time')
        yaw_error_rad = wrap_pi(corr_yaw_rad - orig_yaw_rad)
        pitch_error_rad = wrap_pi(corr_pitch_rad - orig_pitch_rad)
        roll_error_rad = wrap_pi(corr_roll_rad - orig_roll_rad)
        n_error_m = ned_opt[0] - ned_init[0]
        e_error_m = ned_opt[1] - ned_init[1]
        d_error_m = ned_opt[2] - ned_init[2]
        print(timestamp, yaw_error_rad, pitch_error_rad, roll_error_rad, n_error_m, e_error_m, d_error_m)


