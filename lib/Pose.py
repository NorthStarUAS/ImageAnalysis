#!/usr/bin/python

# pose related functions

import fileinput
import math
import os
import re

import navpy
from props import getNode

import transformations
import Image

# a helpful constant
d2r = math.pi / 180.0
r2d = 180.0 / math.pi

# quaternions represent a rotation from one coordinate system to
# another (i.e. from NED space to aircraft body space).  You can
# back translate a vector against this quaternion to project a camera
# vector into NED space.
#
# body angles represent the aircraft orientation
# camera angles represent the fixed mounting offset of the camera
# relative to the body
# +x axis = forward, +roll = left wing down
# +y axis = right, +pitch = nose down
# +z axis = up, +yaw = nose right


# define the image aircraft poses from Sentera meta data file
def setAircraftPoses(proj, posefile="", order='ypr'):
    images_dir = proj.dir_node.getString('images_source')
    meta_dir = os.path.join(proj.project_dir, 'meta')
    proj.image_list = []
    
    f = fileinput.input(posefile)
    for line in f:
        line.strip()
        if re.match('^\s*#', line):
            print("skipping comment ", line)
            continue
        if re.match('^\s*File', line):
            print("skipping header ", line)
            continue
        field = line.split(',')
        name = field[0]
        lat_deg = float(field[1])
        lon_deg = float(field[2])
        alt_m = float(field[3])
        if order == 'ypr':
            ya_deg = float(field[4])
            pitch_deg = float(field[5])
            roll_deg = float(field[6])
        elif order == 'rpy':
            roll_deg = float(field[4])
            pitch_deg = float(field[5])
            yaw_deg = float(field[6])
            
        if not os.path.isfile( os.path.join(images_dir, name) ):
            # no associated full image file, skip
            continue
        if abs(roll_deg) > 25.0:
            # rolled into a turn, skip
            print('skipping:', name, 'roll:', roll_deg)
            continue

        base, ext = os.path.splitext(name)
        image = Image.Image(images_dir, meta_dir, base)
        image.set_aircraft_pose(lat_deg, lon_deg, alt_m,
                                yaw_deg, pitch_deg, roll_deg)
        print(name, 'yaw=%.1f pitch=%.1f roll=%.1f' % (yaw_deg, pitch_deg, roll_deg))
        proj.image_list.append(image)

# for each image, compute the estimated camera pose in NED space from
# the aircraft body pose and the relative camera orientation
def compute_camera_poses(proj):
    mount_node = getNode('/config/camera/mount', True)
    ref_node = getNode('/config/ned_reference', True)
    images_node = getNode('/images', True)

    camera_yaw = mount_node.getFloat('yaw_deg')
    camera_pitch = mount_node.getFloat('pitch_deg')
    camera_roll = mount_node.getFloat('roll_deg')
    print(camera_yaw, camera_pitch, camera_roll)
    body2cam = transformations.quaternion_from_euler(camera_yaw * d2r,
                                                     camera_pitch * d2r,
                                                     camera_roll * d2r,
                                                     'rzyx')

    ref_lat = ref_node.getFloat('lat_deg')
    ref_lon = ref_node.getFloat('lon_deg')
    ref_alt = ref_node.getFloat('alt_m')

    for image in proj.image_list:
        ac_pose_node = image.node.getChild("aircraft_pose", True)
        #cam_pose_node = images_node.getChild(name + "/camera_pose", True)
        
        aircraft_lat = ac_pose_node.getFloat('lat_deg')
        aircraft_lon = ac_pose_node.getFloat('lon_deg')
        aircraft_alt = ac_pose_node.getFloat('alt_m')
        ned2body = []
        for i in range(4):
            ned2body.append( ac_pose_node.getFloatEnum('quat', i) )
        
        ned2cam = transformations.quaternion_multiply(ned2body, body2cam)
        (yaw_rad, pitch_rad, roll_rad) = transformations.euler_from_quaternion(ned2cam, 'rzyx')
        ned = navpy.lla2ned( aircraft_lat, aircraft_lon, aircraft_alt,
                             ref_lat, ref_lon, ref_alt )

        image.set_camera_pose(ned, yaw_rad*r2d, pitch_rad*r2d, roll_rad*r2d)

