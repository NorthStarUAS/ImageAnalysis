#!/usr/bin/python

# pose related functions

import fileinput
import math
import re

import navpy
from props import getNode

import transformations

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
def setAircraftPoses(proj, metafile="", order='ypr'):
    f = fileinput.input(metafile)
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
        quat = transformations.quaternion_from_euler(yaw_deg * d2r,
                                                     pitch_deg * d2r,
                                                     roll_deg * d2r,
                                                     'rzyx')
        images_node = getNode("/images", True)
        image_node = images_node.getChild(name, True)
        ac_pose_node = image_node.getChild('aircraft_pose', True)
        ac_pose_node.setFloat('lat_deg', lat_deg)
        ac_pose_node.setFloat('lon_deg', lon_deg)
        ac_pose_node.setFloat('alt_m', alt_m)
        ac_pose_node.setFloat('yaw_deg', yaw_deg)
        ac_pose_node.setFloat('pitch_deg', pitch_deg)
        ac_pose_node.setFloat('roll_deg', roll_deg)
        ac_pose_node.setLen('quat', 4)
        for i in range(4):
            ac_pose_node.setFloatEnum('quat', i, quat[i])
        print(name, 'yaw=%.1f pitch=%.1f roll=%.1f' % (yaw_deg, pitch_deg, roll_deg))

    #getNode("/images").pretty_print()
    
    # save the changes
    proj.save_images_info()
                
# compute the camera pose in NED space, assuming the aircraft
# body pose has already been computed in lla space and the orientation
# transform is represented as a quaternion.
def computeCameraPoseFromAircraft(image, cam, ref):
    lla, ypr, ned2body = image.get_aircraft_pose()
    aircraft_lat, aircraft_lon, aircraft_alt = lla
    #print "aircraft quat =", world2body
    msl = aircraft_alt + image.alt_bias

    (camera_yaw, camera_pitch, camera_roll) = cam.get_mount_params()
    body2cam = transformations.quaternion_from_euler(camera_yaw * d2r,
                                                     camera_pitch * d2r,
                                                     camera_roll * d2r,
                                                     'rzyx')
    ned2cam = transformations.quaternion_multiply(ned2body, body2cam)
    (yaw, pitch, roll) = transformations.euler_from_quaternion(ned2cam, 'rzyx')
    ned = navpy.lla2ned( aircraft_lat, aircraft_lon, aircraft_alt,
                         ref[0], ref[1], ref[2] )
    #print "aircraft=%s ref=%s ned=%s" % (image.get_aircraft_pose(), ref, ned)
    return (ned.tolist(), [yaw/d2r, pitch/d2r, roll/d2r])


# for each image, compute the estimated camera pose in NED space from
# the aircraft body pose and the relative camera orientation
def compute_camera_poses():
    mount_node = getNode('/config/camera/mount', True)
    ref_node = getNode('/config/ned_reference', True)
    images_node = getNode('/images', True)

    camera_yaw = mount_node.getFloat('yaw_deg')
    camera_pitch = mount_node.getFloat('pitch_deg')
    camera_roll = mount_node.getFloat('roll_deg')
    body2cam = transformations.quaternion_from_euler(camera_yaw * d2r,
                                                     camera_pitch * d2r,
                                                     camera_roll * d2r,
                                                     'rzyx')

    ref_lat = ref_node.getFloat('lat_deg')
    ref_lon = ref_node.getFloat('lon_deg')
    ref_alt = ref_node.getFloat('alt_m')
    
    for name in images_node.getChildren():
        ac_pose_node = images_node.getChild(name + "/aircraft_pose", True)
        cam_pose_node = images_node.getChild(name + "/camera_pose", True)
        
        #ned, ypr, quat = image.get_camera_pose()
        #ned, ypr = computeCameraPoseFromAircraft(image, cam, ref)
        #lla, ypr, ned2body = image.get_aircraft_pose()
        
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
        #print "aircraft=%s ref=%s ned=%s" % (image.get_aircraft_pose(), ref, ned)
        #return (ned.tolist(), [yaw/d2r, pitch/d2r, roll/d2r])
        
        for i in range(3):
            cam_pose_node.setFloatEnum('ned', i, ned[i])
        cam_pose_node.setFloat('yaw_deg', yaw_rad * r2d)
        cam_pose_node.setFloat('pitch_deg', pitch_rad * r2d)
        cam_pose_node.setFloat('roll_deg', roll_rad * r2d)
        cam_pose_node.setLen('quat', 4)
        for i in range(4):
            cam_pose_node.setFloatEnum('quat', i, ned2cam[i])

        print('camera pose:', name)
        cam_pose_node.pretty_print("  ")

