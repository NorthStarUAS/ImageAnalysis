#!/usr/bin/python

# pose related functions

import fileinput
import math

import navpy

import transformations

# a helpful constant
d2r = math.pi / 180.0

# quaternions represent a rotation from one coordinate system to
# another (i.e. from cartesian space to aircraft body space).  You can
# back translate a vector against this quaternion to project a camera
# vector into cartesian space.
#
# body angles represent the aircraft orientation
# camera angles represent the fixed mounting offset of the camera
# relative to the body
# +x axis = forward, +roll = left wing down
# +y axis = right, +pitch = nose down
# +z axis = up, +yaw = nose right


# from Sentera meta data file
def setAircraftPoses(proj, metafile="", force=True, weight=True):
    f = fileinput.input(metafile)
    for line in f:
        line.strip()
        field = line.split(',')
        name = field[0]
        lat = float(field[1])
        lon = float(field[2])
        msl = float(field[3])
        yaw = float(field[4])
        pitch = float(field[5])
        roll = float(field[6])

        image = proj.findImageByName(name)
        if image != None:
            if force or (math.fabs(image.aircraft_lon) < 0.01 and math.fabs(image.aircraft_lat) < 0.01):
                image.set_aircraft_pose( lon, lat, msl, roll, pitch, yaw )
                image.weight = 1.0
                image.save_meta()
                print "%s yaw=%.1f pitch=%.1f roll=%.1f" % (image.name, yaw, pitch, roll)

# assuming the aircraft body pose has already been computed and
# represented as a quaternion, compute the camera pose in cartesian
# space.
def computeCameraPoseFromAircraft(image, cam, ref,
                                  yaw_bias=0.0, roll_bias=0.0,
                                  pitch_bias=0.0, alt_bias=0.0):
    aircraft_lon, aircraft_lat, aircraft_alt, aircraft_roll, aircraft_pitch, aircraft_yaw, world2body = image.get_aircraft_pose()
    print "aircraft quat =", world2body
    msl = aircraft_alt + image.alt_bias + alt_bias

    (camera_yaw, camera_pitch, camera_roll) = cam.get_mount_params()
    body2cam = transformations.quaternion_from_euler(camera_yaw * d2r,
                                                     camera_pitch * d2r,
                                                     camera_roll * d2r,
                                                     'rzyx')
    world2cam = transformations.quaternion_multiply(world2body, body2cam)
    (yaw, pitch, roll) = transformations.euler_from_quaternion(world2cam, 'rzyx')
    ref_lon = ref['longitude-deg']
    ref_lat = ref['latitude-deg']
    ref_alt = ref['altitude-m']
    ned = navpy.lla2ned(aircraft_lat, aircraft_lon, aircraft_alt,
                        ref_lat, ref_lon, ref_alt)
    print "aircraft=%s ref=%s ned=%s" % (image.get_aircraft_pose(), ref, ned)
    return (yaw/d2r, pitch/d2r, roll/d2r, ned[1], ned[0], -ned[2])

# for each image in the provided image list, compute the estimated
# camera pose in cartesian space assuming the aircraft body pose has
# already been determined.
def compute_camera_poses(image_list, cam, ref, force=False):
    for image in image_list:
        x, y, z, roll, pitch, yaw, quat = image.get_camera_pose()
        if not force and \
           (math.fabs(yaw) > 0.001 or math.fabs(pitch) > 0.001 \
            or math.fabs(roll) > 0.001 or math.fabs(x) > 0.001 \
            or math.fabs(y) > 0.001 or math.fabs(z) > 0.001):
            continue

        pose = computeCameraPoseFromAircraft(image, cam, ref)
        print "pose from aircraft = %s" % str(pose)
        image.set_camera_pose( pose[3], pose[4], pose[5],
                               pose[0], pose[1], pose[2])

