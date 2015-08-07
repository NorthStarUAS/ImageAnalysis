#!/usr/bin/python

# pose related functions

import fileinput
import math

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
                print "%s roll=%.1f pitch=%.1f" % (image.name, roll, pitch, image.weight)

# assuming the aircraft body pose has already been determined,
# compute the camera pose as a new set of euler angles and NED.
def computeCameraPoseFromAircraft(image,
                                  yaw_bias=0.0, roll_bias=0.0,
                                  pitch_bias=0.0, alt_bias=0.0):
    aircraft_lon, aircraft_lat, aircraft_alt, aircraft_roll, aircraft_pitch, aircraft_yaw = image.get_aircraft_pose()
    msl = aircraft_alt + image.alt_bias + alt_bias

    # aircraft orientation includes our per camera bias
    # (representing the aircraft attitude estimate error
    body_roll = -(aircraft_roll + image.roll_bias + self.group_roll_bias)
    body_pitch = -(aircraft_pitch + image.pitch_bias + self.group_pitch_bias)
    body_yaw = aircraft_yaw + image.yaw_bias + self.group_yaw_bias

    # camera orientation includes our group biases
    # (representing the estimated mounting alignment error of
    # the camera relative to the aircraft)
    cam_yaw_bias = 0.0
    cam_pitch_bias = 0.0
    cam_roll_bias = 0.0
    camera_yaw = self.offset_yaw_deg + cam_yaw_bias
    camera_pitch = -(self.offset_pitch_deg + cam_pitch_bias)
    camera_roll = -(self.offset_roll_deg + cam_roll_bias)

    ned2cam = self.computeNed2Cam(body_yaw, body_pitch, body_roll,
                                  camera_yaw, camera_pitch, camera_roll)
    (yaw, pitch, roll) = transformations.euler_from_quaternion(ned2cam,
                                                               'rzyx')
    (x, y) = ImageList.wgs842cart(aircraft_lon, aircraft_lat, self.ref_lon, self.ref_lat)
    x += image.x_bias
    y += image.y_bias
    z = msl - self.ground_alt_m

    deg2rad = math.pi / 180.0
    return (yaw/deg2rad, pitch/deg2rad, roll/deg2rad, x, y, z)

# for each image in the provided image list, compute the estimated
# camera pose in cartesian space assuming the aircraft body pose has
# already been determined.
def compute_camera_poses(image_list, force=False):
    for image in image_list:
        x, y, z, roll, pitch, yaw = image.get_camera_pose()
        if not force and \
           (math.fabs(yaw) > 0.001 or math.fabs(pitch) > 0.001 \
            or math.fabs(roll) > 0.001 or math.fabs(x) > 0.001 \
            or math.fabs(y) > 0.001 or math.fabs(z) > 0.001):
            continue

        pose = computeCameraPoseFromAircraft(image)
        #print "pose from aircraft = %s" % str(pose)
        image.set_camera_pose( pose[3], pose[4], pose[5],
                               pose[0], pose[1], pose[2])

