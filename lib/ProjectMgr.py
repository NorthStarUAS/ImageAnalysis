#!/usr/bin/python

import commands
import cv2
import fileinput
import fnmatch
import json
import math
from matplotlib import pyplot as plt
import numpy as np
import os.path
import subprocess
import sys

import geojson

from getchar import find_getch
import Image
import ImageList
import Matcher
import Placer
import Render
import transformations


class ProjectMgr():
    def __init__(self, project_dir=None, max_features=100, detect_grid=8, match_ratio=0.75):
        # directories
        self.project_dir = None  # project working directory
        self.source_dir = None   # original images
        self.image_dir = None    # working set of images

        self.image_list = []

        self.detector = 'SIFT'  # one of SIFT, SURF, or ORB
        detector_params = { 'nfeatures': 2000 } # SIFT
        #detector_params = { 'hessian_threshold': 600 } # SURF
        #detector_params = { 'nfeatures': 2000, 'dense_detect_grid': 4 } # ORB

        # the following member variables need to be reviewed/organized

        self.flight_data = None  # flight data
        cells = detect_grid * detect_grid
        self.max_features = int(max_features / cells)
        self.match_ratio = match_ratio
        self.detect_grid = detect_grid
        self.ac3d_steps = 8
        self.ref_lon = None
        self.ref_lat = None
        self.shutter_latency = 0.0
        self.group_roll_bias = 0.0
        self.group_pitch_bias = 0.0
        self.group_yaw_bias = 0.0
        #self.group_alt_bias = 0.0
        self.k1 = 0.0
        self.k2 = 0.0
        self.m = Matcher.Matcher()
        self.placer = Placer.Placer()
        self.render = Render.Render()
        matcherparams = { 'matcher': 'FLANN', 'match_ratio': match_ratio }
        matcherparams = { 'matcher': 'Brute Force', 'match_ratio': match_ratio }
        self.m.configure(detectparams, matcherparams)
        
        if project_dir != None:
            self.load( project_dir )

    # project_dir is a new folder for all derived files
    def set_project_dir(self, project_dir, create_if_needed=True):
        self.project_dir = project_dir
        
        if not os.path.exists(self.project_dir):
            if create_if_needed:
                print "Notice: creating project directory =", self.project_dir
                os.makedirs(self.project_dir)
            else:
                print "Error: project dir doesn't exist =", self.project_dir
                return False

        # and make children directories
        self.image_dir = project_dir + "/" + "Images"
        if not os.path.exists(self.image_dir):
            if create_if_needed:
                print "Notice: creating image directory =", self.image_dir
                os.makedirs(self.image_dir)
            else:
                print "Error: image dir doesn't exist =", self.image_dir
                return False
            
        # all is good
        return True

    # source_dir is the folder containing all the raw/original images.
    # The expected work flow is that we will import/scale all the
    # original images into our project folder leaving the original
    # image set completely untouched.
    def set_source_dir(self, source_dir):
        if source_dir == self.project_dir:
            print "Error: image source and project dirs must be different."
            return

        if not os.path.exists(source_dir):
            print "Error: image source path does not exist =", source_path
            
        self.source_dir = source_dir

    def save(self):
        # create a dictionary and write it out as json
        if not os.path.exists(self.project_dir):
            print "Error: project doesn't exist =", self.project_dir
            return

        dirs = {}
        dirs['source'] = self.source_dir
        dirs['image'] = self.image_dir
        project_dict = {}
        project_dict['directories'] = dirs
        project_file = self.project_dir + "/Project.json"
        try:
            f = open(project_file, 'w')
            json.dump(project_dict, f, indent=4, sort_keys=True)
            f.close()
        except IOError as e:
            print "Save project(): I/O error({0}): {1}".format(e.errno, e.strerror)
            return
        except:
            raise

    def load(self, project_dir, create_if_needed=True):
        if not self.set_project_dir( project_dir ):
            return

        project_file = self.project_dir + "/Project.json"
        try:
            f = open(project_file, 'r')
            project_dict = json.load(f)
            f.close()
            
            dirs = project_dict['directories']
            self.source_dir = dirs['source']
            self.image_dir = dirs['image']
        except:
            print "Error: unable to read =", project_file
            print "Continuing with an empty project configuration"

    # import an image set into the project directory, possibly scaling them
    # to a lower resolution for faster processing.
    def import_images(self, scale=0.25, converter='imagemagick'):
        if self.source_dir == None:
            print "Error: source_dir not defined."
            return

        if self.image_dir == None:
            print "Error: project's image_dir not defined."
            return
        
        if self.source_dir == self.image_dir:
            print "Error: source and destination directories must be different."
            return

        if not os.path.exists(self.source_dir):
            print "Error: source directory not found =", self.source_dir
            return

        if not os.path.exists(self.image_dir):
            print "Error: destination directory not found =", self.image_dir
            return

        files = []
        for file in os.listdir(self.source_dir):
            if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
                files.append(file)
        files.sort()

        for file in files:
            name_in = self.source_dir + "/" + file
            name_out = self.image_dir + "/" + file
            if converter == 'imagemagick':
                command = "convert -resize %d%% %s %s" % ( int(scale*100.0), name_in, name_out )
                print command
                commands.getstatusoutput( command )
            elif converter == 'opencv':
                src = cv2.imread(name_in)
                #method = cv2.INTER_AREA
                method = cv2.INTER_LANCZOS4
                dst = cv2.resize(src, (0,0), fx=scale, fy=scale,
                                 interpolation=method)
                cv2.imwrite(name_out, dst)
                print "Scaling (%.1f%%) %s to %s" % ((scale*100.0), name_in, name_out)
            else:
                print "Error: unknown converter =", converter

    def load_image_info(self):
        file_list = []
        for file in os.listdir(self.image_dir):
            if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
                file_list.append(file)
        file_list.sort()

        # wipe image list (so we don't double load)
        self.image_list = []
        for file_name in file_list:
            image = Image.Image(self.image_dir, file_name)
            self.image_list.append( image )

        # make sure our matcher gets a copy of the image list
        self.m.setImageList(self.image_list)
        self.placer.setImageList(self.image_list)
        self.render.setImageList(self.image_list)

    def load_features(self):
        for image in self.image_list:
            print "Loading keypoints and descriptors: " + image.name
            image.load_keys()
            image.load_descriptors()

    def load_matches(self):
        print "Loading pair matches"
        for image in self.image_list:
            image.load_matches()

    def save_info(self):
        for image in self.image_list:
            image.save_info()

    def detect_features(self, force=True):
        for image in self.image_list:
            if force or len(image.kp_list) == 0 or image.des_list == None:
                print "detecting features and computing descriptors: " + image.name
                if image.img_rgb == None:
                    image.load_rgb()
                image.kp_list = self.m.denseDetect(image.img_rgb)
                image.kp_list, image.des_list \
                    = self.m.computeDescriptors(image.img_rgb, image.kp_list)
                # wipe matches because we've messed with keypoints and
                # descriptors
                image.match_list = []
                image.save_keys()
                image.save_descriptors()
                #image.show_keypoints()

    def setCameraParams(self, horiz_mm=23.5, vert_mm=15.7, focal_len_mm=30.0):
        self.horiz_mm = horiz_mm
        self.vert_mm = vert_mm
        self.focal_len_mm = focal_len_mm

    def setCameraOffsets(self, roll_deg=0.0, pitch_deg=-90, yaw_deg=0.0):
        self.offset_roll_deg = roll_deg
        self.offset_pitch_deg = pitch_deg
        self.offset_yaw_deg = yaw_deg

    def setWorldParams(self, ground_alt_m=0.0, shutter_latency=0.0,
                       yaw_bias=0.0, roll_bias=0.0, pitch_bias=0.0):
        print "Setting ground=%.1f shutter=%.2f yaw=%.2f roll=%.2f pitch=%.2f"\
            % (ground_alt_m, shutter_latency, yaw_bias, roll_bias, pitch_bias)
        self.ground_alt_m = ground_alt_m
        self.shutter_latency = shutter_latency
        self.group_yaw_bias = yaw_bias
        self.group_roll_bias = roll_bias
        self.group_pitch_bias = pitch_bias

    def genKeypointUsageMap(self):
        # make the keypoint usage map (used so we don't have to
        # project every keypoint every time)
        print "Building the keypoint usage map... ",
        for i1 in self.image_list:
            i1.kp_usage = np.zeros(len(i1.kp_list), np.bool_)
        for i, i1 in enumerate(self.image_list):
            for j, pairs in enumerate(i1.match_list):
                if len(pairs) == 0:
                    continue
                if i == j:
                    continue
                i2 = self.image_list[j]
                print "%s vs %s" % (i1.name, i2.name)
                for pair in pairs:
                    i1.kp_usage[pair[0]] = True
                    i2.kp_usage[pair[1]] = True
        print "done."

    def interpolateAircraftPositions(self, correlator, force=False,
                                     weight=True):
        # tag each image with the flight data parameters at the time
        # the image was taken
        for match in correlator.best_matchups:
            pict, trig = correlator.get_match(match)
            image = self.m.findImageByName(pict[2])
            if image != None:
                if force or (math.fabs(image.aircraft_lon) < 0.01 and math.fabs(image.aircraft_lat) < 0.01):
                    # only if we are forcing a new position
                    # calculation or the position is not already set
                    # from a save file.
                    t = trig[0] + self.shutter_latency
                    lon, lat, msl = correlator.get_position(t)
                    roll, pitch, yaw = correlator.get_attitude(t)
                    image.set_location( lon, lat, msl, roll, pitch, yaw )
                    if weight:
                        # presumes a pitch/roll distance of 10, 10 gives a
                        # zero weight
                        w = 1.0 - (roll*roll + pitch*pitch)/200.0
                        if w < 0.01:
                            w = 0.01
                        image.weight = w
                    else:
                        image.weight = 1.0
                    image.save_info()
                    #print "%s roll=%.1f pitch=%.1f weight=%.2f" % (image.name, roll, pitch, image.weight)

    # from Sentera meta data file
    def setAircraftPositions(self, meta_file="", force=False, weight=True):
        f = fileinput.input(meta_file)
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
            
            image = self.m.findImageByName(name)
            if image != None:
                if force or (math.fabs(image.aircraft_lon) < 0.01 and math.fabs(image.aircraft_lat) < 0.01):
                    image.set_location( lon, lat, msl, roll, pitch, yaw )
                    if weight:
                        # presumes a pitch/roll distance of 10, 10 gives a
                        # zero weight
                        w = 1.0 - (roll*roll + pitch*pitch)/200.0
                        if w < 0.01:
                            w = 0.01
                        image.weight = w
                    else:
                        image.weight = 1.0
                    image.save_info()
                    print "%s roll=%.1f pitch=%.1f weight=%.2f" % (image.name, roll, pitch, image.weight)



    # assuming the aircraft body pose has already been determined,
    # compute the camera pose as a new set of euler angles and NED.
    def computeCameraPoseFromAircraft(self, image,
                                      yaw_bias=0.0, roll_bias=0.0,
                                      pitch_bias=0.0, alt_bias=0.0):
        lon = image.aircraft_lon
        lat = image.aircraft_lat
        msl = image.aircraft_msl + image.alt_bias + alt_bias

        # aircraft orientation includes our per camera bias
        # (representing the aircraft attitude estimate error
        body_roll = -(image.aircraft_roll + image.roll_bias + self.group_roll_bias)
        body_pitch = -(image.aircraft_pitch + image.pitch_bias + self.group_pitch_bias)
        body_yaw = image.aircraft_yaw + image.yaw_bias + self.group_yaw_bias

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
        (x, y) = ImageList.wgs842cart(lon, lat, self.ref_lon, self.ref_lat)
        x += image.x_bias
        y += image.y_bias
        z = msl - self.ground_alt_m

        deg2rad = math.pi / 180.0
        return (yaw/deg2rad, pitch/deg2rad, roll/deg2rad, x, y, z)

    # assuming the aircraft body pose has already been determined,
    # compute the camera pose as a new set of euler angles and
    # position in cartesian space.
    def estimateCameraPoses(self, force=False):
        for image in self.image_list:
            if not force and \
               (math.fabs(image.camera_yaw) > 0.001 \
                or math.fabs(image.camera_pitch) > 0.001 \
                or math.fabs(image.camera_roll) > 0.001 \
                or math.fabs(image.camera_x) > 0.001 \
                or math.fabs(image.camera_y) > 0.001 \
                or math.fabs(image.camera_z) > 0.001):
                continue
        
            pose = self.computeCameraPoseFromAircraft(image)
            #print "pose from aircraft = %s" % str(pose) 
            image.camera_yaw = pose[0]
            image.camera_pitch = pose[1]
            image.camera_roll = pose[2]
            image.camera_x = pose[3]
            image.camera_y = pose[4]
            image.camera_z = pose[5]

    def computeWeights(self, force=None):
        # tag each image with the flight data parameters at the time
        # the image was taken
        for image in self.image_list:
            roll = image.aircraft_roll + image.roll_bias
            pitch = image.aircraft_pitch + image.pitch_bias
            if force != None:
                image.weight = force
            else:
                # presumes a pitch/roll distance of 10, 10 gives a
                # zero weight
                w = 1.0 - (roll*roll + pitch*pitch)/200.0
                if w < 0.01:
                    w = 0.01
                    image.weight = w
            image.save_info()
            #print "%s roll=%.1f pitch=%.1f weight=%.2f" % (image.name, roll, pitch, image.weight)

    def computeConnections(self, force=None):
        for image in self.image_list:
            image.connections = 0
            for pairs in image.match_list:
                if len(pairs) >= self.m.min_pairs:
                    image.connections += 1
            image.save_info()
            print "%s connections: %d" % (image.name, image.connections)

    # compute a center reference location (lon, lat) for the group of
    # images, then compute the x, y offset for each image relative to
    # the reference position.
    def computeRefLocation(self):
        # requires images to have their location computed/loaded
        lon_sum = 0.0
        lat_sum = 0.0
        for image in self.image_list:
            lon_sum += image.aircraft_lon
            lat_sum += image.aircraft_lat
        self.ref_lon = lon_sum / len(self.image_list)
        self.ref_lat = lat_sum / len(self.image_list)
        self.render.setRefCoord(self.ref_lon, self.ref_lat)
        print "Reference: lon = %.6f lat = %.6f" % (self.ref_lon, self.ref_lat)
        for image in self.image_list:
            (x, y) = ImageList.wgs842cart(image.aircraft_lon,
                                          image.aircraft_lat,
                                          self.ref_lon, self.ref_lat)
            image.aircraft_x = x
            image.aircraft_y = y
            image.save_info()


    # undistort x, y using a simple radial lens distortion model.  (We
    # call the original image values the 'distorted' values.)  Input
    # x,y are expected to be normalize (0.0 - 1.0) in image pixel
    # space with 0.5 being the center of image (and hopefully the
    # center of distortion.)
    def doLensUndistort(self, aspect_ratio, xnorm, ynorm):
        xd = (xnorm * 2.0 - 1.0) * aspect_ratio
        yd = ynorm * 2.0 - 1.0
        r = math.sqrt(xd*xd + yd*yd)
        #print "ar=%.3f xd=%.3f yd=%.3f r=%.2f" % (aspect_ratio, xd, yd, r)
        factor = 1.0 + self.k1 * r*r + self.k2 * r*r*r*r
        xu = xd * factor
        yu = yd * factor
        xnorm_u = (xu / aspect_ratio + 1.0) / 2.0
        ynorm_u = (yu + 1.0) / 2.0
        #print "  (%.3f %.3f) -> (%.3f %.3f)" % (xnorm, ynorm, xnorm_u, ynorm_u)
        return xnorm_u, ynorm_u

    def projectPoint2(self, image, q, pt, z_m, horiz_mm, vert_mm, focal_len_mm):
        h = image.fullh
        w = image.fullw
        ar = float(w)/float(h)  # aspect ratio

        # normalized pixel coordinates to [0.0, 1.0]
        xnorm = pt[0] / float(w-1)
        ynorm = pt[1] / float(h-1)
        #print "norm = %.4f %.4f" % (xnorm, ynorm)

        # lens un-distortion
        xnorm_u, ynorm_u = self.doLensUndistort(ar, xnorm, ynorm)

        # compute pixel coordinate in sensor coordinate space (mm
        # units) with (0mm, 0mm) being the center of the image.
        x_mm = (xnorm_u * 2.0 - 1.0) * (horiz_mm * 0.5)
        y_mm = -1.0 * (ynorm_u * 2.0 - 1.0) * (vert_mm * 0.5)

        # the forward vector (out the nose when the aircraft is
        # straight, level, and flying north) is (x=1.0, y=0.0, z=0.0).
        # This vector will get projected to the camera center point,
        # thus we have to remap the axes.
        #camvec = [y_mm, x_mm, focal_len_mm]
        camvec = [focal_len_mm, x_mm, y_mm]
        camvec = transformations.unit_vector(camvec) # normalize
        #print "%.3f %.3f %.3f" % (camvec[0], camvec[1], camvec[2])

        # transform camera vector (in body reference frame) to ned
        # reference frame
        ned = transformations.quaternion_backTransform(q, camvec)
        #print "ned = %s" % str(ned)
        
        # solve projection
        if ned[2] >= 0.0:
            # no interseciton
            return (0.0, 0.0)
        factor = z_m / -ned[2]
        #print "z_m = %s" % str(z_m)
        x_proj = ned[0] * factor
        y_proj = ned[1] * factor
        #print "proj dist = %.2f" % math.sqrt(x_proj*x_proj + y_proj*y_proj)
        return [x_proj, y_proj]

    # returns a quaternion representing the rotation from ned
    # coordinates to camera pos coordinates.  You can back translate a
    # vector against this quaternion to project a camera vector into
    # ned space.
    #
    # body angles represent the aircraft orientation
    # camera angles represent the fixed mounting offset of the camera
    # relative to the body
    # +x axis = forward, +roll = left wing down
    # +y axis = right, +pitch = nose down
    # +z axis = up, +yaw = nose right    
    def computeNed2Cam(self,
                       body_yaw, body_pitch, body_roll,
                       camera_yaw, camera_pitch, camera_roll):
        deg2rad = math.pi / 180.0
        ned2body = transformations.quaternion_from_euler(body_yaw * deg2rad,
                                                         body_pitch * deg2rad,
                                                         body_roll * deg2rad,
                                                         'rzyx')
        body2cam = transformations.quaternion_from_euler(camera_yaw * deg2rad,
                                                         camera_pitch * deg2rad,
                                                         camera_roll * deg2rad,
                                                         'rzyx')
        ned2cam = transformations.quaternion_multiply(ned2body, body2cam)
        return ned2cam

    # project keypoints based on body reference system + body biases
    # transformed by camera mounting + camera mounting biases
    def projectImageKeypointsNative2(self, image, yaw_bias=0.0,
                                     roll_bias=0.0, pitch_bias=0.0,
                                     alt_bias=0.0):
        if image.img == None:
            image.load_rgb()
        h = image.fullh
        w = image.fullw
        ar = float(w)/float(h)  # aspect ratio

        pose = self.computeCameraPoseFromAircraft(image)
        #print "Computed new image pose for %s = %s" % (image.name, str(pose))

        # save the computed camera pose
        image.camera_yaw = pose[0]
        image.camera_pitch = pose[1]
        image.camera_roll = pose[2]
        image.camera_x = pose[3]
        image.camera_y = pose[4]
        image.camera_z = pose[5]
        image.save_info()

        (coord_list, corner_list, grid_list) = \
            self.projectImageKeypointsNative3(image, pose, yaw_bias, roll_bias,
                                              pitch_bias, alt_bias)

        return coord_list, corner_list, grid_list

    # project keypoints using the provided camera pose 
    # pose = (yaw_deg, pitch_deg, roll_deg, x_m, y_m, z_m)
    def projectImageKeypointsNative3(self, image, pose,
                                     yaw_bias=0.0, roll_bias=0.0,
                                     pitch_bias=0.0, alt_bias=0.0,
                                     all_keypoints=False):
        #print "Project3 for %s" % image.name
        if image.img == None:
            image.load_rgb()
        h = image.fullh
        w = image.fullw
        ar = float(w)/float(h)  # aspect ratio

        deg2rad = math.pi / 180.0
        ned2cam = transformations.quaternion_from_euler((pose[0]+yaw_bias)*deg2rad,
                                                        (pose[1]+pitch_bias)*deg2rad,
                                                        (pose[2]+roll_bias)*deg2rad,
                                                        'rzyx')
        x_m = pose[3]
        y_m = pose[4]
        z_m = pose[5] + alt_bias
        #print "ref offset = %.2f %.2f" % (x_m, y_m)

        coord_list = [None] * len(image.kp_list)
        corner_list = []
        grid_list = []

        # project the paired keypoints into world space
        for i, kp in enumerate(image.kp_list):
            if not all_keypoints and not image.kp_usage[i]:
                continue
            # print "ned2cam = %s" % str(ned2cam)
            proj = self.projectPoint2(image, ned2cam, kp.pt, z_m,
                                      self.horiz_mm, self.vert_mm,
                                      self.focal_len_mm)
            #print "project3: kp=%s proj=%s" %(str(kp.pt), str(proj))
            coord_list[i] = [proj[1] + x_m, proj[0] + y_m]
        #print "coord_list = %s" % str(coord_list)

        # compute the corners (2x2 polygon grid) in image space
        dx = image.fullw - 1
        dy = image.fullh - 1
        y = 0.0
        for j in xrange(2):
            x = 0.0
            for i in xrange(2):
                #print "corner %.2f %.2f" % (x, y)
                proj = self.projectPoint2(image, ned2cam, [x, y], z_m,
                                          self.horiz_mm, self.vert_mm,
                                          self.focal_len_mm)
                corner_list.append( [proj[1] + x_m, proj[0] + y_m] )
                x += dx
            y += dy

        # compute the ac3d polygon grid in image space
        dx = image.fullw / float(self.ac3d_steps)
        dy = image.fullh / float(self.ac3d_steps)
        y = 0.0
        for j in xrange(self.ac3d_steps+1):
            x = 0.0
            for i in xrange(self.ac3d_steps+1):
                #print "grid %.2f %.2f" % (xnorm_u, ynorm_u)
                proj = self.projectPoint2(image, ned2cam, [x, y], z_m,
                                          self.horiz_mm, self.vert_mm,
                                          self.focal_len_mm)
                grid_list.append( [proj[1] + x_m, proj[0] + y_m] )
                x += dx
            y += dy

        return coord_list, corner_list, grid_list

    def projectKeypoints(self, all_keypoints=False):
        for image in self.image_list:
            pose = (image.camera_yaw, image.camera_pitch, image.camera_roll,
                    image.camera_x, image.camera_y, image.camera_z)
            # print "project from pose = %s" % str(pose)
            coord_list, corner_list, grid_list \
                = self.projectImageKeypointsNative3(image, pose,
                                                    all_keypoints=all_keypoints)
            image.coord_list = coord_list
            image.corner_list = corner_list
            image.grid_list = grid_list
            # test
            # coord_list, corner_list, grid_list \
            #    = self.projectImageKeypointsNative2(image)
            #print "orig corners = %s" % str(image.corner_list)
            #print "new corners = %s" % str(corner_list)

    def findImageRotate(self, i1, gain):
        #self.findImageAffine(i1) # temp test
        error_sum = 0.0
        weight_sum = i1.weight  # give ourselves an appropriate weight
        for i, match in enumerate(i1.match_list):
            if len(match) >= self.m.min_pairs:
                i2 = self.image_list[i]
                print "Rotating %s vs %s" % (i1.name, i2.name)
                for pair in match:
                    # + 180 (camera is mounted backwards)
                    y1 = i1.yaw + i1.rotate + 180.0
                    y2 = i2.yaw + i2.rotate + 180.0
                    dy = y2 - y1
                    while dy < -180.0:
                        dy += 360.0;
                    while dy > 180.0:
                        dy -= 360.0

                    # angle is in opposite direction from yaw
                    #a1 = i1.yaw + i1.rotate + 180 + i1.kp_list[pair[0]].angle
                    #a2 = i2.yaw + i2.rotate + 180 + i2.kp_list[pair[1]].angle
                    a1 = i1.kp_list[pair[0]].angle
                    a2 = i2.kp_list[pair[1]].angle
                    da = a1 - a2
                    while da < -180.0:
                        da += 360.0;
                    while da > 180.0:
                        da -= 360.0
                    print "yaw diff = %.1f  angle diff = %.1f" % (dy, da)

                    error = dy - da
                    while error < -180.0:
                        error += 360.0;
                    while error > 180.0:
                        error -= 360.0

                    error_sum += error * i2.weight
                    weight_sum += i2.weight
                    print str(pair)
                    print " i1: %.1f %.3f %.1f" % (i1.yaw, i1.kp_list[pair[0]].angle, a1)
                    print " i2: %.1f %.3f %.1f" % (i2.yaw, i2.kp_list[pair[1]].angle, a2)
                    print " error: %.1f  weight: %.2f" % (error, i2.weight)
                    print
                #self.showMatch(i1, i2, match)
        update = 0.0
        if weight_sum > 0.0:
            update = error_sum / weight_sum
        i1.rotate += update * gain
        print "Rotate %s delta=%.2f = %.2f" % (i1.name,  update, i1.rotate)

    def rotateImages(self, gain=0.10):
        for image in self.image_list:
            self.findImageRotate(image, gain)
        for image in self.image_list:
            print "%s: yaw error = %.2f" % (image.name, image.rotate)
                    
    def findImagePairShift(self, i1, i2, match):
        xerror_sum = 0.0
        yerror_sum = 0.0
        for pair in match:
            c1 = i1.coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            xerror_sum += dx
            yerror_sum += dy
        # divide by pairs + 1 gives some weight to our own position
        # (i.e. a zero rotate)
        xshift = xerror_sum / len(match)
        yshift = yerror_sum / len(match)
        #print " %s -> %s = (%.2f %.2f)" % (i1.name, i2.name, xshift, yshift)
        return (xshift, yshift)

    def findImageShift(self, i1, gain=0.10, placing=False):
        xerror_sum = 0.0
        yerror_sum = 0.0
        weight_sum = i1.weight  # give ourselves an appropriate weight
        for i, match in enumerate(i1.match_list):
            if len(match) < self.m.min_pairs:
                continue
            i2 = self.image_list[i]
            #if not i2.placed:
            #    continue
            (xerror, yerror) = self.findImagePairShift( i1, i2, match )
            xerror_sum += xerror * i2.weight
            yerror_sum += yerror * i2.weight
            weight_sum += i2.weight
        xshift = xerror_sum / weight_sum
        yshift = yerror_sum / weight_sum
        print "Shift %s -> (%.2f %.2f)" % (i1.name, xshift, yshift)
        #print " %s bias before (%.2f %.2f)" % (i1.name, i1.x_bias, i1.y_bias)
        i1.x_bias += xshift * gain
        i1.y_bias += yshift * gain
        #print " %s bias after (%.2f %.2f)" % (i1.name, i1.x_bias, i1.y_bias)
        i1.save_info()

    def shiftImages(self, gain=0.10):
        for image in self.image_list:
            self.findImageShift(image, gain)

    # method="average": return the weighted average of the errors.
    # method="stddev": return the weighted average of the stddev of the errors.
    # method="max": return the max error of the subcomponents.
    def groupError(self, method="average"):
        #print "compute group error, method = %s" % method
        if len(self.image_list):
            error_sum = 0.0
            weight_sum = 0.0
            for i, image in enumerate(self.image_list):
                e = 0.0
                e = self.m.imageError(i, method=method)
                #print "%s error = %.2f" % (image.name, e)
                error_sum += e*e * image.weight
                weight_sum += image.weight
            return math.sqrt(error_sum / weight_sum)
        else:
            return 0.0

    # zero all biases (if we want to start over with a from scratch fit)
    def zeroImageBiases(self):
        for image in self.image_list:
            image.yaw_bias = 0.0
            image.roll_bias = 0.0
            image.pitch_bias = 0.0
            image.alt_bias = 0.0
            image.x_bias = 0.0
            image.y_bias = 0.0
            image.save_info()

    # try to fit individual images by manipulating various parameters
    # and testing to see if that produces a better fit metric
    def estimateParameter(self, i, ground_alt_m, method,
                          param="", start_value=0.0, step_size=1.0,
                          refinements=3):
        image = self.image_list[i]

        pose = (image.camera_yaw, image.camera_pitch, image.camera_roll,
                image.camera_x, image.camera_y, image.camera_z)

        #print "Estimate %s for %s" % (param, image.name)
        var = False
        if method == "average":
            var = False
        elif method == "stddev":
            var = True
        for k in xrange(refinements):
            best_error = self.m.imageError(i, method=method)
            best_value = start_value
            test_value = start_value - 5*step_size
            #print "start value = %.2f error = %.1f" % (best_value, best_error)

            while test_value <= start_value + 5*step_size + (step_size*0.1):
                coord_list = []
                corner_list = []
                grid_list = []
                if param == "yaw":
                    coord_list, corner_list, grid_list \
                        = self.projectImageKeypointsNative3(image, pose,
                                                            yaw_bias=test_value)
                elif param == "roll":
                    coord_list, corner_list, grid_list \
                        = self.projectImageKeypointsNative3(image, pose,
                                                            roll_bias=test_value)
                elif param == "pitch":
                    coord_list, corner_list, grid_list \
                        = self.projectImageKeypointsNative3(image, pose,
                                                            pitch_bias=test_value)
                elif param == "altitude":
                    coord_list, corner_list, grid_list \
                        = self.projectImageKeypointsNative3(image, pose,
                                                            alt_bias=test_value)
                error = self.m.imageError(i, alt_coord_list=coord_list,
                                          method=method)
                #print "Test %s error @ %.2f = %.2f" % ( param, test_value, error )
                if error < best_error:
                    best_error = error
                    best_value = test_value
                    #print " better value = %.2f, error = %.1f" % (best_value, best_error)
                test_value += step_size
            # update values for next iteration
            start_value = best_value
            step_size /= 5.0
        return best_value, best_error

    # try to fit individual images by manipulating various parameters
    # and testing to see if that produces a better fit metric
    def fitImage(self, i, method, gain):
        # parameters to manipulate = yaw, roll, pitch
        yaw_step = 2.0
        roll_step = 1.0
        pitch_step = 1.0
        refinements = 4

        image = self.image_list[i]

        # start values should be zero because previous values are
        # already included so we are computing a new offset from the
        # past solution.
        yaw, e = self.estimateParameter(i, self.ground_alt_m, method,
                                        "yaw", start_value=0.0,
                                        step_size=1.0, refinements=refinements)
        roll, e = self.estimateParameter(i, self.ground_alt_m, method,
                                         "roll", start_value=0.0,
                                         step_size=1.0, refinements=refinements)
        pitch, e = self.estimateParameter(i, self.ground_alt_m, method,
                                          "pitch", start_value=0.0,
                                          step_size=1.0,
                                          refinements=refinements)
        alt, e = self.estimateParameter(i, self.ground_alt_m, method,
                                        "altitude", start_value=0.0,
                                        step_size=2.0, refinements=refinements)
        image.camera_yaw += yaw*gain
        image.camera_roll += roll*gain
        image.camera_pitch += pitch*gain
        image.camera_z += alt*gain
        coord_list = []
        corner_list = []
        grid_list = []
        # but don't save the results so we don't bias future elements
        # with moving previous elements
        coord_list, corner_list, grid_list = self.projectImageKeypointsNative2(image)
        error = self.m.imageError(i, alt_coord_list=coord_list, method=method)
        if method == "average":
            image.error = error
        elif method == "stddev":
            image.stddev = error
        print "Fit %s (%s) is %.2f %.2f %.2f %.2f (avg=%.3f stddev=%.3f)" \
            % (image.name, method,
               image.camera_yaw, image.camera_roll, image.camera_pitch,
               image.camera_z, image.error, image.stddev)
        image.save_info()

    # try to fit individual images by manipulating various parameters
    # and testing to see if that produces a better fit metric
    def fitImageAffine3d(self, i, method, gain):
        i1 = self.image_list[i]
        angles_sum = [0.0, 0.0, 0.0]
        weight_sum = i1.weight
        for j, pairs in enumerate(i1.match_list):
            if len(pairs) < self.m.min_pairs:
                continue
            i2 = self.image_list[j]
            src = [[], [], []]
            dst = [[], [], []]
            for pair in pairs:
                c1 = i1.coord_list[pair[0]]
                c2 = i2.coord_list[pair[1]] 
                src[0].append(c1[0])
                src[1].append(c1[1])
                src[2].append(0.0)
                dst[0].append(c2[0])
                dst[1].append(c2[1])
                dst[2].append(0.0)
            Aff3D = transformations.superimposition_matrix(src, dst)
            scale, shear, angles, trans, persp = transformations.decompose_matrix(Aff3D)
            print "%s vs. %s" % (i1.name, i2.name)
            #print "  scale = %s" % str(scale)
            #print "  shear = %s" % str(shear)
            print "  angles = %s" % str(angles)
            #print "  trans = %s" % str(trans)
            #print "  persp = %s" % str(persp)

            # this is all based around the assumption that our angle
            # differences area relatively small
            for k in range(3):
                a = angles[k]
                if a < -180.0:
                    a += 360.0
                if a > 180.0:
                    a -= 360.0
                angles_sum[k] += a
            weight_sum += i2.weight
        angles = [ angles_sum[0] / weight_sum,
                   angles_sum[1] / weight_sum,
                   angles_sum[2] / weight_sum ]
        print "average angles = %s" % str(angles)

        rad2deg = 180.0 / math.pi
        i1.roll_bias += angles[0] * rad2deg * gain
        i1.pitch_bias += angles[1] * rad2deg * gain
        i1.yaw_bias += angles[2] * rad2deg * gain

        coord_list = []
        corner_list = []
        grid_list = []
        # but don't save the results so we don't bias future elements
        # with moving previous elements
        coord_list, corner_list, grid_list = self.projectImageKeypointsNative2(i1)
        error = self.m.imageError(i, alt_coord_list=coord_list, method="average")
        stddev = self.m.imageError(i, alt_coord_list=coord_list, method="stddev")
        print "average error = %.3f" % error
        print "average stddev = %.3f" % stddev
        i1.save_info()

    def fitImagesIndividually(self, method, gain):
        for i, image in enumerate(self.image_list):
            self.fitImage(i, method, gain)
            #self.fitImageAffine3d(i, method, gain)

    def geotag_pictures( self, correlator, dir = ".", geotag_dir = "." ):
        ground_sum = 0.0
        ground_count = 0
        print "master_time_offset = " + str(correlator.master_time_offset)
        for match in correlator.best_matchups:
            pict, trig = correlator.get_match(match)
            trig_time = trig[0] + correlator.master_time_offset
            pict_time = pict[0]

            time_diff = trig_time - pict_time
            #print str(match[0]) + " <=> " + str(match[1])
            #print str(pict_time) + " <=> " + str(trig_time)
            print pict[2] + " -> " + str(trig[2]) + ", " + str(trig[3]) + ": " + str(trig[4]) + " (" + str(time_diff) + ")"
            agl_ft = trig[4]
            lon_deg, lat_deg, msl = correlator.get_position( trig[0] )
            msl_ft = msl / 0.3048
            ground_sum += (msl_ft - agl_ft)
            ground_count += 1
            ground_agl_ft = ground_sum / ground_count
            print "  MSL: " + str( msl_ft ) + " AGL: " + str(agl_ft) + " Ground: " + str(ground_agl_ft)

            # double check geotag dir exists and make it if not
            if not os.path.exists(geotag_dir):
                os.makedirs(geotag_dir)

            # update a resized copy if needed
            name_in = dir + "/" + pict[2]
            name_out = geotag_dir + "/" + pict[2]
            if not os.path.isfile( name_out ):
                command = 'convert -geometry 684x456 ' + name_in + ' ' + name_out
                #command = 'convert -geometry 512x512\! ' + name_in + ' ' + name_out
                print command
                commands.getstatusoutput( command )

            # update the gps meta data
            exif = pyexiv2.ImageMetadata(name_out)
            exif.read()
            #exif.set_gps_info(lat_deg, lon_deg, (msl_ft*0.3048))
            altitude = msl_ft*0.3048
            GPS = 'Exif.GPSInfo.GPS'
            exif[GPS + 'AltitudeRef']  = '0' if altitude >= 0 else '1'
            exif[GPS + 'Altitude']     = Fraction(altitude)
            exif[GPS + 'Latitude']     = decimal_to_dms(lat_deg)
            exif[GPS + 'LatitudeRef']  = 'N' if lat_deg >= 0 else 'S'
            exif[GPS + 'Longitude']    = decimal_to_dms(lon_deg)
            exif[GPS + 'LongitudeRef'] = 'E' if lon_deg >= 0 else 'W'
            exif[GPS + 'MapDatum']     = 'WGS-84'
            exif.write()

    def fixup_timestamps( self, correlator, camera_time_error, geotag_dir = "." ):
        for match in correlator.best_matchups:
            pict, trig = correlator.get_match(match)
            unixtime = pict[0]
            name = geotag_dir + "/" + pict[2]

            unixtime += camera_time_error
            newdatetime = datetime.datetime.utcfromtimestamp(round(unixtime)).strftime('%Y:%m:%d %H:%M:%S')
            exif = pyexiv2.ImageMetadata(name)
            exif.read()
            print "old: " + str(exif['Exif.Image.DateTime']) + "  new: " + newdatetime
            exif['Exif.Image.DateTime'] = newdatetime
            exif.write()

    def generate_aircraft_location_report(self):
        for image in self.image_list:
            print "%s\t%.10f\t%.10f\t%.2f" \
                % (image.name, image.aircraft_lon, image.aircraft_lat,
                   image.aircraft_msl)

    def generate_ac3d(self, ref_image = False, base_name="quick", version=None ):
        max_roll = 30.0
        max_pitch = 30.0
        min_agl = 50.0
        min_time = 0.0 # the further into the flight hopefully the better the filter convergence

        ref_lon = None
        ref_lat = None

        # count matching images (starting with 1 to include the reference image)
        match_count = 0
        if ref_image:
            match_count += 1
        for image in self.image_list:
            msl = image.aircraft_msl
            roll = -image.aircraft_roll
            pitch = -image.aircraft_pitch
            agl = msl - self.ground_alt_m
            if image.num_matches >= 0 and math.fabs(roll) <= max_roll and math.fabs(pitch) <= max_pitch and agl >= min_agl:
                print image.name
                match_count += 1

        # write AC3D header
        name = self.image_dir
        name += "/"
        name += base_name
        if version:
            name += ("-%02d" % version)
        name += ".ac"
        f = open( name, "w" )
        f.write("AC3Db\n")
        f.write("MATERIAL \"\" rgb 1 1 1  amb 0.6 0.6 0.6  emis 0 0 0  spec 0.5 0.5 0.5  shi 10  trans 0.4\n")
        f.write("OBJECT world\n")
        f.write("rot 1.0 0.0 0.0  0.0 0.0 1.0 0.0 1.0 0.0")
        f.write("kids " + str(match_count) + "\n")

        for image in self.image_list:
            msl = image.aircraft_msl
            roll = -image.aircraft_roll
            pitch = -image.aircraft_pitch
            agl = msl - self.ground_alt_m
            if not image.num_matches >= 0 or math.fabs(roll) > max_roll or math.fabs(pitch) > max_pitch or agl < min_agl:
                continue

            # compute a priority function (higher priority tiles are raised up)
            priority = (1.0-image.weight) - agl/400.0

            #ll = list(image.grid_list[0])
            #ll.append( -priority )
            #lr = list(image.grid_list[1])
            #lr.append( -priority )
            #ur = list(image.grid_list[2])
            #ur.append( -priority )
            #ul = list(image.grid_list[3])
            #ul.append( -priority )

            f.write("OBJECT poly\n")
            f.write("name \"rect\"\n")
            f.write("texture \"./" + image.name + "\"\n")
            f.write("loc 0 0 0\n")

            f.write("numvert %d\n" % ((self.ac3d_steps+1) * (self.ac3d_steps+1)))
            # output the ac3d polygon grid (note the grid list is in
            # this specific order because that is how we generated it
            # earlier
            pos = 0
            for j in xrange(self.ac3d_steps+1):
                for i in xrange(self.ac3d_steps+1):
                    v = image.grid_list[pos]
                    f.write( "%.3f %.3f %.3f\n" % (v[0], v[1], priority) )
                    pos += 1
  
            f.write("numsurf %d\n" % (self.ac3d_steps * self.ac3d_steps))
            dx = 1.0 / float(self.ac3d_steps)
            dy = 1.0 / float(self.ac3d_steps)
            y = 1.0
            for j in xrange(self.ac3d_steps):
                x = 0.0
                for i in xrange(self.ac3d_steps):
                    c = (j * (self.ac3d_steps+1)) + i
                    d = ((j+1) * (self.ac3d_steps+1)) + i
                    f.write("SURF 0x20\n")
                    f.write("mat 0\n")
                    f.write("refs 4\n")
                    f.write("%d %.3f %.3f\n" % (d, x, y-dy))
                    f.write("%d %.3f %.3f\n" % (d+1, x+dx, y-dy))
                    f.write("%d %.3f %.3f\n" % (c+1, x+dx, y))
                    f.write("%d %.3f %.3f\n" % (c, x, y))
                    x += dx
                y -= dy
            f.write("kids 0\n")

        if ref_image:
            # reference poly
            f.write("OBJECT poly\n")
            f.write("name \"rect\"\n")
            f.write("texture \"Reference/3drc.png\"\n")
            f.write("loc 0 0 0\n")
            f.write("numvert 4\n")

            f.write(str(gul[0]) + " " + str(gul[1]) + " " + str(gul[2]-15) + "\n")
            f.write(str(gur[0]) + " " + str(gur[1]) + " " + str(gur[2]-15) + "\n")
            f.write(str(glr[0]) + " " + str(glr[1]) + " " + str(glr[2]-15) + "\n")
            f.write(str(gll[0]) + " " + str(gll[1]) + " " + str(gll[2]-15) + "\n")
            f.write("numsurf 1\n")
            f.write("SURF 0x20\n")
            f.write("mat 0\n")
            f.write("refs 4\n")
            f.write("3 0 0\n")
            f.write("2 1 0\n")
            f.write("1 1 1\n")
            f.write("0 0 1\n")
            f.write("kids 0\n")

        f.close()

    def draw_epilines(self, img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r,c,d = img1.shape
        print img1.shape
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            cv2.line(img1, (x0,y0), (x1,y1), color,1)
            cv2.circle(img1,tuple(pt1),5,color,-1)
            cv2.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2

    def sfm_test(self):
        for i, i1 in enumerate(self.image_list):
            for j, pairs in enumerate(i1.match_list):
                if i == j:
                    continue
                if len(pairs) < 8:
                    # 8+ pairs are required to compute the fundamental matrix
                    continue
                i2 = self.image_list[j]
                pts1 = []
                pts2 = []
                for pair in pairs:
                    p1 = i1.kp_list[pair[0]].pt
                    p2 = i2.kp_list[pair[1]].pt
                    pts1.append( p1 )
                    pts2.append( p2 )
                pts1 = np.float32(pts1)
                pts2 = np.float32(pts2)
                print "pts1 = %s" % str(pts1)
                print "pts2 = %s" % str(pts2)
                F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

                print "loading full res images ..."
                img1 = i1.load_source_rgb(self.source_dir)
                img2 = i2.load_source_rgb(self.source_dir)

                # Find epilines corresponding to points in right image
                # (second image) and drawing its lines on left image
                lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
                lines1 = lines1.reshape(-1,3)
                img5,img6 = self.draw_epilines(img1,img2,lines1,pts1,pts2)

                # Find epilines corresponding to points in left image (first image) and
                # drawing its lines on right image
                lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
                lines2 = lines2.reshape(-1,3)
                img3,img4 = self.draw_epilines(img2,img1,lines2,pts2,pts1)

                plt.subplot(121),plt.imshow(img5)
                plt.subplot(122),plt.imshow(img3)
                plt.show()

    # this really doesn't work right because the euler pose angles derived
    # might be correct, but aren't all consistent apparently ... the back
    # solver to extract angles from an arbitrary rotation matrix doesn't seem
    # always be consistant. (this probably should be depricated at some point)
    def fitImagesWithSolvePnP1(self):
        for i, i1 in enumerate(self.image_list):
            #print "sovelPNP() for %s" % i1.name
            fx = (self.focal_len_mm/self.horiz_mm) * float(i1.fullw)
            fy = (self.focal_len_mm/self.vert_mm) * float(i1.fullh)
            cx = i1.fullw * 0.5
            cy = i1.fullh * 0.5
            cam = np.array( [ [fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0] ] )
            att_sum = [ [0.0, 0.0], [0.0, 0.0], [0.0, 0.0] ]
            pos_sum = [0.0, 0.0, 0.0]
            weight_sum = 0.0
            for j, pairs in enumerate(i1.match_list):
                if i == j:
                    continue
                if len(pairs) < 8:
                    # we need at least 8 pairs to call solvePNP()
                    continue
                i2 = self.image_list[j]
                img_pts = []
                obj_pts = []
                for pair in pairs:
                    p1 = i1.kp_list[pair[0]].pt
                    p2 = i2.coord_list[pair[1]]
                    img_pts.append( p1 )
                    obj_pts.append( [p2[0], p2[1], 0.0] )
                img_pts = np.float32(img_pts)
                obj_pts = np.float32(obj_pts)
                #print "img_pts = %s" % str(img_pts)
                #print "obj_pts = %s" % str(obj_pts)
                (result, rvec, tvec) = cv2.solvePnP(obj_pts, img_pts, cam, None)
                #print "  result = %s, rvec = %s, tvec = %s" \
                #    % (result, rvec, tvec)
                print "  rvec = %.2f %.2f %.2f" % (rvec[0], rvec[1], rvec[2])
                R, jac = cv2.Rodrigues(rvec)
                #print "  R =\n%s" % str(R)
                # googled how to derive the position in object
                # coordinates from solvePNP()
                pos = -np.matrix(R).T * np.matrix(tvec)
                #print "solved pos = %s" % str(pos)
                #print "  pos[0] = %s" % str(pos[0])
                #print "  pos.item(0) = %s" % str(pos.item(0))
                for k in range(0,3):
                    pos_sum[k] += pos.item(k)
                #print "  PNP pos = %s" % str(pos)

                # Remap the R matrix to match our coordinate system
                # (by inspection...)
                # [ [a, b, c], [d,  e,  f], [g, h,  i] ] =>
                # [ [h, b, a], [g, -e, -d], [i, c, -f] ]
                # this will be tedius code ...
                #Rconv[:3, 0] = R[:3, 2] # swap col 0 <=> col 2
                #Rconv[:3, 2] = R[:3, 0]
                #Rconv[1, :3] *= -1.0    # negate the middle row
                Rconv = R.copy()
                Rconv[0,0] = R[2,1]
                Rconv[0,1] = R[0,1]
                Rconv[0,2] = R[0,0]
                Rconv[1,0] = R[2,0]
                Rconv[1,1] = -R[1,1]
                Rconv[1,2] = -R[1,0]
                Rconv[2,0] = R[2,2]
                Rconv[2,1] = R[0,2]
                Rconv[2,2] = -R[1,2]
                #print "Rconv =\n%s" % str(Rconv)
                (yaw, pitch, roll) = transformations.euler_from_matrix(Rconv, 'rzyx')
                att_sum[0][0] += math.cos(yaw)
                att_sum[0][1] += math.sin(yaw)
                att_sum[1][0] += math.cos(pitch)
                att_sum[1][1] += math.sin(pitch)
                att_sum[2][0] += math.cos(roll)
                att_sum[2][1] += math.sin(roll)
                weight_sum += i2.weight

                deg2rad = math.pi / 180.0
                print "  pair euler = %.2f %.2f %.2f" % (yaw/deg2rad,
                                                         pitch/deg2rad,
                                                         roll/deg2rad)
                #print "  est = %.2f %.2f %.2f" % (i1.camera_yaw,
                #                                  i1.camera_pitch,
                #                                  i1.camera_roll)

                Rcam = transformations.euler_matrix(i1.camera_yaw*deg2rad,
                                                    i1.camera_pitch*deg2rad,
                                                    i1.camera_roll*deg2rad,
                                                    'rzyx')

                #print "solvePNP =\n%s" % str(Rconv)
                #print "my FIT =\n%s" % str(Rcam)

                v = np.array( [1.0, 0.0, 0.0] )
                vh = np.array( [1.0, 0.0, 0.0, 1.0] )
                #print "  v = %s" % str(v)
                #print "  Rconv * v = %s" % str(np.dot(Rconv, v))
                #print "  Rcam * v = %s" % str(np.dot(Rcam, vh))
            if weight_sum < 0.0001:
                continue
            i1.camera_x = pos_sum[0] / weight_sum
            i1.camera_y = pos_sum[1] / weight_sum
            i1.camera_z = pos_sum[2] / weight_sum
            print "Camera pose for image %s:" % i1.name
            print "  PNP pos = %.2f %.2f %.2f" % (i1.camera_x,
                                                  i1.camera_y,
                                                  i1.camera_z)
            yaw_avg = math.atan2(att_sum[0][1]/weight_sum,
                                 att_sum[0][0]/weight_sum)
            pitch_avg = math.atan2(att_sum[1][1]/weight_sum,
                                   att_sum[1][0]/weight_sum)
            roll_avg =  math.atan2(att_sum[2][1]/weight_sum,
                                   att_sum[2][0]/weight_sum)
            i1.camera_yaw = yaw_avg / deg2rad
            i1.camera_pitch = pitch_avg / deg2rad
            i1.camera_roll = roll_avg / deg2rad
            print "  PNP att = %.2f %.2f %.2f" % (i1.camera_yaw,
                                                  i1.camera_pitch,
                                                  i1.camera_roll)
            i1.save_info()

    # call solvePnP() on all the matching pairs from all the matching
    # images simultaneously.  This works, but inherently weights the
    # fit much more towards the images with more matching pairs ... on
    # the other hand, that may be kind of what we want because images
    # with a few matches over a small area can grossly magnify any
    # errors into the result of solvePnP().
    def fitImagesWithSolvePnP2(self):
        for i, i1 in enumerate(self.image_list):
            #print "sovlePNP() for %s" % i1.name
            fx = (self.focal_len_mm/self.horiz_mm) * float(i1.fullw)
            fy = (self.focal_len_mm/self.vert_mm) * float(i1.fullh)
            cx = i1.fullw * 0.5
            cy = i1.fullh * 0.5
            cam = np.array( [ [fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0] ] )
            img_pts = []
            obj_pts = []
            for j, pairs in enumerate(i1.match_list):
                if i == j:
                    # include the match with ourselves ... we have
                    # self worth too!
                    for k, flag in enumerate(i1.kp_usage):
                        if flag:
                            p1 = i1.kp_list[k].pt
                            p2 = i1.coord_list[k]
                            img_pts.append( p1 )
                            obj_pts.append( [p2[0], p2[1], 0.0] )
                if len(pairs) < 8:
                    # we need at least 8 pairs to call solvePNP()
                    continue
                i2 = self.image_list[j]
                for pair in pairs:
                    p1 = i1.kp_list[pair[0]].pt
                    p2 = i2.coord_list[pair[1]]
                    img_pts.append( p1 )
                    obj_pts.append( [p2[0], p2[1], 0.0] )
            # now call the solver if we have enough points
            if len(img_pts) < 8:
                continue
            img_pts = np.float32(img_pts)
            obj_pts = np.float32(obj_pts)
            #print "img_pts = %s" % str(img_pts)
            #print "obj_pts = %s" % str(obj_pts)

            #(result, rvec, tvec) = cv2.solvePnP(obj_pts, img_pts, cam, None)
            if hasattr(i1, 'rvec'):
                (result, i1.rvec, i1.tvec) \
                    = cv2.solvePnP(obj_pts, img_pts, cam, None,
                                         i1.rvec, i1.tvec,
                                         useExtrinsicGuess=True)
            else:
                # first time
                (result, i1.rvec, i1.tvec) \
                    = cv2.solvePnP(obj_pts, img_pts, cam, None)

            #print "  result = %s, rvec = %s, tvec = %s" \
            #    % (result, i1.rvec, i1.tvec)
            # print "  rvec = %.2f %.2f %.2f" % (i1.rvec[0], i1.rvec[1], i1.rvec[2])
            R, jac = cv2.Rodrigues(i1.rvec)
            #print "  R =\n%s" % str(R)
            # googled how to derive the position in object
            # coordinates from solvePNP()
            pos = -np.matrix(R).T * np.matrix(i1.tvec)
            #print "solved pos = %s" % str(pos)
            #print "  pos[0] = %s" % str(pos[0])
            #print "  pos.item(0) = %s" % str(pos.item(0))
            #print "  PNP pos = %s" % str(pos)

            # Remap the R matrix to match our coordinate system
            # (by inspection...)
            # [ [a, b, c], [d,  e,  f], [g, h,  i] ] =>
            # [ [h, b, a], [g, -e, -d], [i, c, -f] ]
            # this will be tedius code ...
            #Rconv[:3, 0] = R[:3, 2] # swap col 0 <=> col 2
            #Rconv[:3, 2] = R[:3, 0]
            #Rconv[1, :3] *= -1.0    # negate the middle row
            Rconv = R.copy()
            Rconv[0,0] = R[2,1]
            Rconv[0,1] = R[0,1]
            Rconv[0,2] = R[0,0]
            Rconv[1,0] = R[2,0]
            Rconv[1,1] = -R[1,1]
            Rconv[1,2] = -R[1,0]
            Rconv[2,0] = R[2,2]
            Rconv[2,1] = R[0,2]
            Rconv[2,2] = -R[1,2]
            #print "Rconv =\n%s" % str(Rconv)
            (yaw, pitch, roll) = transformations.euler_from_matrix(Rconv, 'rzyx')
            deg2rad = math.pi / 180.0
            #print "  pair euler = %.2f %.2f %.2f" % (yaw/deg2rad,
            #                                         pitch/deg2rad,
            #                                         roll/deg2rad)
            #print "  est = %.2f %.2f %.2f" % (i1.camera_yaw,
            #                                  i1.camera_pitch,
            #                                  i1.camera_roll)

            Rcam = transformations.euler_matrix(i1.camera_yaw*deg2rad,
                                                i1.camera_pitch*deg2rad,
                                                i1.camera_roll*deg2rad,
                                                'rzyx')

            print "Beg cam pose %s %.2f %.2f %.2f  %.2f %.2f %.2f" \
                % (i1.name, i1.camera_yaw, i1.camera_pitch, i1.camera_roll,
                   i1.camera_x, i1.camera_y, i1.camera_z)
            i1.camera_yaw = yaw/deg2rad
            i1.camera_pitch = pitch/deg2rad
            i1.camera_roll = roll/deg2rad
            i1.camera_x = pos.item(0)
            i1.camera_y = pos.item(1)
            i1.camera_z = pos.item(2)
            i1.save_info()
            print "New cam pose %s %.2f %.2f %.2f  %.2f %.2f %.2f" \
                % (i1.name, i1.camera_yaw, i1.camera_pitch, i1.camera_roll,
                   i1.camera_x, i1.camera_y, i1.camera_z)

    # find the pose estimate for each match individually and use that
    # pose to project the keypoints.  Then average all the keypoint
    # projections together ... this weights image pairs equally and
    # averaging points in cartesian space is much easier than trying
    # to figure out how to average euler angles.
    #
    # Problem ... too many pairwise matches are unstable for
    # solvePnP() because of clustered or linear data leading to a
    # whole lot of nonsense
    def fitImagesWithSolvePnP3(self):
        for i, i1 in enumerate(self.image_list):
            print "solvePnP() (3) for %s" % i1.name

            if i1.connections == 0:
                print "  ... no connections, skipping ..."
                continue

            fx = (self.focal_len_mm/self.horiz_mm) * float(i1.fullw)
            fy = (self.focal_len_mm/self.vert_mm) * float(i1.fullh)
            cx = i1.fullw * 0.5
            cy = i1.fullh * 0.5
            cam = np.array( [ [fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0] ] )

            master_list = []
            master_list.append(i1.coord_list) # weight ourselves in the mix

            for j, pairs in enumerate(i1.match_list):
                # include the match with ourselves ... we have self worth too!
                #if i == j:
                #    continue
                if len(pairs) < 8:
                    # we need at least 8 pairs to call solvePNP()
                    continue
                i2 = self.image_list[j]

                # assemble the data points for the solver
                img_pts = []
                obj_pts = []
                for pair in pairs:
                    p1 = i1.kp_list[pair[0]].pt
                    p2 = i2.coord_list[pair[1]]
                    img_pts.append( p1 )
                    obj_pts.append( [p2[0], p2[1], 0.0] )

                # now call the solver
                img_pts = np.float32(img_pts)
                obj_pts = np.float32(obj_pts)
                #(result, rvec, tvec) \
                #    = cv2.solvePnP(obj_pts, img_pts, cam, None)
                (rvec, tvec, status) \
                    = cv2.solvePnPRansac(obj_pts, img_pts, cam, None)
                size = len(status)
                inliers = np.sum(status)
                if inliers < size: 
                    print '%s vs %s: %d / %d  inliers/matched' \
                        % (i1.name, i2.name, inliers, size)
                    status = self.m.showMatch(i1, i2, matches, status)
                    delete_list = []
                    for k, flag in enumerate(status):
                        if not flag:
                            print "    deleting: " + str(matches[k])
                            #match[i] = (-1, -1)
                            delete_list.append(matches[k])
                    for pair in delete_list:
                        self.deletePair(i, j, pair)

                #print "  result = %s, rvec = %s, tvec = %s" \
                #    % (result, rvec, tvec)
                # print "  rvec = %.2f %.2f %.2f" % (rvec[0], rvec[1], rvec[2])
                R, jac = cv2.Rodrigues(rvec)
                #print "  R =\n%s" % str(R)
                # googled how to derive the position in object
                # coordinates from solvePNP()
                pos = -np.matrix(R).T * np.matrix(tvec)
                #print "solved pos = %s" % str(pos)
                #print "  pos[0] = %s" % str(pos[0])
                #print "  pos.item(0) = %s" % str(pos.item(0))
                #print "  PNP pos = %s" % str(pos)

                # Remap the R matrix to match our coordinate system
                # (by inspection...)
                # [ [a, b, c], [d,  e,  f], [g, h,  i] ] =>
                # [ [h, b, a], [g, -e, -d], [i, c, -f] ]
                Rconv = R.copy()
                Rconv[0,0] = R[2,1]
                Rconv[0,1] = R[0,1]
                Rconv[0,2] = R[0,0]
                Rconv[1,0] = R[2,0]
                Rconv[1,1] = -R[1,1]
                Rconv[1,2] = -R[1,0]
                Rconv[2,0] = R[2,2]
                Rconv[2,1] = R[0,2]
                Rconv[2,2] = -R[1,2]
                #print "Rconv =\n%s" % str(Rconv)
                (yaw, pitch, roll) = transformations.euler_from_matrix(Rconv,
                                                                       'rzyx')
                deg2rad = math.pi / 180.0
                camera_pose = (yaw/deg2rad, pitch/deg2rad, roll/deg2rad,
                               pos.item(0), pos.item(1), pos.item(2))

                # project out the image keypoints for this pair's
                # estimated camera pose
                coord_list, corner_list, grid_list \
                    = self.projectImageKeypointsNative3(i1, camera_pose)
                #print "len(coord_list) = %d" % len(coord_list)

                # save the results for averaging purposes
                master_list.append(coord_list)

                #print "  pair euler = %.2f %.2f %.2f" % (yaw/deg2rad,
                #                                         pitch/deg2rad,
                #                                         roll/deg2rad)
                #print "  est = %.2f %.2f %.2f" % (i1.camera_yaw,
                #                                  i1.camera_pitch,
                #                                  i1.camera_roll)

                #Rcam = transformations.euler_matrix(i1.camera_yaw*deg2rad,
                #                                    i1.camera_pitch*deg2rad,
                #                                    i1.camera_roll*deg2rad,
                #                                    'rzyx')

                #print "solvePNP =\n%s" % str(Rconv)
                #print "my FIT =\n%s" % str(Rcam)

                print " %s vs %s cam pose %.2f %.2f %.2f  %.2f %.2f %.2f" \
                    % (i1.name, i2.name,
                       camera_pose[0], camera_pose[1], camera_pose[2],
                       camera_pose[3], camera_pose[4], camera_pose[5])
    
            # find the average coordinate locations from the set of pair
            # projections
            coord_list = []
            size = len(master_list[0]) # number of coordinates
            #print "size = %d" % size
            n = len(master_list)       # number of projections
            #print "n = %d" % n
            for i in range(0, size):
                #print "i = %d" % i
                if not i1.kp_usage[i]:
                    coord_list.append(None)
                    continue
                x_sum = 0.0
                y_sum = 0.0
                for list in master_list:
                    #print "len(list) = %d" % len(list)
                    x_sum += list[i][0]
                    y_sum += list[i][1]
                x = x_sum / float(n)
                y = y_sum / float(n)
                coord_list.append( [x, y] )

            # now finally call solvePnP() on the average of the projections
            img_pts = []
            obj_pts = []
            for i in range(0, size):
                if not i1.kp_usage[i]:
                    continue
                img_pts.append( i1.kp_list[i].pt )
                obj_pts.append( [coord_list[i][0], coord_list[i][1], 0.0] )
            img_pts = np.float32(img_pts)
            obj_pts = np.float32(obj_pts)
            (result, rvec, tvec) = cv2.solvePnP(obj_pts, img_pts, cam, None)

            # and extract the average camera pose
            R, jac = cv2.Rodrigues(rvec)
            pos = -np.matrix(R).T * np.matrix(tvec)

            # Remap the R matrix to match our coordinate system
            # (by inspection...)
            # [ [a, b, c], [d,  e,  f], [g, h,  i] ] =>
            # [ [h, b, a], [g, -e, -d], [i, c, -f] ]
            Rconv = R.copy()
            Rconv[0,0] = R[2,1]
            Rconv[0,1] = R[0,1]
            Rconv[0,2] = R[0,0]
            Rconv[1,0] = R[2,0]
            Rconv[1,1] = -R[1,1]
            Rconv[1,2] = -R[1,0]
            Rconv[2,0] = R[2,2]
            Rconv[2,1] = R[0,2]
            Rconv[2,2] = -R[1,2]
            (yaw, pitch, roll) = transformations.euler_from_matrix(Rconv,
                                                                   'rzyx')
            deg2rad = math.pi / 180.0
            
            print "Beg cam pose %s %.2f %.2f %.2f  %.2f %.2f %.2f" \
                % (i1.name, i1.camera_yaw, i1.camera_pitch, i1.camera_roll,
                   i1.camera_x, i1.camera_y, i1.camera_z)
            i1.camera_yaw = yaw/deg2rad
            i1.camera_pitch = pitch/deg2rad
            i1.camera_roll = roll/deg2rad
            i1.camera_x = pos.item(0)
            i1.camera_y = pos.item(1)
            i1.camera_z = pos.item(2)
            i1.save_info()
            print "New cam pose %s %.2f %.2f %.2f  %.2f %.2f %.2f" \
                % (i1.name, i1.camera_yaw, i1.camera_pitch, i1.camera_roll,
                   i1.camera_x, i1.camera_y, i1.camera_z)

    def triangulate_test(self):
        for i, i1 in enumerate(self.image_list):
            print "pnp for %s" % i1.name
            fx = (self.focal_len_mm/self.horiz_mm) * float(i1.fullw)
            fy = (self.focal_len_mm/self.vert_mm) * float(i1.fullh)
            cx = i1.fullw * 0.5
            cy = i1.fullh * 0.5
            cam = np.array( [ [fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0] ] )
            att_sum = [ [0.0, 0.0], [0.0, 0.0], [0.0, 0.0] ]
            pos_sum = [0.0, 0.0, 0.0]
            weight_sum = 0.0
            for j, pairs in enumerate(i1.match_list):
                if i == j:
                    continue
                if len(pairs) < 8:
                    # start with only well correlated pairs
                    continue
                i2 = self.image_list[j]
                R1, jac = cv2.Rodrigues(i1.rvec)
                R2, jac = cv2.Rodrigues(i2.rvec)
 


    def pnp_test(self):
        for i, i1 in enumerate(self.image_list):
            print "pnp for %s" % i1.name
            fx = (self.focal_len_mm/self.horiz_mm) * float(i1.fullw)
            fy = (self.focal_len_mm/self.vert_mm) * float(i1.fullh)
            cx = i1.fullw * 0.5
            cy = i1.fullh * 0.5
            cam = np.array( [ [fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0] ] )
            att_sum = [ [0.0, 0.0], [0.0, 0.0], [0.0, 0.0] ]
            pos_sum = [0.0, 0.0, 0.0]
            weight_sum = 0.0
            for j, pairs in enumerate(i1.match_list):
                if i == j:
                    continue
                if len(pairs) < 8:
                    # start with only well correlated pairs
                    continue
                i2 = self.image_list[j]
                img_pts = []
                obj_pts = []
                for pair in pairs:
                    p1 = i1.kp_list[pair[0]].pt
                    p2 = i2.coord_list[pair[1]]
                    img_pts.append( p1 )
                    obj_pts.append( [p2[0], p2[1], 0.0] )
                img_pts = np.float32(img_pts)
                obj_pts = np.float32(obj_pts)
                #print "img_pts = %s" % str(img_pts)
                #print "obj_pts = %s" % str(obj_pts)
                (result, rvec, tvec) = cv2.solvePnP(obj_pts, img_pts, cam, None)
                print "  result = %s, rvec = %s, tvec = %s" \
                    % (result, rvec, tvec)
                R, jac = cv2.Rodrigues(rvec)
                print "  R =\n%s" % str(R)
                # googled how to derive the position in object
                # coordinates from solvePNP()
                pos = -np.matrix(R).T * np.matrix(tvec)
                for k in range(0,3):
                    pos_sum[k] += pos[k]
                print "  PNP pos = %s" % str(pos)

                # Remap the R matrix to match our coordinate system
                # (by inspection...)
                # [ [a, b, c], [d,  e,  f], [g, h,  i] ] =>
                # [ [h, b, a], [g, -e, -d], [i, c, -f] ]
                # this will be tedius code ...
                #Rconv[:3, 0] = R[:3, 2] # swap col 0 <=> col 2
                #Rconv[:3, 2] = R[:3, 0]
                #Rconv[1, :3] *= -1.0    # negate the middle row
                Rconv = R.copy()
                Rconv[0,0] = R[2,1]
                Rconv[0,1] = R[0,1]
                Rconv[0,2] = R[0,0]
                Rconv[1,0] = R[2,0]
                Rconv[1,1] = -R[1,1]
                Rconv[1,2] = -R[1,0]
                Rconv[2,0] = R[2,2]
                Rconv[2,1] = R[0,2]
                Rconv[2,2] = -R[1,2]
                print "Rconv =\n%s" % str(Rconv)
                (yaw, pitch, roll) = transformations.euler_from_matrix(Rconv, 'rzyx')
                att_sum[0][0] += math.cos(yaw)
                att_sum[0][1] += math.sin(yaw)
                att_sum[1][0] += math.cos(pitch)
                att_sum[1][1] += math.sin(pitch)
                att_sum[2][0] += math.cos(roll)
                att_sum[2][1] += math.sin(roll)
                weight_sum += i2.weight

                deg2rad = math.pi / 180.0
                print "  euler = %.2f %.2f %.2f" % (yaw/deg2rad,
                                                    pitch/deg2rad,
                                                    roll/deg2rad)
                print "  est = %.2f %.2f %.2f" % (i1.camera_yaw,
                                                  i1.camera_pitch,
                                                  i1.camera_roll)

                Rcam = transformations.euler_matrix(i1.camera_yaw*deg2rad,
                                                    i1.camera_pitch*deg2rad,
                                                    i1.camera_roll*deg2rad,
                                                    'rzyx')

                print "solvePNP =\n%s" % str(Rconv)
                print "my FIT =\n%s" % str(Rcam)

                v = np.array( [1.0, 0.0, 0.0] )
                vh = np.array( [1.0, 0.0, 0.0, 1.0] )
                print "  v = %s" % str(v)
                print "  Rconv * v = %s" % str(np.dot(Rconv, v))
                print "  Rcam * v = %s" % str(np.dot(Rcam, vh))
            if weight_sum < 0.0001:
                continue
            print "Camera pose for image %s:" % i1.name
            print "  PNP pos = %.2f %.2f %.2f" % (pos_sum[0]/weight_sum,
                                              pos_sum[1]/weight_sum,
                                              pos_sum[2]/weight_sum)
            print "  Fit pos = %s" % str((i1.camera_x, i1.camera_y, i1.camera_z))            
            yaw_avg = math.atan2(att_sum[0][1]/weight_sum,
                                 att_sum[0][0]/weight_sum)
            pitch_avg = math.atan2(att_sum[1][1]/weight_sum,
                                   att_sum[1][0]/weight_sum)
            roll_avg =  math.atan2(att_sum[2][1]/weight_sum,
                                   att_sum[2][0]/weight_sum)
            print "  PNP att = %.2f %.2f %.2f" % ( yaw_avg / deg2rad,
                                               pitch_avg / deg2rad,
                                               roll_avg / deg2rad )
            print "  Fit att = %.2f %.2f %.2f" % (i1.camera_yaw,
                                                  i1.camera_pitch,
                                                  i1.camera_roll)

    # should reset the fullw, fullh values in the keys files
    def recomputeWidthHeight(self):
        for image in self.image_list:
            if image.img == None:
                image.load_keys()
                image.load_rgb()
                image.save_keys()

    # write out the camera positions as geojson
    def save_geojson(self, path="mymap", cm_per_pixel=15.0 ):
        feature_list = []

        if not os.path.exists(path):
            os.makedirs(path)

        for i, image in enumerate(self.image_list):
            # camera point
            cam = geojson.Point( (image.aircraft_lon, image.aircraft_lat) )

            # coverage polys
            geo_list = []
            for pt in image.corner_list:
                lon = self.render.x2lon(pt[0])
                lat = self.render.y2lat(pt[1])
                geo_list.append( (lon, lat) )
            if len(geo_list) == 4:
                tmp = geo_list[2]
                geo_list[2] = geo_list[3]
                geo_list[3] = tmp
            poly = geojson.Polygon( [ geo_list ] )

            # group
            gc = geojson.GeometryCollection( [cam, poly] )
            source = "%s/%s" % (self.source_dir, image.name)
            work = "%s/%s" % (self.image_dir, image.name)
            f = geojson.Feature(geometry=gc, id=i,
                                properties={"name": image.name,
                                            "source": source,
                                            "work": work}) 
            feature_list.append( f )
        fc = geojson.FeatureCollection( feature_list )
        dump = geojson.dumps(fc)
        print str(dump)

        f = open( path + "/points.geojson", "w" )
        f.write(dump)
        f.close()

        warped_dir = path + "/warped"
        if not os.path.exists(warped_dir):
            os.makedirs(warped_dir )
        for i, image in enumerate(self.image_list):
            print "rendering %s" % image.name
            w, h, warp = \
                self.render.drawImage(image,
                                      source_dir=self.source_dir,
                                      cm_per_pixel=cm_per_pixel,
                                      keypoints=False,
                                      bounds=None)
            cv2.imwrite( warped_dir + "/" + image.name, warp )
