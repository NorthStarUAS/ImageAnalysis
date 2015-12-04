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
from progress.bar import Bar
import scipy.interpolate
import subprocess
import sys

import geojson

from getchar import find_getch
import Camera
import Image

import ImageList
import Matcher
import Placer
import Render
import transformations


class ProjectMgr():
    def __init__(self, project_dir=None):
        # directories
        self.project_dir = None  # project working directory
        self.source_dir = None   # original images
        self.image_dir = None    # working set of images

        self.cam = Camera.Camera()
        
        self.image_list = []

        self.detector_params = { 'detector': 'SIFT', # { SIFT, SURF, ORB, Star }
                                 'grid-detect': 1,
                                 'sift-max-features': 2000,
                                 'surf-hessian-threshold': 600,
                                 'surf-noctaves': 4,
                                 'orb-max-features': 2000,
                                 'star-max-size': 16,
                                 'star-response-threshold': 30,
                                 'star-line-threshold-projected': 10,
                                 'star-line-threshold-binarized': 8,
                                 'star-suppress-nonmax-size': 5 }
        self.matcher_params = { 'matcher': 'FLANN', # { FLANN or 'BF' }
                                'match-ratio': 0.75 }

        self.ned_reference_lla = []
        
        # the following member variables need to be reviewed/organized

        self.ac3d_steps = 8
        self.group_roll_bias = 0.0
        self.group_pitch_bias = 0.0
        self.group_yaw_bias = 0.0
        #self.group_alt_bias = 0.0
        self.k1 = 0.0
        self.k2 = 0.0
        #self.m = Matcher.Matcher()
        self.placer = Placer.Placer()
        self.render = Render.Render()
        
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
        # create a project dictionary and write it out as json
        if not os.path.exists(self.project_dir):
            print "Error: project doesn't exist =", self.project_dir
            return

        dirs = {}
        dirs['images-source'] = self.source_dir
        project_dict = {}
        project_dict['detector'] = self.detector_params
        project_dict['matcher'] = self.matcher_params
        project_dict['directories'] = dirs
        project_dict['ned-reference-lla'] = self.ned_reference_lla
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

        # save camera configuration
        self.cam.save(self.project_dir)

    def load(self, project_dir, create_if_needed=True):
        if not self.set_project_dir( project_dir ):
            return

        # load project configuration
        project_file = self.project_dir + "/Project.json"
        try:
            f = open(project_file, 'r')
            project_dict = json.load(f)
            f.close()

            if 'detector' in project_dict:
                self.detector_params = project_dict['detector']
            if 'matcher' in project_dict:
                self.matcher_params = project_dict['matcher']
            dirs = project_dict['directories']
            self.source_dir = dirs['images-source']
            self.ned_reference_lla = project_dict['ned-reference-lla']
        except:
            print "load error: " + str(sys.exc_info()[1])
            print "Notice: unable to read =", project_file
            print "Continuing with an empty project configuration"
 
        # load camera configuration
        self.cam.load(self.project_dir)

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

    def load_image_info(self, force_compute_sizes=False):
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

        # load rgb and determine image dimensions of this step has not
        # already been done
        bar = Bar('Computing image dimensions:', max = len(self.image_list))
        for image in self.image_list:
            if force_compute_sizes or image.height == 0 or image.width == 0:
                image.load_rgb(force_resize=True)
                image.save_meta()
            bar.next()
        bar.finish()
            
        # make sure our matcher gets a copy of the image list
        #self.m.setImageList(self.image_list)
        self.placer.setImageList(self.image_list)
        self.render.setImageList(self.image_list)

    def load_features(self):
        bar = Bar('Loading keypoints and descriptors:',
                  max = len(self.image_list))
        for image in self.image_list:
            image.load_features()
            image.load_descriptors()
            bar.next()
        bar.finish()

    def load_matches(self):
        bar = Bar('Loading keypoint (pair) matches:',
                  max = len(self.image_list))
        for image in self.image_list:
            image.load_matches()
            bar.next()
        bar.finish()

    def save_images_meta(self):
        for image in self.image_list:
            image.save_meta()

    def set_detector_params(self, dparams):
        self.detector_params = dparams
        
    def set_matcher_params(self, mparams):
        self.matcher_params = mparams
        
    def detect_features(self, force=True, show=False):
        if not show:
            bar = Bar('Detecting features:', max = len(self.image_list))
        for image in self.image_list:
            if force or len(image.kp_list) == 0 or image.des_list == None:
                #print "detecting features and computing descriptors: " + image.name
                gray = image.load_gray()
                image.detect_features(self.detector_params, gray)
                image.save_features()
                image.save_descriptors()
                image.save_matches()
                if show:
                    result = image.show_features()
                    if result == 27 or result == ord('q'):
                        break
            if not show:
                bar.next()
        if not show:
            bar.finish()

    def show_features_image(self, image):
        result = image.show_features()
        return result
        
    def show_features_images(self, name=None):
        for image in self.image_list:
            result = self.show_features_image(image)
            if result == 27 or result == ord('q'):
                break
                
    def findImageByName(self, name):
        for i in self.image_list:
            if i.name == name:
                return i
        return None

    # compute a center reference location (lon, lat) for the group of
    # images.
    def compute_ned_reference_lla(self):
        # requires images to have their location computed/loaded
        lon_sum = 0.0
        lat_sum = 0.0
        for image in self.image_list:
            lla, ypr, quat = image.get_aircraft_pose()
            lon_sum += lla[1]
            lat_sum += lla[0]
        self.ned_reference_lla = [ lat_sum / len(self.image_list),
                                   lon_sum / len(self.image_list),
                                   0.0 ]
        self.render.setRefCoord(self.ned_reference_lla)

    def undistort_uvlist(self, image, uv_orig):
        if len(uv_orig) == 0:
            return []
        # camera parameters
        camw, camh = self.cam.get_image_params()
        dist_coeffs = np.array(self.cam.camera_dict['dist-coeffs'],
                               dtype=np.float32)
        # scaled calibration matrix
        scale = float(image.width) / float(camw)
        K = self.cam.get_K(scale)
        # assemble the points in the proper format
        uv_raw = np.zeros((len(uv_orig),1,2), dtype=np.float32)
        for i, kp in enumerate(uv_orig):
            uv_raw[i][0] = (kp[0], kp[1])
        # do the actual undistort
        uv_new = cv2.undistortPoints(uv_raw, K, dist_coeffs, P=K)
        # return the results in an easier format
        result = []
        for i, uv in enumerate(uv_new):
            result.append(uv_new[i][0])
            #print "  orig = %s  undistort = %s" % (uv_raw[i][0], uv_new[i][0]
        return result
        
    # for each feature in each image, compute the undistorted pixel
    # location (from the calibrated distortion parameters)
    def undistort_keypoints(self):
        bar = Bar('Undistorting keypoints:', max = len(self.image_list))
        camw, camh = self.cam.get_image_params()
        for image in self.image_list:
            if len(image.kp_list) == 0:
                continue
            scale = float(image.width) / float(camw)
            K = self.cam.get_K(scale)
            uv_raw = np.zeros((len(image.kp_list),1,2), dtype=np.float32)
            for i, kp in enumerate(image.kp_list):
                uv_raw[i][0] = (kp.pt[0], kp.pt[1])
            dist_coeffs = np.array(self.cam.camera_dict['dist-coeffs'],
                                   dtype=np.float32)
            uv_new = cv2.undistortPoints(uv_raw, K, dist_coeffs, P=K)
            image.uv_list = []
            for i, uv in enumerate(uv_new):
                image.uv_list.append(uv_new[i][0])
                #print "  orig = %s  undistort = %s" % (uv_raw[i][0], uv_new[i][0])
            bar.next()
        bar.finish()
                
    # for each uv in the provided uv list, apply the distortion
    # formula to compute the original distorted value.
    def redistort(self, uv_list, K, dist_coeffs):
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]
        k1, k2, p1, p2, k3 = dist_coeffs
        
        uv_distorted = []
        for pt in uv_list:
            x = (pt[0] - cx) / fx
            y = (pt[1] - cy) / fy

            # Compute radius^2
            r2 = x**2 + y**2
            r4, r6 = r2**2, r2**3

            # Compute tangential distortion
            dx = 2*p1*x*y + p2*(r2 + 2*x*x)
            dy = p1*(r2 + 2*y*y) + 2*p2*x*y

            # Compute radial factor
            Lr = 1.0 + k1*r2 + k2*r4 + k3*r6

            ud = Lr*x + dx
            vd = Lr*y + dy
            uv_distorted.append( [ud * fx + cx, vd * fy + cy] )
            
        return uv_distorted
    
    # project the list of (u, v) pixels from image space into camera
    # space, remap that to a vector in ned space (for camera
    # ypr=[0,0,0], and then transform that by the camera pose, returns
    # the vector from the camera, through the pixel, into ned space
    def projectVectors(self, IK, image, uv_list, pose='direct'):
        if pose == 'direct':
            body2ned = image.get_body2ned() # IR
        elif pose == 'sba':
            body2ned = image.get_body2ned_sba() # IR
        # M is a transform to map the lens coordinate system (at zero
        # roll/pitch/yaw to the ned coordinate system at zero
        # roll/pitch/yaw).  It is essentially a +90 pitch followed by
        # +90 roll (or equivalently a +90 yaw followed by +90 pitch.)
        cam2body = image.get_cam2body()
        proj_list = []
        for uv in uv_list:
            uvh = np.array([uv[0], uv[1], 1.0])
            proj = body2ned.dot(cam2body).dot(IK).dot(uvh)
            proj_norm = transformations.unit_vector(proj)
            proj_list.append(proj_norm)

        #for uv in uv_list:
        #    print "uv:", uv
        #    uvh = np.array([uv[0], uv[1], 1.0])
        #    print "cam vec=", transformations.unit_vector(IR.dot(IK).dot(uvh))
        return proj_list

    # given a set of vectors in the ned frame, and a starting point.
    # Find the ground intersection point.  For any vectors which point into
    # the sky, return just the original reference/starting point.
    def intersectVectorsWithGroundPlane(self, pose, ground_m, v_list):
        pose_ned = pose['ned']
        pt_list = []
        for v in v_list:
            # solve projection
            p = pose_ned
            if v[2] > 0.0:
                d_proj = -(pose_ned[2] + ground_m)
                factor = d_proj / v[2]
                n_proj = v[0] * factor
                e_proj = v[1] * factor
                p = [ pose_ned[0] + n_proj, pose_ned[1] + e_proj, pose_ned[2] + d_proj ]
            pt_list.append(p)
        return pt_list

    def compute_kp_usage(self, all=False):
        print "Determing feature usage in matching pairs..."
        # but they may have different scaling or other attributes important
        # during feature matching
        if all:
            for image in self.image_list:
                image.kp_used = np.ones(len(image.kp_list), np.bool_)
        else:
            for image in self.image_list:
                image.kp_used = np.zeros(len(image.kp_list), np.bool_)
            for i1 in self.image_list:
                for j, matches in enumerate(i1.match_list):
                    i2 = self.image_list[j]
                    for k, pair in enumerate(matches):
                        i1.kp_used[ pair[0] ] = True
                        i2.kp_used[ pair[1] ] = True
                    
    # build an interpolation table for 'fast' projection of keypoints
    # into 3d world space
    #
    # 1. make a grid (i.e. 8x8) of uv coordinates covering the whole image
    # 2. undistort these uv coordinates
    # 3. project them into vectors
    # 4. intersect them with the srtm terrain to get ned coordinates
    # 5. use linearndinterpolator ... g = scipy.interpolate.LinearNDInterpolator([[0,0],[1,0],[0,1],[1,1]], [[0,4,8],[1,3,2],[2,2,-4],[4,1,0]])
    #    with origin uv vs. 3d location to build a table
    # 6. interpolate original uv coordinates to 3d locations
    def fastProjectKeypointsTo3d(self, sss):
        bar = Bar('Projecting keypoints to 3d:',
                  max = len(self.image_list))
        for image in self.image_list:
            camw, camh = self.cam.get_image_params()
            scale = float(image.width) / float(camw)
            K = self.cam.get_K(scale)
            IK = np.linalg.inv(K)
            
            # build a regular grid of uv coordinates
            size = 16
            u_grid = np.linspace(0, image.width, size+1)
            v_grid = np.linspace(0, image.height, size+1)
            uv_raw = []
            for u in u_grid:
                for v in v_grid:
                    uv_raw.append( [u,v] )
                    
            # undistort the grid of points
            uv_grid = self.undistort_uvlist(image, uv_raw)
            
            # project the grid out into vectors
            vec_list = self.projectVectors(IK, image, uv_grid)

            # intersect the vectors with the surface to find the 3d points
            coord_list = sss.interpolate_vectors(image.camera_pose, vec_list)

            # build the multidimenstional interpolator that relates
            # undistored uv coordinates to their 3d location.  Note we
            # could also relate the original raw/distored points to
            # their 3d locations and interpolate from the raw uv's,
            # but we already have a convenient list of undistored uv
            # points.
            g = scipy.interpolate.LinearNDInterpolator(uv_grid, coord_list)

            # interpolate all the keypoints now to approximate their
            # 3d locations
            image.coord_list = []
            for i, uv in enumerate(image.uv_list):
                if image.kp_used[i]:
                    coord = g(uv)
                    # coord[0] is the 3 element vector
                    if not np.isnan(coord[0][0]):
                        image.coord_list.append(coord[0])
                    else:
                        print "nan alert!"
                        print "  uv:", uv, "coord:", coord
                        print "check your image width/height values in the <image>.info files and figure out why they are wrong!"
                        quit()
                        #or append zeros which would be a hack until
                        #figuring out the root cause of the problem
                        #... if it isn't wrong image dimensions in the
                        #.info file...
                        #
                        #image.coord_list.append(np.zeros(3))
                else:
                    image.coord_list.append(np.zeros(3))
            bar.next()
        bar.finish()
                
#
# Below this point all the code needs to be reviewed/refactored
#

    def setWorldParams(self, ground_alt_m=0.0, yaw_bias=0.0, roll_bias=0.0, pitch_bias=0.0):
        print "Setting ground=%.1f yaw=%.2f roll=%.2f pitch=%.2f"\
            % (ground_alt_m, yaw_bias, roll_bias, pitch_bias)
        self.ground_alt_m = ground_alt_m
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

    def interpolateAircraftPositions(self, correlator, shutter_latency=0.0,
                                     force=False, weight=True):
        # tag each image with the flight data parameters at the time
        # the image was taken
        for match in correlator.best_matchups:
            pict, trig = correlator.get_match(match)
            image = self.findImageByName(pict[2])
            if image != None:
                aircraft_lon = 0.0
                aircraft_lat = 0.0
                if image.aircraft_pose:
                    aircraft_lat = image.aircraft_pose['lla'][0]
                    aircraft_lon = image.aircraft_pose['lla'][1]
                if force or (math.fabs(aircraft_lon) < 0.01 and math.fabs(aircraft_lat) < 0.01):
                    # only if we are forcing a new position
                    # calculation or the position is not already set
                    # from a save file.
                    t = trig[0] + shutter_latency
                    lon, lat, msl = correlator.get_position(t)
                    roll, pitch, yaw = correlator.get_attitude(t)
                    image.set_aircraft_pose( [lat, lon, msl],
                                             [yaw, pitch, roll] )
                    if weight:
                        # presumes a pitch/roll distance of 10, 10 gives a
                        # zero weight
                        w = 1.0 - (roll*roll + pitch*pitch)/200.0
                        if w < 0.01:
                            w = 0.01
                        image.weight = w
                    else:
                        image.weight = 1.0
                    image.save_meta()
                    #print "%s roll=%.1f pitch=%.1f weight=%.2f" % (image.name, roll, pitch, image.weight)

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
            image.save_meta()
            #print "%s roll=%.1f pitch=%.1f weight=%.2f" % (image.name, roll, pitch, image.weight)

    def computeConnections(self, force=None):
        for image in self.image_list:
            image.connections = 0
            for pairs in image.match_list:
                if len(pairs) >= self.m.min_pairs:
                    image.connections += 1
            image.save_meta()
            print "%s connections: %d" % (image.name, image.connections)


    # depricate this function .... or replace with better one (or just
    # use opencv)
    #
    # undistort x, y using a simple radial lens distortion model.  (We
    # call the original image values the 'distorted' values.)  Input
    # x,y are expected to be normalize (0.0 - 1.0) in image pixel
    # space with 0.5 being the center of image (and hopefully the
    # center of distortion.)
    def doLensUndistort(self, aspect_ratio, xnorm, ynorm):
        print "DEPRICATED..."
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

    def projectPoint2(self, image, q, pt, z_m):
        horiz_mm, vert_mm, focal_len_mm = self.cam.get_lens_params()
        h = image.height
        w = image.width
        print [h, w, self.cam.get_lens_params()]
        ar = float(w)/float(h)  # aspect ratio

        # normalized pixel coordinates to [0.0, 1.0]
        xnorm = pt[0] / float(w-1)
        ynorm = pt[1] / float(h-1)
        print "norm = %.4f %.4f" % (xnorm, ynorm)

        # lens un-distortion
        xnorm_u, ynorm_u = self.doLensUndistort(ar, xnorm, ynorm)
        print "norm_u = %.4f %.4f" % (xnorm_u, ynorm_u)

        # compute pixel coordinate in sensor coordinate space (mm
        # units) with (0mm, 0mm) being the center of the image.
        x_mm = (xnorm_u * 2.0 - 1.0) * (horiz_mm * 0.5)
        y_mm = (ynorm_u * 2.0 - 1.0) * (vert_mm * 0.5)
        print "x_mm = %.4f y_mm = %.4f" % ( x_mm, y_mm )
        
        # the forward vector (out the nose when the aircraft is
        # straight, level, and flying north) is (x=1.0, y=0.0, z=0.0).
        # This vector will get projected to the camera center point,
        # thus we have to remap the axes.
        #camvec = [y_mm, x_mm, focal_len_mm]
        camvec = [focal_len_mm, x_mm, y_mm]
        print "camvec orig = ", camvec
        camvec = transformations.unit_vector(camvec) # normalize
        print "camvec = %.3f %.3f %.3f" % (camvec[0], camvec[1], camvec[2])

        # transform camera vector (in body reference frame) to ned
        # reference frame
        ned = transformations.quaternion_backTransform(q, camvec)
        print "q = %s  ned = %s" % (str(q), str(ned))
        
        # solve projection
        if ned[2] < 0.0:
            # no interseciton
            return [0.0, 0.0]
        factor = z_m / ned[2]
        #print "z_m = %s" % str(z_m)
        x_proj = -ned[0] * factor
        y_proj = -ned[1] * factor
        #print "proj dist = %.2f" % math.sqrt(x_proj*x_proj + y_proj*y_proj)
        return [x_proj, y_proj]

    # project keypoints based on body reference system + body biases
    # transformed by camera mounting + camera mounting biases
    def projectImageKeypointsNative2(self, image, yaw_bias=0.0,
                                     roll_bias=0.0, pitch_bias=0.0,
                                     alt_bias=0.0):
        if image.img == None:
            image.load_rgb()
        h = image.height
        w = image.width
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
        image.save_meta()

        (coord_list, corner_list, grid_list) = \
            self.projectImageKeypointsNative3(image, pose, yaw_bias, roll_bias,
                                              pitch_bias, alt_bias)

        return coord_list, corner_list, grid_list

    d2r = math.pi / 180.0
    # project keypoints using the provided camera pose 
    # pose = (yaw_deg, pitch_deg, roll_deg, x_m, y_m, z_m)
    def projectImageKeypointsNative3(self, image, pose,
                                     yaw_bias=0.0, roll_bias=0.0,
                                     pitch_bias=0.0, alt_bias=0.0,
                                     all_keypoints=False):
        #print "Project3 for %s" % image.name
        if image.img == None:
            image.load_rgb()
        h = image.height
        w = image.width
        ar = float(w)/float(h)  # aspect ratio

        ned2cam = transformations.quaternion_from_euler((pose[0]+yaw_bias)*d2r,
                                                        (pose[1]+pitch_bias)*d2r,
                                                        (pose[2]+roll_bias)*d2r,
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
            proj = self.projectPoint2(image, ned2cam, kp.pt, z_m)
            #print "project3: kp=%s proj=%s" %(str(kp.pt), str(proj))
            coord_list[i] = [proj[1] + x_m, proj[0] + y_m]
        #print "coord_list = %s" % str(coord_list)

        # compute the corners (2x2 polygon grid) in image space
        dx = image.width - 1
        dy = image.height - 1
        y = 0.0
        for j in xrange(2):
            x = 0.0
            for i in xrange(2):
                #print "corner %.2f %.2f" % (x, y)
                proj = self.projectPoint2(image, ned2cam, [x, y], z_m)
                corner_list.append( [proj[1] + x_m, proj[0] + y_m] )
                x += dx
            y += dy

        # compute the ac3d polygon grid in image space
        dx = image.width / float(self.ac3d_steps)
        dy = image.height / float(self.ac3d_steps)
        y = 0.0
        for j in xrange(self.ac3d_steps+1):
            x = 0.0
            for i in xrange(self.ac3d_steps+1):
                #print "grid %.2f %.2f" % (xnorm_u, ynorm_u)
                proj = self.projectPoint2(image, ned2cam, [x, y], z_m)
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
        i1.save_meta()

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
            image.save_meta()

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
        image.save_meta()

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
        i1.save_meta()

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
            K = self.cam.get_K()
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
                (result, rvec, tvec) = cv2.solvePnP(obj_pts, img_pts, K, None)
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
            i1.save_meta()

    # call solvePnP() on all the matching pairs from all the matching
    # images simultaneously.  This works, but inherently weights the
    # fit much more towards the images with more matching pairs ... on
    # the other hand, that may be kind of what we want because images
    # with a few matches over a small area can grossly magnify any
    # errors into the result of solvePnP().
    def fitImagesWithSolvePnP2(self):
        for i, i1 in enumerate(self.image_list):
            #print "sovlePNP() for %s" % i1.name
            K = self.cam.get_K()
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
                    = cv2.solvePnP(obj_pts, img_pts, K, None,
                                   i1.rvec, i1.tvec,
                                   useExtrinsicGuess=True)
            else:
                # first time
                (result, i1.rvec, i1.tvec) \
                    = cv2.solvePnP(obj_pts, img_pts, K, None)

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
            i1.save_meta()
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

            K = self.cam.get_K()
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
                #    = cv2.solvePnP(obj_pts, img_pts, K, None)
                (rvec, tvec, status) \
                    = cv2.solvePnPRansac(obj_pts, img_pts, K, None)
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
            i1.save_meta()
            print "New cam pose %s %.2f %.2f %.2f  %.2f %.2f %.2f" \
                % (i1.name, i1.camera_yaw, i1.camera_pitch, i1.camera_roll,
                   i1.camera_x, i1.camera_y, i1.camera_z)

    def triangulate_test(self):
        for i, i1 in enumerate(self.image_list):
            print "pnp for %s" % i1.name
            K = self.cam.get_K()
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
            K = self.cam.get_K()
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
                (result, rvec, tvec) = cv2.solvePnP(obj_pts, img_pts, K, None)
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

    # should reset the width, height values in the keys files
    def recomputeWidthHeight(self):
        for image in self.image_list:
            if image.img == None:
                image.load_features()
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
