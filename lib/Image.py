#!/usr/bin/python

# Image.py - manage all the data scructures associated with an image

import cv2
import json
import math
from matplotlib import pyplot as plt
import numpy as np
import os.path
import sys

import transformations


d2r = math.pi / 180.0           # a helpful constant
    
class Image():
    def __init__(self, image_dir=None, image_file=None):
        self.name = None
        self.img = None
        self.img_rgb = None
        self.height = 0
        self.width = 0
        self.kp_list = []       # opencv keypoint list
        self.kp_usage = []
        self.des_list = []      # opencv descriptor list
        self.match_list = []

        self.uv_list = []       # the 'undistorted' uv coordinates of all kp's
        
        self.aircraft_pose = None
        self.camera_pose = None

        # cam2body/body2cam are transforms to map between the standard
        # lens coordinate system (at zero roll/pitch/yaw and the
        # standard ned coordinate system at zero roll/pitch/yaw).
        # cam2body is essentially a +90 pitch followed by +90 roll (or
        # equivalently a +90 yaw followed by +90 pitch.)  This
        # transform simply maps coordinate systems and has nothing to
        # do with camera mounting offset or pose or anything other
        # than converting from one system to another.
        self.cam2body = np.array( [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                                  dtype=float )
        self.body2cam = np.linalg.inv(self.cam2body)

        self.yaw_bias = 0.0
        self.roll_bias = 0.0
        self.pitch_bias = 0.0
        self.alt_bias = 0.0
        self.x_bias = 0.0
        self.y_bias = 0.0

        # fixme: num_matches and connections appear to be the same
        # idea computed and used in different places.  We should be
        # able to collapse this into a single consistent value.
        self.num_matches = 0
        self.connections = 0.0
        self.weight = 1.0

        self.error = 0.0
        self.stddev = 0.0
        self.placed = False

        self.coord_list = []
        self.corner_list = []
        self.grid_list = []
        
        if image_file:
            self.name = image_file
            root, ext = os.path.splitext(image_file)
            file_root = image_dir + "/" + root
            self.image_file = image_dir + "/" + image_file
            self.features_file = file_root + ".feat"
            self.des_file = file_root + ".desc"
            self.match_file = file_root + ".match"
            self.info_file = file_root + ".info"
            # only load meta data when instance is created, other
            # items will be loaded 'just in time' depending on the
            # task to be performed later on
            self.load_meta()
            
    def load_meta(self):
        if not os.path.exists(self.info_file):
            # no info file, create a new file
            self.save_meta()
            return
        
        try:
            f = open(self.info_file, 'r')
            image_dict = json.load(f)
            f.close()
            self.num_matches = image_dict['num-matches']
            if 'aircraft-pose' in image_dict:
                self.aircraft_pose = image_dict['aircraft-pose']
            if 'camera-pose' in image_dict:
                self.camera_pose = image_dict['camera-pose']
            self.alt_bias = image_dict['altitude-bias']
            self.roll_bias = image_dict['roll-bias']
            self.pitch_bias = image_dict['pitch-bias']
            self.yaw_bias = image_dict['yaw-bias']
            self.x_bias = image_dict['x-bias']
            self.y_bias = image_dict['y-bias']
            self.weight = image_dict['weight']
            self.connections = image_dict['connections']
            self.error = image_dict['error']
            self.stddev = image_dict['stddev']
        except:
            print self.info_file + ":\n" + "  load error: " \
                + str(sys.exc_info()[1])

    def load_rgb(self):
        if self.img == None:
            #print "Loading " + self.image_file
            try:
                self.img_rgb = cv2.imread(self.image_file)
                self.height, self.width, self.fulld = self.img_rgb.shape
                self.img = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
                # self.img = cv2.equalizeHist(gray)
                # self.img = gray
                return self.img_rgb
                
            except:
                print self.image_file + ":\n" + "  load error: " \
                    + str(sys.exc_info()[1])
        else:
            self.height, self.width, self.fulld = self.img_rgb.shape

    def load_source_rgb(self, source_dir):
        #print "Loading " + self.image_file
        source_name = source_dir + "/" + self.name
        try:
            source_image = cv2.imread(source_name)
            return source_image

        except:
            print source_image + ":\n" + "  load error: " \
                + str(sys.exc_info()[1])
            return None

    def load_features(self):
        if len(self.kp_list) == 0 and os.path.exists(self.features_file):
            #print "Loading " + self.features_file
            try:
                f = open(self.features_file, 'r')
                feature_dict = json.load(f)
                f.close()
            except:
                print self.features_file + ":\n" + "  load error: " \
                    + str(sys.exc_info()[0]) + ": " + str(sys.exc_info()[1])
                return
            
            self.width = feature_dict['width']
            self.height = feature_dict['height']
            feature_list = feature_dict['features']
            for i, kp_dict in enumerate(feature_list):
                angle = kp_dict['angle']
                class_id = kp_dict['class-id']
                octave = kp_dict['octave']
                pt = kp_dict['pt']
                response = kp_dict['response']
                size = kp_dict['size']
                self.kp_list.append( cv2.KeyPoint(pt[0], pt[1], size,
                                                  angle, response, octave,
                                                  class_id) )

    def load_descriptors(self):
        filename = self.des_file + ".npy"
        if len(self.des_list) == 0 and os.path.exists(filename):
            #print "Loading " + filename
            try:
                self.des_list = np.load(filename)
                #print np.any(self.des_list)
                #val = "%s" % self.des_list
                #print
                #print "des_list.size =", self.des_list.size
                #print val
                #print
            except:
                print filename + ":\n" + "  load error: " \
                    + str(sys.exc_info()[1])
        else:
            print "no file:", filename
            
    def load_matches(self):
        try:
            f = open(self.match_file, 'r')
            self.match_list = json.load(f)
            f.close()
        except:
            print self.features_file + ":\n" + "  load error: " \
                + str(sys.exc_info()[0]) + ": " + str(sys.exc_info()[1])
            return

    def save_features(self):
        # convert from native opencv kp class to a dictionary
        feature_list = []
        feature_dict = { 'width': self.width,
                         'height': self.height,
                         'features': feature_list }
        for i, kp in enumerate(self.kp_list):
            kp_dict = { 'angle': kp.angle,
                        'class-id': kp.class_id,
                        'octave': kp.octave,
                        'pt': kp.pt,
                        'response': kp.response,
                        'size': kp.size }
            feature_list.append( kp_dict)
        try:
            f = open(self.features_file, 'w')
            json.dump(feature_dict, f, indent=2, sort_keys=True)
            f.close()
        except IOError as e:
            print "save_features(): I/O error({0}): {1}".format(e.errno, e.strerror)
            return
        except:
            raise

    def save_descriptors(self):
        # write descriptors as 'ppm image' format
        try:
            result = np.save(self.des_file, self.des_list)
        except:
            print self.des_file + ": error saving file: " \
                + str(sys.exc_info()[1])

    def save_matches(self):
        try:
            f = open(self.match_file, 'w')
            json.dump(self.match_list, f, sort_keys=True)
            f.close()
        except IOError as e:
            print self.info_file + ": error saving file: " \
                + str(sys.exc_info()[1])
            return
        except:
            raise

    def save_meta(self):
        image_dict = {}
        image_dict['num-matches'] = self.num_matches
        image_dict['aircraft-pose'] = self.aircraft_pose
        image_dict['camera-pose'] = self.camera_pose
        image_dict['altitude-bias'] = self.alt_bias
        image_dict['roll-bias'] = self.roll_bias
        image_dict['pitch-bias'] = self.pitch_bias
        image_dict['yaw-bias'] = self.yaw_bias
        image_dict['x-bias'] = self.x_bias
        image_dict['y-bias'] = self.y_bias
        image_dict['weight'] = self.weight
        image_dict['connections'] = self.connections
        image_dict['error'] = self.error
        image_dict['stddev'] = self.stddev

        try:
            f = open(self.info_file, 'w')
            json.dump(image_dict, f, indent=4, sort_keys=True)
            f.close()
        except IOError as e:
            print self.info_file + ": error saving file: " \
                + str(sys.exc_info()[1])
            return
        except:
            raise

    def make_detector(self, dparams):
        detector = None
        if dparams['detector'] == 'SIFT':
            max_features = int(dparams['sift-max-features'])
            detector = cv2.SIFT(nfeatures=max_features)
        elif dparams['detector'] == 'SURF':
            threshold = float(dparams['surf-hessian-threshold'])
            nOctaves = int(dparams['surf-noctaves'])
            print "octaves = ", nOctaves
            detector = cv2.SURF(hessianThreshold=threshold, nOctaves=nOctaves)
        elif dparams['detector'] == 'ORB':
            max_features = int(dparams['orb-max-features'])
            grid_size = int(dparams['grid-detect'])
            cells = grid_size * grid_size
            max_cell_features = int(max_features / cells)
            detector = cv2.ORB(max_cell_features)
        elif dparams['detector'] == 'Star':
            maxSize = int(dparams['star-max-size'])
            responseThreshold = int(dparams['star-response-threshold'])
            lineThresholdProjected = int(dparams['star-line-threshold-projected'])
            lineThresholdBinarized = int(dparams['star-line-threshold-binarized'])
            suppressNonmaxSize = int(dparams['star-suppress-nonmax-size'])
            detector = cv2.StarDetector(maxSize, responseThreshold,
                                        lineThresholdProjected,
                                        lineThresholdBinarized,
                                        suppressNonmaxSize)
        return detector

    def orb_grid_detect(self, detector, image, grid_size):
        steps = grid_size
        kp_list = []
        h, w = image.shape
        dx = 1.0 / float(steps)
        dy = 1.0 / float(steps)
        x = 0.0
        for i in xrange(steps):
            y = 0.0
            for j in xrange(steps):
                #print "create mask (%dx%d) %d %d" % (w, h, i, j)
                #print "  roi = %.2f,%.2f %.2f,%2f" % (y*h,(y+dy)*h-1, x*w,(x+dx)*w-1)
                mask = np.zeros((h,w,1), np.uint8)
                mask[y*h:(y+dy)*h-1,x*w:(x+dx)*w-1] = 255
                kps = detector.detect(image, mask)
                kp_list.extend( kps )
                y += dy
            x += dx
        return kp_list

    def detect_features(self, dparams):
        detector = self.make_detector(dparams)
        grid_size = int(dparams['grid-detect'])
        if dparams['detector'] == 'ORB' and grid_size > 1:
            kp_list = self.orb_grid_detect(detector, self.img, grid_size)
        else:
            kp_list = detector.detect(self.img)

        # compute the descriptors for the found features (Note: Star
        # is a special case that uses the brief extractor
        #
        # compute() could potential add/remove keypoints so we want to
        # save the returned keypoint list, not our original detected
        # keypoint list
        if dparams['detector'] == 'Star':
            extractor = cv2.DescriptorExtractor_create('ORB')
        else:
            extractor = detector
        self.kp_list, self.des_list = extractor.compute(self.img, kp_list)
        
        # wipe matches because we've touched the keypoints
        self.match_list = []

    # Displays the image in a window and waits for a keystroke and
    # then destroys the window.  Returns the value of the keystroke.
    def show_features(self, flags=0):
        # flags=0: draw only keypoints location
        # flags=4: draw rich keypoints
        if self.img == None:
            self.load_rgb()
        h, w = self.img.shape
        scale = 1000.0 / float(h)
        kp_list = []
        for kp in self.kp_list:
            angle = kp.angle
            class_id = kp.class_id
            octave = kp.octave
            pt = kp.pt
            response = kp.response
            size = kp.size
            x = pt[0] * scale
            y = pt[1] * scale
            kp_list.append( cv2.KeyPoint(x, y, size, angle, response,
                                         octave, class_id) )

        scaled_image = cv2.resize(self.img_rgb, (0,0), fx=scale, fy=scale)
        res = cv2.drawKeypoints(scaled_image, kp_list,
                                color=(0,255,0), flags=flags)
        cv2.imshow(self.name, res)
        print 'waiting for keyboard input...'
        key = cv2.waitKey() & 0xff
        cv2.destroyWindow(self.name)
        return key

    def coverage(self):
        if not len(self.corner_list):
            return (0.0, 0.0, 0.0, 0.0)

        # find the min/max area of the image
        p0 = self.corner_list[0]
        xmin = p0[0]; xmax = p0[0]; ymin = p0[1]; ymax = p0[1]
        for pt in self.corner_list:
            if pt[0] < xmin:
                xmin = pt[0]
            if pt[0] > xmax:
                xmax = pt[0]
            if pt[1] < ymin:
                ymin = pt[1]
            if pt[1] > ymax:
                ymax = pt[1]
        #print "%s coverage: (%.2f %.2f) (%.2f %.2f)" \
        #    % (self.name, xmin, ymin, xmax, ymax)
        return (xmin, ymin, xmax, ymax)
    
    def set_aircraft_pose(self, lla=[0.0, 0.0, 0.0], ypr=[0.0, 0.0, 0.0]):
        quat = transformations.quaternion_from_euler(ypr[0] * d2r,
                                                     ypr[1] * d2r,
                                                     ypr[2] * d2r,
                                                     'rzyx')
        self.aircraft_pose = { 'lla': lla, 'ypr': ypr, 'quat': quat.tolist() }

    def get_aircraft_pose(self):
        p = self.aircraft_pose
        if p and 'lla' in p and 'ypr' in p:
            return p['lla'], p['ypr'], np.array(p['quat'])
        else:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.zeros(4)

    # ned = [n_m, e_m, d_m] relative to the project ned reference point
    # ypr = [yaw_deg, pitch_deg, roll_deg] in the ned coordinate frame
    # note that the matrix derived from 'quat' is inv(R) is transpose(R)
    def set_camera_pose(self, ned=[0.0, 0.0, 0.0], ypr=[0.0, 0.0, 0.0]):
        quat = transformations.quaternion_from_euler(ypr[0] * d2r,
                                                     ypr[1] * d2r,
                                                     ypr[2] * d2r,
                                                     'rzyx')
        self.camera_pose = { 'ned': ned, 'ypr': ypr, 'quat': quat.tolist() }

    def get_camera_pose(self):
        p = self.camera_pose
        if p and 'ned' in p and 'ypr' in p:
            return p['ned'], p['ypr'], np.array(p['quat'])
        else:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.zeros(4)

    # cam2body rotation matrix (M)
    def get_cam2body(self):
        return self.cam2body

    # body2cam rotation matrix (IM)
    def get_body2cam(self):
        return self.body2cam

    # ned2body (R) rotation matrix
    def get_ned2body(self):
        p = self.camera_pose
        body2ned = transformations.quaternion_matrix(np.array(p['quat']))[:3,:3]
        return np.matrix(body2ned).T
    
    # body2ned (IR) rotation matrix
    def get_body2ned(self):
        p = self.camera_pose
        return transformations.quaternion_matrix(np.array(p['quat']))[:3,:3]

    # compute rvec and tvec (used to build the camera projection
    # matrix for things like cv2.triangulatePoints) from camera pose
    def get_proj(self):
        body2cam = self.get_body2cam()
        ned2body = self.get_ned2body()
        R = body2cam.dot( ned2body )
        rvec, jac = cv2.Rodrigues(R)
        ned = self.camera_pose['ned']
        tvec = -np.matrix(R) * np.matrix(ned).T
        return rvec, tvec
