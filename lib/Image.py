#!/usr/bin/python

# Image.py - manage all the data scructures associated with an image

import cv2
import json
import lxml.etree as ET
from matplotlib import pyplot as plt
import numpy as np
import os.path
import sys

class Image():
    def __init__(self, image_dir=None, image_file=None):
        self.name = None
        self.img = None
        self.img_rgb = None
        self.height = 0
        self.width = 0
        self.kp_list = []
        self.kp_usage = []
        self.des_list = None
        self.match_list = []

        self.aircraft_pose = None
        self.camera_pose = None

        #self.camera_yaw = 0.0
        #self.camera_pitch = 0.0
        #self.camera_roll = 0.0
        #self.camera_x = 0.0
        #self.camera_y = 0.0
        #self.camera_z = 0.0

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
            if 'aircraft_pose' in image_dict:
                self.aircraft_pose = image_dict['aircraft-pose']
            if 'camera_pose' in image_dict:
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
            # if 'camera-yaw' in image_dict:
            #     self.camera_yaw = image_dict['camera-yaw']
            # if 'camera-pitch' in image_dict:
            #     self.camera_pitch = image_dict['camera-pitch']
            # if 'camera-roll' in image_dict:
            #     self.camera_roll = image_dict['camera-roll']
            # if 'camera-x' in image_dict:
            #     self.camera_x = image_dict['camera-x']
            # if 'camera-y' in image_dict:
            #     self.camera_y = image_dict['camera-y']
            # if 'camera-z' in image_dict:
            #     self.camera_z = image_dict['camera-z']
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
        if self.des_list == None and os.path.exists(self.des_file):
            #print "Loading " + self.des_file
            try:
                self.des_list = np.load(self.des_file)
            except:
                print self.des_file + ":\n" + "  load error: " \
                    + str(sys.exc_info()[1])

    def load_matches(self):
        if len(self.match_list) == 0 and os.path.exists(self.match_file):
            #print "Loading " + self.match_file
            try:
                self.match_xml = ET.parse(self.match_file)
            except:
                print self.match_file + ":\n" + "  load error: " \
                    + str(sys.exc_info()[1])
            root = self.match_xml.getroot()
            for match_node in root.findall('pairs'):
                pairs_str = match_node.text
                if pairs_str == None:
                    self.match_list.append( [] )
                else:
                    pairs = pairs_str.split('), (')
                    matches = []
                    for p in pairs:
                        p = p.replace('(', '')
                        p = p.replace(')', '')
                        i1, i2 = p.split(',')
                        matches.append( (int(i1), int(i2)) )
                    self.match_list.append( matches )
            # print str(self.match_list)

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
        root = ET.Element('matches')
        xml = ET.ElementTree(root)

        # generate matches xml tree
        for i in xrange(len(self.match_list)):
            match = self.match_list[i]
            match_node = ET.SubElement(root, 'pairs')
            if len(match) >= 4:
                pairs = str(match)
                pairs = pairs.replace('[', '')
                pairs = pairs.replace(']', '')
                match_node.text = pairs

        # write xml file
        try:
            xml.write(self.match_file, encoding="us-ascii",
                      xml_declaration=False, pretty_print=True)
        except:
            print self.match_file + ": error saving file: " \
                + str(sys.exc_info()[1])

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
        # image_dict['camera-yaw'] = self.camera_yaw
        # image_dict['camera-pitch'] = self.camera_pitch
        # image_dict['camera-roll'] = self.camera_roll
        # image_dict['camera-x'] = self.camera_x
        # image_dict['camera-y'] = self.camera_y
        # image_dict['camera-z'] = self.camera_z

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
            #norm = cv2.NORM_L2
        elif dparams['detector'] == 'SURF':
            threshold = float(dparams['surf-hessian-threshold'])
            detector = cv2.SURF(threshold)
            #norm = cv2.NORM_L2
        elif dparams['detector'] == 'ORB':
            max_features = int(dparams['orb-max-features'])
            grid_size = int(dparams['grid-detect'])
            cells = grid_size * grid_size
            max_cell_features = int(max_features / cells)
            detector = cv2.ORB(max_cell_features)
            #norm = cv2.NORM_HAMMING
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

        # compute() could potential add/remove keypoints so we want to
        # save the returned keypoint list, not our original detected
        # keypoint list
        self.kp_list, self.des_list = detector.compute(self.img, kp_list)
        
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
        hscale = float(h) / float(self.height)
        wscale = float(w) / float(self.width)
        kp_list = []
        for kp in self.kp_list:
            angle = kp.angle
            class_id = kp.class_id
            octave = kp.octave
            pt = kp.pt
            response = kp.response
            size = kp.size
            x = pt[0] * wscale
            y = pt[1] * hscale
            kp_list.append( cv2.KeyPoint(x, y, size, angle, response,
                                         octave, class_id) )

        res = cv2.drawKeypoints(self.img_rgb, kp_list,
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
    
    def set_aircraft_pose(self,
                          lon_deg=0.0, lat_deg=0.0, alt_m=0.0,
                          roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
        self.aircraft_pose = { 'longitude-deg': lon_deg,
                               'latitude-deg': lat_deg,
                               'altitude-m': alt_m,
                               'yaw-deg': yaw_deg,
                               'pitch-deg': pitch_deg,
                               'roll-deg': roll_deg }

    def get_aircraft_pose(self):
        p = self.aircraft_pose
        if p:
            return p['longitude-deg'], p['latitude-deg'], p['altitude-m'], p['roll-deg'], p['pitch-deg'], p['yaw-deg']
        else:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def set_camera_pose(self,
                        x_m=0.0, y_m=0.0, z_m=0.0,
                        roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
        self.camera_pose = { 'x-m': x_m,
                             'y-m': y_m,
                             'z-m': z_m,
                             'yaw-deg': yaw_deg,
                             'pitch-deg': pitch_deg,
                             'roll-deg': roll_deg }

    def get_camera_pose(self):
        p = self.camera_pose
        if p:
            return p['x-m'], p['y-m'], p['z-m'], p['roll-deg'], p['pitch-deg'], p<['yaw-deg']
        else:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
