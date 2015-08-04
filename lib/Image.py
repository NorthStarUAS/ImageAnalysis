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
        self.fullh = 0
        self.fullw = 0
        self.kp_list = []
        self.kp_usage = []
        self.des_list = None
        self.match_list = []
        self.num_matches = 0

        self.aircraft_yaw = 0.0
        self.aircraft_pitch = 0.0
        self.aircraft_roll = 0.
        self.aircraft_lon = 0.0
        self.aircraft_lat = 0.0
        self.aircraft_msl = 0.0
        self.aircraft_x = 0.0
        self.aircraft_y = 0.0

        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        self.camera_roll = 0.0
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_z = 0.0

        self.yaw_bias = 0.0
        self.roll_bias = 0.0
        self.pitch_bias = 0.0
        self.alt_bias = 0.0
        self.x_bias = 0.0
        self.y_bias = 0.0

        self.weight = 1.0
        self.connections = 0.0
        self.error = 0.0
        self.stddev = 0.0
        self.placed = False

        self.coord_list = []
        self.corner_list = []
        self.grid_list = []
        
        if image_file:
            self.load(image_dir, image_file)

    def set_location(self,
                     lon=0.0, lat=0.0, msl=0.0,
                     roll=0.0, pitch=0.0, yaw=0.0):
        self.aircraft_lon = lon
        self.aircraft_lat = lat
        self.aircraft_msl = msl
        self.aircraft_roll = roll
        self.aircraft_pitch = pitch
        self.aircraft_yaw = yaw

    def load_rgb(self):
        if self.img == None:
            #print "Loading " + self.image_file
            try:
                self.img_rgb = cv2.imread(self.image_file)
                self.fullh, self.fullw, self.fulld = self.img_rgb.shape
                self.img = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
                # self.img = cv2.equalizeHist(gray)
                # self.img = gray
                return self.img_rgb
                
            except:
                print self.image_file + ":\n" + "  load error: " \
                    + str(sys.exc_info()[1])
        else:
            self.fullh, self.fullw, self.fulld = self.img_rgb.shape

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

    def load_keys(self):
        if len(self.kp_list) == 0 and os.path.exists(self.keys_file):
            #print "Loading " + self.keys_file
            try:
                self.keys_xml = ET.parse(self.keys_file)
            except:
                print self.keys_file + ":\n" + "  load error: " \
                    + str(sys.exc_info()[1])
            root = self.keys_xml.getroot()
            self.fullw = int(root.find('width').text)
            self.fullh = int(root.find('height').text)
            for kp in root.findall('kp'):
                angle = float(kp.find('angle').text)
                class_id = int(kp.find('class_id').text)
                octave = int(kp.find('octave').text)
                pt = kp.find('pt').text
                x, y = map( float, str(pt).split() )
                response = float(kp.find('response').text)
                size = float(kp.find('size').text)
                self.kp_list.append( cv2.KeyPoint(x, y, size, angle, response, octave, class_id) )

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

    def load_info(self):
        if not os.path.exists(self.info_file):
            # no info file, create a new file
            self.save_info()
            return
        
        try:
            f = open(self.info_file, 'r')
            image_dict = json.load(f)
            f.close()
            self.num_matches = image_dict['num-matches']
            lon = image_dict['aircraft-longitude']
            lat = image_dict['aircraft-latitude']
            msl = image_dict['aircraft-msl']
            roll = image_dict['aircraft-roll']
            pitch = image_dict['aircraft-pitch']
            yaw = image_dict['aircraft-yaw']
            self.set_location(lon, lat, msl, roll, pitch, yaw)
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
            if 'camera-yaw' in image_dict:
                self.camera_yaw = image_dict['camera-yaw']
            if 'camera-pitch' in image_dict:
                self.camera_pitch = image_dict['camera-pitch']
            if 'camera-roll' in image_dict:
                self.camera_roll = image_dict['camera-roll']
            if 'camera-x' in image_dict:
                self.camera_x = image_dict['camera-x']
            if 'camera-y' in image_dict:
                self.camera_y = image_dict['camera-y']
            if 'camera-z' in image_dict:
                self.camera_z = image_dict['camera-z']
        except:
            print self.info_file + ":\n" + "  load error: " \
                + str(sys.exc_info()[1])

    def load(self, image_dir, image_file):
        print "Loading " + image_file
        self.name = image_file
        root, ext = os.path.splitext(image_file)
        file_root = image_dir + "/" + root
        self.image_file = image_dir + "/" + image_file
        self.keys_file = file_root + ".keys"
        self.des_file = file_root + ".npy"
        self.match_file = file_root + ".match"
        self.info_file = file_root + ".info"
        self.load_info()

    def save_keys(self):
        root = ET.Element('keypoints')
        xml = ET.ElementTree(root)

        width = ET.SubElement(root, 'width')
        width.text = str(self.fullw)
        height = ET.SubElement(root, 'height')
        height.text = str(self.fullh)

        # generate keypoints xml tree
        for i in xrange(len(self.kp_list)):
            kp = self.kp_list[i]
            e = ET.SubElement(root, 'kp')
            idx = ET.SubElement(e, 'index')
            idx.text = str(i)
            angle = ET.SubElement(e, 'angle')
            angle.text = str(kp.angle)
            class_id = ET.SubElement(e, 'class_id')
            class_id.text = str(kp.class_id)
            octave = ET.SubElement(e, 'octave')
            octave.text = str(kp.octave)
            pt = ET.SubElement(e, 'pt')
            pt.text = str(kp.pt[0]) + " " + str(kp.pt[1])
            response = ET.SubElement(e, 'response')
            response.text = str(kp.response)
            size = ET.SubElement(e, 'size')
            size.text = str(kp.size)
        # write xml file
        try:
            xml.write(self.keys_file, encoding="us-ascii",
                      xml_declaration=False, pretty_print=True)
        except:
            print self.keys_file + ": error saving file: " \
                + str(sys.exc_info()[1])

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

    def save_info(self):
        image_dict = {}
        image_dict['num-matches'] = self.num_matches
        image_dict['aircraft-longitude'] = self.aircraft_lon
        image_dict['aircraft-latitude'] = self.aircraft_lat
        image_dict['aircraft-msl'] = self.aircraft_msl
        image_dict['aircraft-yaw'] = self.aircraft_yaw
        image_dict['aircraft-pitch'] = self.aircraft_pitch
        image_dict['aircraft-roll'] = self.aircraft_roll
        image_dict['aircraft-x'] = self.aircraft_x
        image_dict['aircraft-y'] = self.aircraft_y
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
        image_dict['camera-yaw'] = self.camera_yaw
        image_dict['camera-pitch'] = self.camera_pitch
        image_dict['camera-roll'] = self.camera_roll
        image_dict['camera-x'] = self.camera_x
        image_dict['camera-y'] = self.camera_y
        image_dict['camera-z'] = self.camera_z

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

    def make_detector(self, detector, dparams):
        detector = None
        if detector == 'SIFT':
            nfeatures = dparams['nfeatures']
            detector = cv2.SIFT(nfeatures=nfeatures)
            #norm = cv2.NORM_L2
        elif detector == 'SURF':
            threshold = dparams['hessian_threshold']
            detector = cv2.SURF(threshold)
            #norm = cv2.NORM_L2
        elif detector == 'ORB':
            dmax_features = dparams['nfeatures']
            detector = cv2.ORB(dmax_features)
            #norm = cv2.NORM_HAMMING
        
        #if 'dense_detect_grid' in dparams:
        #    self.dense_detect_grid = dparams['dense_detect_grid']
        return detector

    def denseDetect(self, grid_size):
        steps = grid_size
        kp_list = []
        h, w, d = image.shape
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
                kps = self.detector.detect(image, mask)
                kp_list.extend( kps )
                y += dy
            x += dx
        return kp_list
        
    def show_keypoints(self, flags=0):
        # flags=0: draw only keypoints location
        # flags=4: draw rich keypoints
        if self.img == None:
            self.load_rgb()
        h, w = self.img.shape
        hscale = float(h) / float(self.fullh)
        wscale = float(w) / float(self.fullw)
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
        fig1, plt1 = plt.subplots(1)
        plt1 = plt.imshow(res)
        plt.show(block=True) #block=True/Flase

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

