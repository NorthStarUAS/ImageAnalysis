#!/usr/bin/python

import copy
import cv2
import lxml.etree as ET
from matplotlib import pyplot as plt
import numpy as np
import os.path
import sys

from find_obj import filter_matches,explore_match

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
        self.has_matches = True

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

    def load_image(self):
        if self.img == None:
            #print "Loading " + self.image_file
            try:
                self.img_rgb = cv2.imread(self.image_file)
                self.fullh, self.fullw, self.fulld = self.img_rgb.shape
                self.img = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
                # self.img = cv2.equalizeHist(gray)
                #self.img = gray
                return self.img_rgb
                
            except:
                print self.image_file + ":\n" + "  load error: " \
                    + str(sys.exc_info()[1])
        else:
            self.fullh, self.fullw, self.fulld = self.img_rgb.shape

    def load_full_image(self, source_dir):
        #print "Loading " + self.image_file
        full_name = source_dir + "/" + self.name
        try:
            full_image = cv2.imread(full_name)
            return full_image

        except:
            print full_image + ":\n" + "  load error: " \
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
        if os.path.exists(self.info_file):
            #print "Loading " + self.info_file
            try:
                xml = ET.parse(self.info_file)
                root = xml.getroot()
                self.has_matches = bool(root.find('has-matches').text)
                if root.find('longitude') is not None:
                    lon = float(root.find('longitude').text)
                else:
                    lon = float(root.find('aircraft-longitude').text)
                if root.find('latitude') is not None:
                    lat = float(root.find('latitude').text)
                else:
                    lat = float(root.find('aircraft-latitude').text)
                if root.find('altitude-msl') is not None:
                    msl = float(root.find('altitude-msl').text)
                else:
                    msl = float(root.find('aircraft-msl').text)
                if root.find('roll') is not None:
                    roll = float(root.find('roll').text)
                else:
                    roll = float(root.find('aircraft-roll').text)
                if root.find('pitch') is not None:
                    pitch = float(root.find('pitch').text)
                else:
                    pitch = float(root.find('aircraft-pitch').text)
                if root.find('yaw') is not None:
                    yaw = float(root.find('yaw').text)
                else:
                    yaw = float(root.find('aircraft-yaw').text)
                self.set_location(lon, lat, msl, roll, pitch, yaw)
                self.alt_bias = float(root.find('altitude-bias').text)
                self.roll_bias = float(root.find('roll-bias').text)
                self.pitch_bias = float(root.find('pitch-bias').text)
                self.yaw_bias = float(root.find('yaw-bias').text)
                self.x_bias = float(root.find('x-bias').text)
                self.y_bias = float(root.find('y-bias').text)
                self.weight = float(root.find('weight').text)
                self.connections = float(root.find('connections').text)
                self.error = float(root.find('error').text)
                self.stddev = float(root.find('stddev').text)
                if root.find('camera-yaw') is not None:
                    self.camera_yaw = float(root.find('camera-yaw').text)
                if root.find('camera-pitch') is not None:
                    self.camera_pitch = float(root.find('camera-pitch').text)
                if root.find('camera-roll') is not None:
                    self.camera_roll = float(root.find('camera-roll').text)
                if root.find('camera-x') is not None:
                    self.camera_x = float(root.find('camera-x').text)
                if root.find('camera-y') is not None:
                    self.camera_y = float(root.find('camera-y').text)
                if root.find('camera-z') is not None:
                    self.camera_z = float(root.find('camera-z').text)
            except:
                print self.info_file + ":\n" + "  load error: " \
                    + str(sys.exc_info()[1])

    def load(self, image_dir, image_file):
        print "Loading " + image_file
        self.name = image_file
        root, ext = os.path.splitext(image_file)
        self.file_root = image_dir + "/" + root
        self.image_file = image_dir + "/" + image_file
        self.keys_file = self.file_root + ".keys"
        self.des_file = self.file_root + ".npy"
        self.match_file = self.file_root + ".match"
        self.info_file = self.file_root + ".info"
        # lazy load actual image file if/when we need it
        # self.load_image()
        self.load_keys()
        self.load_descriptors()
        self.load_matches()
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
        root = ET.Element('information')
        xml = ET.ElementTree(root)
        ET.SubElement(root, 'has-matches').text = str(self.has_matches)
        ET.SubElement(root, 'aircraft-longitude').text = "%.10f" % self.aircraft_lon
        ET.SubElement(root, 'aircraft-latitude').text = "%.10f" % self.aircraft_lat
        ET.SubElement(root, 'aircraft-msl').text = "%.2f" % self.aircraft_msl
        ET.SubElement(root, 'aircraft-yaw').text = "%.2f" % self.aircraft_yaw
        ET.SubElement(root, 'aircraft-pitch').text = "%.2f" % self.aircraft_pitch
        ET.SubElement(root, 'aircraft-roll').text = "%.2f" % self.aircraft_roll
        ET.SubElement(root, 'aircraft-x').text = "%.3f" % self.aircraft_x
        ET.SubElement(root, 'aircraft-y').text = "%.3f" % self.aircraft_y
        ET.SubElement(root, 'altitude-bias').text = "%.2f" % self.alt_bias
        ET.SubElement(root, 'roll-bias').text = "%.2f" % self.roll_bias
        ET.SubElement(root, 'pitch-bias').text = "%.2f" % self.pitch_bias
        ET.SubElement(root, 'yaw-bias').text = "%.2f" % self.yaw_bias
        ET.SubElement(root, 'x-bias').text = "%.2f" % self.x_bias
        ET.SubElement(root, 'y-bias').text = "%.2f" % self.y_bias
        ET.SubElement(root, 'weight').text = "%.2f" % self.weight
        ET.SubElement(root, 'connections').text = "%d" % self.connections
        ET.SubElement(root, 'error').text = "%.3f" % self.error
        ET.SubElement(root, 'stddev').text = "%.3f" % self.stddev
        ET.SubElement(root, 'camera-yaw').text = "%.3f" % self.camera_yaw
        ET.SubElement(root, 'camera-pitch').text = "%.3f" % self.camera_pitch
        ET.SubElement(root, 'camera-roll').text = "%.3f" % self.camera_roll
        ET.SubElement(root, 'camera-x').text = "%.3f" % self.camera_x
        ET.SubElement(root, 'camera-y').text = "%.3f" % self.camera_y
        ET.SubElement(root, 'camera-z').text = "%.3f" % self.camera_z

        # write xml file
        try:
            xml.write(self.info_file, encoding="us-ascii",
                      xml_declaration=False, pretty_print=True)
        except:
            print self.info_file + ": error saving file: " \
                + str(sys.exc_info()[1])

    def show_keypoints(self, flags=0):
        # flags=0: draw only keypoints location
        # flags=4: draw rich keypoints
        if self.img == None:
            self.load_image()
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

