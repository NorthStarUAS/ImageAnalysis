#!/usr/bin/python

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
        self.kp_list = []
        self.kp_usage = []
        self.des_list = None
        self.match_list = []
        self.has_matches = True
        self.set_location()
        self.yaw_bias = 0.0
        self.roll_bias = 0.0
        self.pitch_bias = 0.0
        self.alt_bias = 0.0
        self.x_bias = 0.0
        self.y_bias = 0.0
        self.weight = 1.0
        self.connections = 0.0
        self.rotate = 0.0       # depricated?
        self.placed = False
        if image_file:
            self.load(image_dir, image_file)

    def set_location(self,
                     lon=0.0, lat=0.0, msl=0.0,
                     roll=0.0, pitch=0.0, yaw=0.0):
        self.lon = lon
        self.lat = lat
        self.msl = msl
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def load_image(self):
        if self.img == None:
            #print "Loading " + self.image_file
            try:
                self.img_rgb = cv2.imread(self.image_file)
                self.img = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
                # self.img = cv2.equalizeHist(gray)
                #self.img = gray
                
            except:
                print self.image_file + ":\n" + "  load error: " \
                    + str(sys.exc_info()[1])

    def load_keys(self):
        if len(self.kp_list) == 0 and os.path.exists(self.keys_file):
            #print "Loading " + self.keys_file
            try:
                self.keys_xml = ET.parse(self.keys_file)
            except:
                print self.keys_file + ":\n" + "  load error: " \
                    + str(sys.exc_info()[1])
            root = self.keys_xml.getroot()
            kp_node = root.find('keypoints')
            for kp in kp_node.findall('kp'):
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
                lon = float(root.find('longitude').text)
                lat = float(root.find('latitude').text)
                msl = float(root.find('altitude-msl').text)
                roll = float(root.find('roll').text)
                pitch = float(root.find('pitch').text)
                yaw = float(root.find('yaw').text)
                self.set_location(lon, lat, msl, roll, pitch, yaw)
                self.alt_bias = float(root.find('altitude-bias').text)
                self.roll_bias = float(root.find('roll-bias').text)
                self.pitch_bias = float(root.find('pitch-bias').text)
                self.yaw_bias = float(root.find('yaw-bias').text)
                self.x_bias = float(root.find('x-bias').text)
                self.y_bias = float(root.find('y-bias').text)
                self.weight = float(root.find('weight').text)
                if len(root.find('weight')):
                    self.connections = float(root.find('weight').text)
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
        root = ET.Element('image')
        xml = ET.ElementTree(root)

        # generate keypoints xml tree
        kp_node = ET.SubElement(root, 'keypoints')
        for i in xrange(len(self.kp_list)):
            kp = self.kp_list[i]
            e = ET.SubElement(kp_node, 'kp')
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
        ET.SubElement(root, 'longitude').text = "%.8f" % self.lon
        ET.SubElement(root, 'latitude').text = "%.8f" % self.lat
        ET.SubElement(root, 'altitude-msl').text = "%.2f" % self.msl
        ET.SubElement(root, 'roll').text = "%.2f" % self.roll
        ET.SubElement(root, 'pitch').text = "%.2f" % self.pitch
        ET.SubElement(root, 'yaw').text = "%.2f" % self.yaw

        ET.SubElement(root, 'altitude-bias').text = "%.2f" % self.alt_bias
        ET.SubElement(root, 'roll-bias').text = "%.2f" % self.roll_bias
        ET.SubElement(root, 'pitch-bias').text = "%.2f" % self.pitch_bias
        ET.SubElement(root, 'yaw-bias').text = "%.2f" % self.yaw_bias
        ET.SubElement(root, 'x-bias').text = "%.2f" % self.x_bias
        ET.SubElement(root, 'y-bias').text = "%.2f" % self.y_bias
        ET.SubElement(root, 'weight').text = "%.2f" % self.weight
        ET.SubElement(root, 'connections').text = "%d" % self.connections

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
        res = cv2.drawKeypoints(self.img, self.kp_list,
                                color=(0,255,0), flags=flags)
        fig1, plt1 = plt.subplots(1)
        plt1 = plt.imshow(res)
        plt.show(block=True) #block=True/Flase

