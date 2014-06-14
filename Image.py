#!/usr/bin/python

import cv2
import fnmatch
import lxml.etree as ET
import math
from matplotlib import pyplot as plt
import numpy as np
import os.path
from pprint import pprint
import subprocess
import sys

from find_obj import filter_matches,explore_match

class Image():
    def __init__(self, image_dir=None, image_file=None):
        self.name = None
        self.img = None
        self.kp_list = []
        self.des_list = None
        self.match_list = []
        self.set_pos()
        self.rotate = 0.0
        self.shift = (0.0, 0.0)
        if image_file:
            self.load(image_dir, image_file)

    def set_pos(self, lon=0.0, lat=0.0, msl=0.0, roll=0.0, pitch=0.0, yaw=0.0):
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
                self.img = cv2.imread(self.image_file, 0)
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
                self.des_list = cv2.imread(self.des_file, 0)
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

    def load(self, image_dir, image_file):
        print "Loading " + image_file
        self.name = image_file
        root, ext = os.path.splitext(image_file)
        self.file_root = image_dir + "/" + root
        self.image_file = image_dir + "/" + image_file
        self.keys_file = self.file_root + ".keys"
        self.des_file = self.file_root + ".ppm"
        self.match_file = self.file_root + ".match"
        # lazy load actual image file if/when we need it
        # self.load_image()
        self.load_keys()
        self.load_descriptors()
        self.load_matches()

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
            result = cv2.imwrite(self.des_file, self.des_list)
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
            if len(match):
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

    def show_keypoints(self, flags=0):
        # flags=0: draw only keypoints location
        # flags=4: draw rich keypoints
        res = cv2.drawKeypoints(self.img, self.kp_list,
                                color=(0,255,0), flags=flags)
        fig1, plt1 = plt.subplots(1)
        plt1 = plt.imshow(res)
        plt.show(block=True) #block=True/Flase


class ImageGroup():
    def __init__(self, max_features=100, detect_grid=8, match_ratio=0.5):
        cells = detect_grid * detect_grid
        self.max_features = int(max_features / cells)
        self.match_ratio = match_ratio
        self.detect_grid = detect_grid
        self.file_list = []
        self.image_list = []
        self.orb = cv2.ORB(nfeatures=self.max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True)
        #self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ac3d_xsteps = 8
        self.ac3d_ysteps = 8

    def setCameraParams(self, horiz_mm=23.5, vert_mm=15.7, focal_len_mm=30.0):
        self.horiz_mm = horiz_mm
        self.vert_mm = vert_mm
        self.focal_len_mm = focal_len_mm

    def dense_detect(self, image):
        xsteps = self.detect_grid
        ysteps = self.detect_grid
        kp_list = []
        h, w = image.shape
        dx = 1.0 / float(xsteps)
        dy = 1.0 / float(ysteps)
        x = 0.0
        for i in xrange(xsteps):
            y = 0.0
            for j in xrange(ysteps):
                #print "create mask (%dx%d) %d %d" % (w, h, i, j)
                #print "  roi = %.2f,%.2f %.2f,%2f" % (y*h,(y+dy)*h-1, x*w,(x+dx)*w-1)
                mask = np.zeros((h,w,1), np.uint8)
                mask[y*h:(y+dy)*h-1,x*w:(x+dx)*w-1] = 255
                kps = self.orb.detect(image, mask)
                if False:
                    res = cv2.drawKeypoints(image, kps,
                                            color=(0,255,0), flags=0)
                    fig1, plt1 = plt.subplots(1)
                    plt1 = plt.imshow(res)
                    plt.show()
                kp_list.extend( kps )
                y += dy
            x += dx
        return kp_list

    def load(self, image_dir):
        self.file_list = []
        for file in os.listdir(image_dir):
            if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
                self.file_list.append(file)
        self.file_list.sort()
        for file_name in self.file_list:
            img = Image(image_dir, file_name)
            if len(img.kp_list) == 0 or img.des_list == None:
                if img.img == None:
                    img.load_image()
                # img.kp_list = self.orb.detect(img.img, None)
                img.kp_list = self.dense_detect(img.img)
                #img.show_keypoints()
                img.kp_list, img.des_list \
                    = self.orb.compute(img.img, img.kp_list)
                # and because we've messed with keypoints and descriptors
                img.match_list = []
                img.save_keys()
                img.save_descriptors()
            self.image_list.append( img )

    def filterMatches1(self, kp1, kp2, matches):
        mkp1, mkp2 = [], []
        idx_pairs = []
        for m in matches:
            if len(m): #and m[0].distance * self.match_ratio:
                print "d = %f" % m[0].distance
                mkp1.append( kp1[m[0].queryIdx] )
                mkp2.append( kp2[m[0].trainIdx] )
                idx_pairs.append( (m[0].queryIdx, m[0].trainIdx) )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, kp_pairs, idx_pairs

    def filterMatches2(self, kp1, kp2, matches):
        mkp1, mkp2 = [], []
        idx_pairs = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.match_ratio:
                #print " dist[0] = %d  dist[1] = %d" % (m[0].distance, m[1].distance)
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
                idx_pairs.append( (m.queryIdx, m.trainIdx) )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, kp_pairs, idx_pairs

    def computeMatches(self, showpairs=False):
        # O(n,n) compare
        for i, i1 in enumerate(self.image_list):
            if len(i1.match_list):
                continue
            for j, i2 in enumerate(self.image_list):
                matches = self.bf.knnMatch(i1.des_list, trainDescriptors=i2.des_list, k=2)
                p1, p2, kp_pairs, idx_pairs = self.filterMatches2(i1.kp_list, i2.kp_list, matches)
                #print "index pairs:"
                #print str(idx_pairs)
                #if i == j:
                #    continue

                if i != j:
                    i1.match_list.append( idx_pairs )
                else:
                    i1.match_list.append( [] )

                if len(idx_pairs):
                    print "Matching " + str(i) + " vs " + str(j) + " = " + str(len(idx_pairs))

                if len(idx_pairs) > 2:
                    if False:
                        # draw only keypoints location,not size and orientation (flags=0)
                        # draw rich keypoints (flags=4)
                        res1 = cv2.drawKeypoints(img_list[i], kp_list[i], color=(0,255,0), flags=0)
                        res2 = cv2.drawKeypoints(img_list[j], kp_list[j], color=(0,255,0), flags=0)
                        fig1, plt1 = plt.subplots(1)
                        plt1 = plt.imshow(res1)
                        fig2, plt2 = plt.subplots(1)
                        plt2 = plt.imshow(res2)
                        plt.show(block=False)

                    if showpairs:
                        if i1.img == None:
                            i1.load_image()
                        if i2.img == None:
                            i2.load_image()
                        explore_match('find_obj', i1.img, i2.img, kp_pairs) #cv2 shows image
                        cv2.waitKey()
                        cv2.destroyAllWindows()
            #print str(i1.match_list)
            i1.save_matches()

    def showMatch(self, i1, i2, idx_pairs):
        kp_pairs = []
        for p in idx_pairs:
            kp1 = i1.kp_list[p[0]]
            kp2 = i2.kp_list[p[1]]
            kp_pairs.append( (kp1, kp2) )
        if i1.img == None:
            i1.load_image()
        if i2.img == None:
            i2.load_image()
        explore_match('find_obj', i1.img, i2.img, kp_pairs) #cv2 shows image
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    def showMatches(self):
        # O(n,n) compare
        for i, i1 in enumerate(self.image_list):
            for j, i2 in enumerate(self.image_list):
                print "Showing matches for image %d and %d" % (i, j)
                print str(i1.match_list[j])
                idx_pairs = i1.match_list[j]
                if len(idx_pairs) > 0:
                    self.showMatch( i1, i2, idx_pairs )

    def findImageByName(self, name):
        for i in self.image_list:
            if i.name == name:
                return i
        return None

    def computeCamPositions(self, correlator, delay=0.0,
                            rollbias=0.0, pitchbias=0.0, yawbias=0.0,
                            mslbias=0.0):
        # tag each image with the flight data parameters at the time
        # the image was taken
        for match in correlator.best_matchups:
            pict, trig = correlator.get_match(match)
            lon, lat, msl = correlator.get_position(float(trig[0])+delay)
            roll, pitch, yaw = correlator.get_attitude(float(trig[0])+delay)
            roll += rollbias
            pitch += pitchbias
            yaw += yawbias
            msl += mslbias
            image = self.findImageByName(pict[2])
            if image != None:
                image.set_pos( lon, lat, msl, roll, pitch, yaw )
                # presumes a pitch/roll distance of 10, 10 gives a zero weight
                w = 1.0 - (roll*roll + pitch*pitch)/200.0
                if w < 0.01:
                    w = 0.01
                image.weight = w

    def computeRefLocation(self):
        # requires images to have their location computed/loaded
        lon_sum = 0.0
        lat_sum = 0.0
        for i in self.image_list:
            lon_sum += i.lon
            lat_sum += i.lat
        self.ref_lon = lon_sum / len(self.image_list)
        self.ref_lat = lat_sum / len(self.image_list)
        print "Reference: lon = " + str(self.ref_lon) + " lat = " + str(self.ref_lat)
            
    def computeKeyPointGeolocation(self, ground_alt_m):
        prog = "/home/curt/Projects/UAS/ugear/build_linux-pc/utils/geo/geolocate"
        for image in self.image_list:
            if image.img == None:
                image.load_image()
            h, w = image.img.shape
            lon = image.lon
            lat = image.lat
            msl = image.msl
            roll = -image.roll
            pitch = -image.pitch
            yaw = image.yaw
            yaw += image.rotate # from fit procedure
            yaw += 180.0        # camera is mounted backwards
            while yaw > 360.0:
                yaw -= 360.0
            while yaw < -360.0:
                yaw += 360.0
            for arg in [prog, str(lon), str(lat), str(msl), \
                        str(ground_alt_m), str(roll), str(pitch), \
                        str(yaw), str(self.horiz_mm), str(self.vert_mm), \
                        str(self.focal_len_mm), str(self.ref_lon), \
                        str(self.ref_lat)]:
                print arg,
            print
            process = subprocess.Popen([prog, str(lon), str(lat), str(msl), \
                                        str(ground_alt_m), str(roll), \
                                        str(pitch), \
                                        str(yaw), str(self.horiz_mm), \
                                        str(self.vert_mm), \
                                        str(self.focal_len_mm), \
                                        str(self.ref_lon), \
                                        str(self.ref_lat)], shell=False, \
                                       stdin=subprocess.PIPE, \
                                       stdout=subprocess.PIPE)
            coords = ""
            for i, kp in enumerate(image.kp_list):
                x = kp.pt[0]
                y = kp.pt[1]
                xnorm = x / float(w-1)
                ynorm = y / float(h-1)
                coords += "kp %.5f %.5f\n" % (xnorm, ynorm)
            # compute the ac3d polygon grid
            dx = 1.0 / float(self.ac3d_xsteps)
            dy = 1.0 / float(self.ac3d_ysteps)
            y = 0.0
            for j in xrange(self.ac3d_ysteps+1):
                x = 0.0
                for i in xrange(self.ac3d_xsteps+1):
                    #print "cc %.2f %.2f" % (x, y)
                    coords += "cc %.3f %.3f\n" % (x, y)
                    x += dx
                y += dy
            #coords += "c 0.0 0.0\n"
            #coords += "c 1.0 0.0\n"
            #coords += "c 1.0 1.0\n"
            #coords += "c 0.0 1.0\n"
            result = process.communicate( coords )
            image.coord_list = []
            image.corner_list = []
            print image.name
            #f = open( 'junk', 'w' )
            for line in str(result[0]).split("\n"):
                #print "line = " + line
                tokens = line.split()
                if len(tokens) != 5 or tokens[0] != "result:":
                    continue
                id = tokens[1]
                x = float(tokens[2])
                y = float(tokens[3])
                z = float(tokens[4])
                #print [ x, y, z ]
                if id == 'kp':
                    image.coord_list.append( [x, y] )
                elif id == 'cc':
                    image.corner_list.append( [x, y] )
                #f.write("%.2f\t%.2f\n" % (x, y))
            #f.close()
            if False:
                res1 = cv2.drawKeypoints(image.img, image.kp_list, color=(0,255,0), flags=0)
                fig1, plt1 = plt.subplots(1)
                plt1 = plt.imshow(res1)
                plt.show()

    # find affine transform between matching i1, i2 keypoints in map
    # space
    def findAffine(self, i1, i2, pairs, fullAffine=False):
        src = []
        dst = []
        for pair in pairs:
            c1 = i1.coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            src.append( c1 )
            dst.append( c2 )
        print str(src)
        print str(dst)
        affine = cv2.estimateRigidTransform(np.array([src]).astype(np.float32),
                                            np.array([dst]).astype(np.float32),
                                            fullAffine)
        print str(affine)
        return affine

    def findImageWeightedAffine1(self, i1):
        # 1. find the affine transform for individual image pairs
        # 2. decompose the affine matrix into scale, rotation, translation
        # 3. weight the decomposed values
        # 4. assemble a final 'weighted' affine matrix from the
        #    weighted average of the decomposed elements

        # initialize sums with the match against ourselves
        sx_sum = 1.0 * i1.weight # we are our same size
        sy_sum = 1.0 * i1.weight # we are our same size
        tx_sum = 0.0            # no translation
        ty_sum = 0.0
        rotate_sum = 0.0        # no rotation
        weight_sum = i1.weight  # our own weight
        for i, pairs in enumerate(i1.match_list):
            if len(pairs) < 3:
                continue
            i2 = self.image_list[i]
            print "Affine %s vs %s" % (i1.name, i2.name)
            affine = self.findAffine(i1, i2, pairs, fullAffine=False)
            if affine == None:
                # it's possible given a degenerate point set, the
                # affine estimator will return None
                continue
            # decompose the affine matrix into it's logical components
            tx = affine[0][2]
            ty = affine[1][2]

            a = affine[0][0]
            b = affine[0][1]
            c = affine[1][0]
            d = affine[1][1]

            sx = math.sqrt( a*a + b*b )
            if a < 0.0:
                sx = -sx
            sy = math.sqrt( c*c + d*d )
            if d < 0.0:
                sy = -sy

            rotate_deg = math.atan2(-b,a) * 180.0/math.pi
            if rotate_deg < -180.0:
                rotate_deg += 360.0
            if rotate_deg > 180.0:
                rotate_deg -= 360.0

            # update sums
            sx_sum += sx * i2.weight
            sy_sum += sy * i2.weight
            tx_sum += tx * i2.weight
            ty_sum += ty * i2.weight
            rotate_sum += rotate_deg * i2.weight
            weight_sum += i2.weight

            print "  shift = %.2f %.2f" % (tx, ty)
            print "  scale = %.2f %.2f" % (sx, sy)
            print "  rotate = %.2f" % (rotate_deg)
            #self.showMatch(i1, i2, pairs)
        # weight_sum should always be greater than zero
        new_sx = sx_sum / weight_sum
        new_sy = sy_sum / weight_sum
        new_tx = tx_sum / weight_sum
        new_ty = ty_sum / weight_sum
        new_rot = rotate_sum / weight_sum

        # compose a new 'weighted' affine matrix
        rot_rad = new_rot * math.pi / 180.0
        costhe = math.cos(rot_rad)
        sinthe = math.sin(rot_rad)
        row1 = [ new_sx * costhe, -new_sx * sinthe, new_tx ]
        row2 = [ new_sy * sinthe, new_sy * costhe, new_ty ]
        i1.new_affine = np.array( [ row1, row2 ] )
        print str(i1.new_affine)
        #i1.next_shift = ( 0.0, 0.0 )
        #i1.rotate += 0.0
        print " image shift = %.2f %.2f" % (new_tx, new_ty)
        print " image rotate = %.2f" % (new_rot)
    
    def findImageWeightedAffine2(self, i1):
        # 1. find the affine transform for individual image pairs
        # 2. find the weighted average of the affine transform matrices

        # initialize sums with the match against ourselves
        affine_sum = np.array( [ [1.0, 0.0, 0.0 ], [0.0, 1.0, 0.0] ] )
        affine_sum *= i1.weight
        weight_sum = i1.weight  # our own weight
        for i, pairs in enumerate(i1.match_list):
            if len(pairs) < 3:
                continue
            i2 = self.image_list[i]
            print "Affine %s vs %s" % (i1.name, i2.name)
            affine = self.findAffine(i1, i2, pairs, fullAffine=False)
            if affine == None:
                # it's possible given a degenerate point set, the
                # affine estimator will return None
                continue
            affine_sum += affine * i2.weight
            weight_sum += i2.weight
            #self.showMatch(i1, i2, pairs)
        # weight_sum should always be greater than zero
        i1.new_affine = affine_sum / weight_sum
        print str(i1.new_affine)
    
    def affineTransformImage(self, image, gain):
        # print "Transforming " + str(image.name)
        for i, coord in enumerate(image.coord_list):
            newcoord = image.new_affine.dot([coord[0], coord[1], 1.0])
            diff = newcoord - coord
            image.coord_list[i] += diff * gain
            # print "old %s -> new %s" % (str(coord), str(newcoord))
        for i, coord in enumerate(image.corner_list):
            newcoord = image.new_affine.dot([coord[0], coord[1], 1.0])
            diff = newcoord - coord
            image.corner_list[i] += diff * gain
            # print "old %s -> new %s" % (str(coord), str(newcoord))

    def affineTransformImages(self, gain=0.1):
        for image in self.image_list:
            self.findImageWeightedAffine2(image)
        for image in self.image_list:
            self.affineTransformImage(image, gain)

    def findImageRotate(self, i1, gain):
        #self.findImageAffine(i1) # temp test
        error_sum = 0.0
        weight_sum = i1.weight  # give ourselves an appropriate weight
        for i, match in enumerate(i1.match_list):
            if len(match) > 2:
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
                    
    def findImageShift(self, i1):
        xerror_sum = 0.0
        yerror_sum = 0.0
        weight_sum = i1.weight  # give ourselves an appropriate weight
        for i, match in enumerate(i1.match_list):
            if len(match) >= 2:
                i2 = self.image_list[i]
                print "Matching %s vs %s " % (i1.name, i2.name)
                for pair in match:
                    c1 = i1.coord_list[pair[0]]
                    c2 = i2.coord_list[pair[1]]
                    dx = (c2[0] + i2.shift[0]) - (c1[0] + i1.shift[0])
                    dy = (c2[1] + i2.shift[1]) - (c1[1] + i1.shift[1])
                    xerror_sum += dx * i2.weight
                    yerror_sum += dy * i2.weight
                    weight_sum += i2.weight
                    print str(pair)
                    print " x = %.2f  dy = %.2f weight = %.2f" % (dx, dy, i2.weight)
                    #print i1.kp_list[pair[0]].pt
                    #print i2.kp_list[pair[1]].pt
                    #print i1.coord_list[pair[0]]
                    #print i2.coord_list[pair[1]]
                    #print
        # divide by pairs + 1 gives some weight to our own position
        # (i.e. a zero rotate)
        xupdate = 0.0
        yupdate = 0.0
        if weight_sum > 0.0:
            xupdate = xerror_sum / weight_sum
            yupdate = yerror_sum / weight_sum
        i1.next_shift = ( xupdate, yupdate )
        print "Old Shift " + i1.name + " " + str(i1.next_shift)

    def shiftImages(self, gain=0.10):
        for image in self.image_list:
            self.findImageShift(image)
        for image in self.image_list:
            xshift = image.shift[0] + image.next_shift[0] * gain
            yshift = image.shift[1] + image.next_shift[1] * gain
            image.shift = ( xshift, yshift )
            print "%s: shift error = %.1f %.1f" % (image.name, image.shift[0], image.shift[1])

    # compute an error metric related to image placement among the group
    def imageError(self, i1):
        dist2_sum = 0.0
        weight_sum = i1.weight  # give ourselves an appropriate weight
        for i, match in enumerate(i1.match_list):
            if len(match) >= 2:
                i2 = self.image_list[i]
                print "Matching %s vs %s " % (i1.name, i2.name)
                for pair in match:
                    c1 = i1.coord_list[pair[0]]
                    c2 = i2.coord_list[pair[1]]
                    dx = (c2[0] + i2.shift[0]) - (c1[0] + i1.shift[0])
                    dy = (c2[1] + i2.shift[1]) - (c1[1] + i1.shift[1])
                    dist2 = dx*dx + dy*dy
                    dist2_sum += dist2 * i2.weight
                    weight_sum += i2.weight
        result = 0.0
        if weight_sum > 0.0:
            result = math.sqrt(dist2_sum / weight_sum)
        return result

    def globalError(self):
        sum = 0.0
        if len(self.image_list):
            for image in self.image_list:
                e = self.imageError(image)
                sum += e*e
            return sum / len(self.image_list)
        else:
            return 0.0

    def generate_ac3d(self, correlator, ground_m, geotag_dir = ".", ref_image = False, base_name="quick", version=None ):
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
            msl = image.msl
            roll = -image.roll
            pitch = -image.pitch
            agl = msl - ground_m
            if math.fabs(roll) <= max_roll and math.fabs(pitch) <= max_pitch and agl >= min_agl:
                match_count += 1

        # write AC3D header
        name = geotag_dir
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
            msl = image.msl
            roll = -image.roll
            pitch = -image.pitch
            agl = msl - ground_m
            if math.fabs(roll) > max_roll or math.fabs(pitch) > max_pitch or agl < min_agl:
                continue

            # compute a priority funciton (higher priority tiles are raised up)
            priority = (math.fabs(roll) + math.fabs(pitch) + agl/100.0) / 3.0

            ll = list(image.corner_list[0])
            ll.append( -priority )
            lr = list(image.corner_list[1])
            lr.append( -priority )
            ur = list(image.corner_list[2])
            ur.append( -priority )
            ul = list(image.corner_list[3])
            ul.append( -priority )

            f.write("OBJECT poly\n")
            f.write("name \"rect\"\n")
            f.write("texture \"./" + image.name + "\"\n")
            f.write("loc 0 0 0\n")

            f.write("numvert %d\n" % ((self.ac3d_xsteps+1) * (self.ac3d_ysteps+1)))
            # output the ac3d polygon grid (note the corner list is in
            # this specific order because that is how we generated it
            # earlier
            pos = 0
            for j in xrange(self.ac3d_ysteps+1):
                for i in xrange(self.ac3d_xsteps+1):
                    v = image.corner_list[pos]
                    f.write( "%.3f %.3f %.3f\n" % (v[0], v[1], -priority) )
                    pos += 1
  
            f.write("numsurf %d\n" % (self.ac3d_xsteps * self.ac3d_ysteps))
            dx = 1.0 / float(self.ac3d_xsteps)
            dy = 1.0 / float(self.ac3d_ysteps)
            y = 1.0
            for j in xrange(self.ac3d_ysteps):
                x = 0.0
                for i in xrange(self.ac3d_xsteps):
                    c = (j * (self.ac3d_ysteps+1)) + i
                    d = ((j+1) * (self.ac3d_ysteps+1)) + i
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
