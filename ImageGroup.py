#!/usr/bin/python

import commands
import cv2
import fnmatch
import lxml.etree as ET
import math
from matplotlib import pyplot as plt
import numpy as np
import os.path
import subprocess
import sys

import PIL                      # test?

from getchar import find_getch
from Image import Image
import Matcher


class ImageGroup():
    def __init__(self, max_features=100, detect_grid=8, match_ratio=0.5):
        cells = detect_grid * detect_grid
        self.max_features = int(max_features / cells)
        self.match_ratio = match_ratio
        self.detect_grid = detect_grid
        self.file_list = []
        self.image_list = []
        self.ac3d_steps = 8
        self.shutter_latency = 0.0
        self.group_roll_bias = 0.0
        self.group_pitch_bias = 0.0
        self.group_yaw_bias = 0.0
        self.group_alt_bias = 0.0
        self.k1 = 0.0
        self.k2 = 0.0
        self.m = Matcher.Match()
        detectparams = dict(detector="sift", nfeatures=4000)
        #detectparams = dict(detector="surf", hessian_threshold=500)
        #detectparams = dict(detector="orb",  orb_max_features=800,
        #                    dense_detect_grid=4)
        matcherparams = dict(matcher="flann", match_ratio=match_ratio)
        #matcherparams = dict(matcher="bruteforce", match_ratio=match_ratio)
        self.m.configure(detectparams, matcherparams)

    def setCameraParams(self, horiz_mm=23.5, vert_mm=15.7, focal_len_mm=30.0):
        self.horiz_mm = horiz_mm
        self.vert_mm = vert_mm
        self.focal_len_mm = focal_len_mm

    def setWorldParams(self, ground_alt_m=0.0):
        self.ground_alt_m = ground_alt_m

    def update_work_dir(self, source_dir="", work_dir="", 
                        width=684, height=456):
        self.source_dir=source_dir
        self.work_dir=work_dir
        # double check work dir exists and make it if not
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        files = []
        for file in os.listdir(self.source_dir):
            if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
                files.append(file)
        files.sort()

        for file in files:
            # create resized working copy if needed
            name_in = self.source_dir + "/" + file
            name_out = self.work_dir + "/" + file
            if not os.path.isfile(name_out):
                command = "convert -geometry 684x456 %s %s" \
                          % (name_in, name_out)
                print command
                commands.getstatusoutput( command )

    def load_project(self):
        project_file = self.work_dir + "/project.xml"
        if os.path.exists(project_file):
            print "Loading " + project_file
            try:
                xml = ET.parse(project_file)
                root = xml.getroot()
                self.shutter_latency = float(root.find('shutter-latency').text)
                self.group_roll_bias = float(root.find('roll-bias').text)
                self.group_pitch_bias = float(root.find('pitch-bias').text)
                self.group_yaw_bias = float(root.find('yaw-bias').text)
                self.group_alt_bias = float(root.find('altitude-bias').text)
            except:
                print project_file + ":\n" + "  load error: " \
                    + str(sys.exc_info()[1])

    def save_project(self):
        project_file = self.work_dir + "/project.xml"
        root = ET.Element('project')
        xml = ET.ElementTree(root)
        ET.SubElement(root, 'shutter-latency').text = "%.2f" % self.shutter_latency
        ET.SubElement(root, 'roll-bias').text = "%.2f" % self.group_roll_bias
        ET.SubElement(root, 'pitch-bias').text = "%.2f" % self.group_pitch_bias
        ET.SubElement(root, 'yaw-bias').text = "%.2f" % self.group_yaw_bias
        ET.SubElement(root, 'altitude-bias').text = "%.2f" % self.group_alt_bias
        # write xml file
        try:
            xml.write(project_file, encoding="us-ascii",
                      xml_declaration=False, pretty_print=True)
        except:
            print project_file + ": error saving file: " \
                + str(sys.exc_info()[1])

    def load(self):
        # load project wide values
        self.load_project()

        self.file_list = []
        for file in os.listdir(self.work_dir):
            if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
                self.file_list.append(file)
        self.file_list.sort()
        for file_name in self.file_list:
            image = Image(self.work_dir, file_name)
            if len(image.kp_list) == 0 or image.des_list == None:
                print "  detecting features and computing descriptors"
                full_image = image.load_full_image(self.source_dir)
                image.kp_list = self.m.denseDetect(full_image)
                image.kp_list, image.des_list \
                    = self.m.computeDescriptors(full_image, image.kp_list)
                # and because we've messed with keypoints and descriptors
                image.match_list = []
                image.save_keys()
                image.save_descriptors()
                #image.show_keypoints()
            self.image_list.append( image )
        # make sure our matcher gets a copy of the image list
        self.m.setImageList(self.image_list)

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
                for pair in pairs:
                    i1.kp_usage[pair[0]] = True
                    i2.kp_usage[pair[1]] = True
        print "done."

    def computeCamPositions(self, correlator, force=False, weight=True):
        # tag each image with the flight data parameters at the time
        # the image was taken
        for match in correlator.best_matchups:
            pict, trig = correlator.get_match(match)
            image = self.m.findImageByName(pict[2])
            if image != None:
                if force or (math.fabs(image.lon) < 0.01 and math.fabs(image.lat) < 0.01):
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

    def computeWeights(self, force=None):
        # tag each image with the flight data parameters at the time
        # the image was taken
        for image in self.image_list:
            roll = image.roll + image.roll_bias
            pitch = image.pitch + image.pitch_bias
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

    def computeRefLocation(self):
        # requires images to have their location computed/loaded
        lon_sum = 0.0
        lat_sum = 0.0
        for i in self.image_list:
            lon_sum += i.lon
            lat_sum += i.lat
        self.ref_lon = lon_sum / len(self.image_list)
        self.ref_lat = lat_sum / len(self.image_list)
        print "Reference: lon = %.6f lat = %.6f" % (self.ref_lon, self.ref_lat)

    # undistort x, y using a simple radial lens distortion model.  (We
    # call the original image values the 'distorted' values.)  Input
    # x,y are expected to be normalize (0.0 - 1.0) in image pixel
    # space with 0.5 being the center of image (and hopefully the
    # center of distortion.)
    def doLensDistortion(self, aspect_ratio, xnorm, ynorm):
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

    def projectImageKeypoints(self, image, do_grid=False,
                              yaw_bias=0.0, roll_bias=0.0, pitch_bias=0.0,
                              alt_bias=0.0):
        Verbose = False

        prog = "/home/curt/Projects/UAS/ugear/build_linux-pc/utils/geo/geolocate"
        if image.img == None:
            image.load_image()
        h = image.fullh
        w = image.fullw
        ar = float(w)/float(h)  # aspect ratio
        lon = image.lon
        lat = image.lat
        msl = image.msl + image.alt_bias + self.group_alt_bias + alt_bias
        roll = -(image.roll + image.roll_bias + self.group_roll_bias + roll_bias)
        pitch = -(image.pitch + image.pitch_bias + self.group_pitch_bias + pitch_bias)
        yaw = image.yaw + image.yaw_bias + self.group_yaw_bias + yaw_bias
        yaw += image.rotate # from simple fit procedure (depricated?)
        yaw += 180.0        # camera is mounted backwards
        while yaw > 360.0:
            yaw -= 360.0
        while yaw < -360.0:
            yaw += 360.0
        #print "%s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % (image.name, image.roll, image.roll_bias, image.pitch, image.pitch_bias, image.yaw, image.yaw_bias, image.msl, image.alt_bias)
        if Verbose:
            for arg in [prog, str(lon), str(lat), str(msl), \
                        str(self.ground_alt_m), str(roll), str(pitch), \
                        str(yaw), str(self.horiz_mm), str(self.vert_mm), \
                        str(self.focal_len_mm), str(self.ref_lon), \
                        str(self.ref_lat)]:
                print arg,
            print
        process = subprocess.Popen([prog, str(lon), str(lat), str(msl), \
                                    str(self.ground_alt_m), str(roll), \
                                    str(pitch), \
                                    str(yaw), str(self.horiz_mm), \
                                    str(self.vert_mm), \
                                    str(self.focal_len_mm), \
                                    str(self.ref_lon), \
                                    str(self.ref_lat)], shell=False, \
                                   stdin=subprocess.PIPE, \
                                   stdout=subprocess.PIPE)

        # compute the keypoint locations in image space: [0.0,1.0][0.0,1.0]
        coords = ""
        for i, kp in enumerate(image.kp_list):
            if not image.kp_usage[i]:
                continue
            x = kp.pt[0]
            y = kp.pt[1]
            #print " project px = %.2f, %.2f" % (x, y)
            xnorm = x / float(w-1)
            ynorm = y / float(h-1)
            xnorm_u, ynorm_u = self.doLensDistortion(ar, xnorm, ynorm)
            coords += "%d %.5f %.5f\n" % (i, xnorm_u, ynorm_u)

        if True:
            # compute the corners (2x2 polygon grid) in image space
            dx = 1.0
            dy = 1.0
            ynorm = 0.0
            for j in xrange(2):
                xnorm = 0.0
                for i in xrange(2):
                    #print "cc %.2f %.2f" % (xnorm_u, ynorm_u)
                    xnorm_u, ynorm_u = self.doLensDistortion(ar, xnorm, ynorm)
                    coords += "cc %.3f %.3f\n" % (xnorm_u, ynorm_u)
                    xnorm += dx
                ynorm += dy

        if do_grid:
            # compute the ac3d polygon grid in image space
            dx = 1.0 / float(self.ac3d_steps)
            dy = 1.0 / float(self.ac3d_steps)
            ynorm = 0.0
            for j in xrange(self.ac3d_steps+1):
                xnorm = 0.0
                for i in xrange(self.ac3d_steps+1):
                    #print "cc %.2f %.2f" % (xnorm_u, ynorm_u)
                    xnorm_u, ynorm_u = self.doLensDistortion(ar, xnorm, ynorm)
                    coords += "gr %.3f %.3f\n" % (xnorm_u, ynorm_u)
                    xnorm += dx
                ynorm += dy

        # call the external project code
        result = process.communicate( coords )

        coord_list = [None] * len(image.kp_list)
        corner_list = []
        grid_list = []
        if Verbose:
            print image.name
        #f = open( 'junk', 'w' )
        for line in str(result[0]).split("\n"):
            #print "line = " + line
            tokens = line.split()
            if len(tokens) != 5 or tokens[0] != "result:":
                continue
            id = tokens[1]
            x = float(tokens[2]) + image.x_bias
            y = float(tokens[3]) + image.y_bias
            z = float(tokens[4])
            #print [ x, y, z ]
            if id == 'cc':
                corner_list.append( [x, y] )
            elif id == 'gr':
                grid_list.append( [x, y] )
            else:
                # print " project map = %.2f, %.2f" % (x, y)
                coord_list[int(id)] = [x, y]
            #f.write("%.2f\t%.2f\n" % (x, y))
        #f.close()
        return coord_list, corner_list, grid_list

    def projectKeypoints(self, do_grid=False):
        for image in self.image_list:
            coord_list, corner_list, grid_list \
                = self.projectImageKeypoints(image, do_grid)
            image.coord_list = coord_list
            image.corner_list = corner_list
            if do_grid:
                image.grid_list = grid_list

    # find affine transform between matching i1, i2 keypoints in map
    # space.  fullAffine=True means unconstrained to include best
    # warp/shear.  fullAffine=False means limit the matrix to only
    # best rotation, translation, and scale.
    def findAffine(self, i1, i2, pairs, fullAffine=False):
        src = []
        dst = []
        for pair in pairs:
            c1 = i1.coord_list[pair[0]]
            c2 = i2.coord_list[pair[1]]
            src.append( c1 )
            dst.append( c2 )
        #print str(src)
        #print str(dst)
        affine = cv2.estimateRigidTransform(np.array([src]).astype(np.float32),
                                            np.array([dst]).astype(np.float32),
                                            fullAffine)
        #print str(affine)
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
            if len(pairs) < self.m.min_pairs:
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
        #i1.rotate += 0.0
        print " image shift = %.2f %.2f" % (new_tx, new_ty)
        print " image rotate = %.2f" % (new_rot)
    
    def findImageWeightedAffine2(self, i1, fullAffine=False):
        # 1. find the affine transform for individual image pairs
        # 2. find the weighted average of the affine transform matrices

        # initialize sums with the match against ourselves
        affine_sum = np.array( [ [1.0, 0.0, 0.0 ], [0.0, 1.0, 0.0] ] )
        affine_sum *= i1.weight
        weight_sum = i1.weight  # our own weight
        for i, pairs in enumerate(i1.match_list):
            if len(pairs) < self.m.min_pairs:
                # skip matchups with < 3 pairs
                continue
            i2 = self.image_list[i]
            #print "Affine %s vs %s" % (i1.name, i2.name)
            affine = self.findAffine(i1, i2, pairs, fullAffine)
            if affine == None:
                # it's possible given a degenerate point set, the
                # affine estimator will return None
                continue
            affine_sum += affine * i2.weight
            weight_sum += i2.weight
            #self.showMatch(i1, i2, pairs)
        # weight_sum should always be greater than zero
        i1.new_affine = affine_sum / weight_sum
        #print str(i1.new_affine)
    
    # compare only against 'placed' images and do not weight ourselves
    def findImageWeightedAffine3(self, i1, fullAffine=False):
        # 1. find the affine transform for individual image pairs
        # 2. find the weighted average of the affine transform matrices

        # initialize sums with the match against ourselves
        affine_sum = np.array( [ [0.0, 0.0, 0.0 ], [0.0, 0.0, 0.0] ] )
        weight_sum = 0.0
        for i, pairs in enumerate(i1.match_list):
            if len(pairs) < self.m.min_pairs:
                continue
            i2 = self.image_list[i]
            if not i2.placed:
                continue
            #print "Affine %s vs %s" % (i1.name, i2.name)
            affine = self.findAffine(i1, i2, pairs, fullAffine)
            if affine == None:
                # it's possible given a degenerate point set, the
                # affine estimator will return None
                continue
            affine_sum += affine * i2.weight
            weight_sum += i2.weight
            #self.showMatch(i1, i2, pairs)
        # weight_sum should always be greater than zero
        if weight_sum > 0.00001:
            result = affine_sum / weight_sum
        else:
            result = np.array( [ [1.0, 0.0, 0.0 ], [0.0, 1.0, 0.0] ] )
        return result
    
    def affineTransformImage(self, image, gain=1.0, M=None):
        if M == None:
            M = image.new_affine

        # print "Transforming " + str(image.name)
        for i, coord in enumerate(image.coord_list):
            if not image.kp_usage[i]:
                continue
            newcoord = M.dot([coord[0], coord[1], 1.0])
            diff = newcoord - coord
            image.coord_list[i] += diff * gain
            # print "old %s -> new %s" % (str(coord), str(newcoord))
        for i, coord in enumerate(image.corner_list):
            newcoord = M.dot([coord[0], coord[1], 1.0])
            diff = newcoord - coord
            image.corner_list[i] += diff * gain
            # print "old %s -> new %s" % (str(coord), str(newcoord))
        for i, coord in enumerate(image.grid_list):
            newcoord = M.dot([coord[0], coord[1], 1.0])
            diff = newcoord - coord
            image.grid_list[i] += diff * gain
            # print "old %s -> new %s" % (str(coord), str(newcoord))

    def affineTransformImages(self, gain=0.1, fullAffine=False):
        for image in self.image_list:
            self.findImageWeightedAffine2(image, fullAffine=fullAffine)
        for image in self.image_list:
            self.affineTransformImage(image, gain)

    # return true if this image has a neighbor that is already been placed
    def hasPlacedNeighbor(self, image):
        for i, pairs in enumerate(image.match_list):
             if len(pairs) >= self.m.min_pairs:
                 i2 = self.image_list[i]
                 if i2.placed:
                     return True
        return False
        
    def affinePlaceImages(self, image_list=None, fullAffine=False):
        if image_list == None:
            image_list = self.image_list

        # reset the placed flag
        for image in image_list:
            image.placed = False
        done = False
        while not done:
            done = True
            maxcon = None
            maxidx = None
            # find an unplaced image with a placed neighbor that has
            # the most connections to other images
            for i, image in enumerate(image_list):
                if not image.placed and self.hasPlacedNeighbor(image) and (maxcon == None or image.connections > maxcon):
                    maxcon = image.connections
                    maxidx = i
                    done = False
            if maxidx == None:
                # find an unplaced image that has the most connections
                # to other images
                for i, image in enumerate(image_list):
                    if not image.placed and (maxcon == None or image.connections > maxcon):
                        maxcon = image.connections
                        maxidx = i
                        done = False
            if maxidx != None:
                image = image_list[maxidx]
                print "Placing %s (connections = %d)" % (image.name, maxcon)
                affine = self.findImageWeightedAffine3(image, fullAffine=fullAffine)
                self.affineTransformImage(image, gain=1.0, M=affine)
                image.placed = True

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

    def placeImages(self):
        for image in self.image_list:
            image.placed = False
        for image in self.image_list:
            self.findImageShift(image, gain=1.0, placing=True)
            self.placed = True

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
                if method == "average":
                    e = self.m.imageError(i)
                elif method == "variance":
                    e = math.sqrt(self.m.imageError(i, method=method))
                elif method == "max":
                    e = self.m.imageError(i, method)
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

        #print "Estimate %s for %s" % (param, image.name)
        var = False
        if method == "direct":
            var = False
        elif method == "variance":
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
                        = self.projectImageKeypoints(image,
                                                     yaw_bias=test_value)
                elif param == "roll":
                    coord_list, corner_list, grid_list \
                        = self.projectImageKeypoints(image,
                                                     roll_bias=test_value)
                elif param == "pitch":
                    coord_list, corner_list, grid_list \
                        = self.projectImageKeypoints(image,
                                                     pitch_bias=test_value)
                elif param == "altitude":
                    coord_list, corner_list, grid_list \
                        = self.projectImageKeypoints(image,
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
        refinements = 3

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
        image.yaw_bias += yaw*gain
        image.roll_bias += roll*gain
        image.pitch_bias += pitch*gain
        image.alt_bias += alt*gain
        image.save_info()
        coord_list = []
        corner_list = []
        grid_list = []
        # but don't save the results so we don't bias future elements
        # with moving previous elements
        coord_list, corner_list, grid_list = self.projectImageKeypoints(image)
        error = self.m.imageError(i, alt_coord_list=coord_list, method=method)
        print "Biases for %s (%s) is %.2f %.2f %.2f %.2f (%.3f)" \
            % (image.name, method,
               image.yaw_bias, image.roll_bias, image.pitch_bias,
               image.alt_bias, error)

    def fitImagesIndividually(self, method, gain):
        for i, image in enumerate(self.image_list):
            self.fitImage(i, method, gain)

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

    def generate_camera_location_report(self):
        for image in self.image_list:
            print "%s\t%.10f\t%.10f\t%.2f" \
                % (image.name, image.lon, image.lat, image.msl)

    def generate_ac3d(self, correlator, ref_image = False, base_name="quick", version=None ):
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
            agl = msl - self.ground_alt_m
            if image.has_matches and math.fabs(roll) <= max_roll and math.fabs(pitch) <= max_pitch and agl >= min_agl:
                match_count += 1

        # write AC3D header
        name = self.work_dir
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
            agl = msl - self.ground_alt_m
            if not image.has_matches or math.fabs(roll) > max_roll or math.fabs(pitch) > max_pitch or agl < min_agl:
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

    def imageCoverage(self, image):
        if not len(image.corner_list):
            return (0.0, 0.0, 0.0, 0.0)

        # find the min/max area of the image
        p0 = image.corner_list[0]
        minx = p0[0]; maxx = p0[0]; miny = p0[1]; maxy = p0[1]
        for pt in image.corner_list:
            if pt[0] < minx:
                minx = pt[0]
            if pt[0] > maxx:
                maxx = pt[0]
            if pt[1] < miny:
                miny = pt[1]
            if pt[1] > maxy:
                maxy = pt[1]
        print "%s coverage: (%.2f %.2f) (%.2f %.2f)" \
            % (image.name, minx, miny, maxx, maxy)
        return (minx, miny, maxx, maxy)

    def render_image(self, image=None,
                     cm_per_pixel=15.0, keypoints=False, bounds=None):
        if not len(image.corner_list):
            return
        if bounds == None:
            (minx, miny, maxx, maxy) = self.imageCoverage(image)
        else:
            (minx, miny, maxx, maxy) = bounds
        x = int(100.0 * (maxx - minx) / cm_per_pixel)
        y = int(100.0 * (maxy - miny) / cm_per_pixel)
        print "Drawing %s: (%d %d)" % (image.name, x, y)
        #print str(image.corner_list)

        full_image = image.load_full_image(self.source_dir)
        h, w, d = full_image.shape
        corners = np.float32([[0,0],[w,0],[0,h],[w,h]])
        target = np.array([image.corner_list]).astype(np.float32)
        for i, pt in enumerate(target[0]):
            #print "i=%d" % i
            target[0][i][0] = 100.0 * (target[0][i][0] - minx) / cm_per_pixel
            target[0][i][1] = 100.0 * (maxy - target[0][i][1]) / cm_per_pixel
        #print str(target)
        if keypoints:
            keypoints = []
            for i, kp in enumerate(image.kp_list):
                if image.kp_usage[i]:
                    keypoints.append(kp)
            src = cv2.drawKeypoints(full_image, keypoints,
                                    color=(0,255,0), flags=0)
        else:
            src = cv2.drawKeypoints(full_image, [],
                                    color=(0,255,0), flags=0)
        M = cv2.getPerspectiveTransform(corners, target)
        out = cv2.warpPerspective(src, M, (x,y))

        # clean up the edges so we don't have a ring of super dark pixels.
        ret, mask = cv2.threshold(out, 1, 255, cv2.THRESH_BINARY)
        kernel3 = np.ones((3,3),'uint8')
        mask = cv2.erode(mask, kernel3)
        out_clean = cv2.bitwise_and(out, mask)

        #cv2.imshow('output', out)
        #cv2.waitKey()
        return x, y, out_clean

    def render_add_to_image(self, base, new, blend_px=21):
        h, w, d = base.shape
        #print "h=%d w=%d d=%d" % ( h, w, d)

        # combine using masks and add operation (assumes pixel
        # image data will always be at least a little non-zero

        # create an inverse mask of the current accumulated imagery
        basegray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)  
        ret, base_mask_inv = cv2.threshold(basegray, 1, 255,
                                           cv2.THRESH_BINARY_INV)
        #cv2.imshow('base_mask_inv', base_mask_inv)

        # create an inverse mask of the new region to be added
        newgray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)  
        ret, new_mask = cv2.threshold(newgray, 1, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow('new_mask', new_mask)

        blendsize = (blend_px,blend_px)
        kernel = np.ones(blendsize,'uint8')
        base_mask_dilate = cv2.dilate(base_mask_inv, kernel)
        #cv2.imshow('base_mask_dilate', base_mask_dilate)
        base_mask_blur = cv2.blur(base_mask_dilate, blendsize)
        #cv2.imshow('base_mask_blur', base_mask_blur)

        base_mask_blur_inv = 255 - base_mask_blur
        #cv2.imshow('base_mask_blur_inv', base_mask_blur_inv)
        base_mask_blur_inv = base_mask_blur_inv | new_mask
        #cv2.imshow('base_mask_blur_inv2', base_mask_blur_inv)

        new[:,:,0] = new[:,:,0] * (base_mask_blur/255.0)
        new[:,:,1] = new[:,:,1] * (base_mask_blur/255.0)
        new[:,:,2] = new[:,:,2] * (base_mask_blur/255.0)
        #cv2.imshow('new masked', new)

        base[:,:,0] = base[:,:,0] * (base_mask_blur_inv/255.0)
        base[:,:,1] = base[:,:,1] * (base_mask_blur_inv/255.0)
        base[:,:,2] = base[:,:,2] * (base_mask_blur_inv/255.0)
        #cv2.imshow('base masked', base)

        fast = True
        if fast:
            # Now clip the new imagery against the area already covered
            #new = cv2.add(base, new, mask=mask_inv)

            # And combine ...
            base = cv2.add(base, new)

        else:
            # alpha blend using the mask as the alpha value, works but
            # is done the hardway because I can't find a native opencv
            # way to do this.
            mask_blur = cv2.blur(mask_inv, (50,50))
            for i in xrange(h):
                for j in xrange(w):
                    #(r0, g0, b0) = base[i][j]
                    #(r1, g1, b1) = new[i][j]
                    #a = mask_blur[i][j] / 255.0 
                    #r = r0*(1.0-a) + r1*a
                    #g = g0*(1.0-a) + g1*a
                    #b = b0*(1.0-a) + b1*a
                    #base = (r, g, b)
                    b = base[i][j]
                    n = new[i][j]
                    a = mask_blur[i][j] / 255.0
                    if n[0] + n[1] + n[2] > 0:
                        base[i][j][0] = b[0]*(1.0-a) + n[0]*a
                        base[i][j][1] = b[1]*(1.0-a) + n[1]*a
                        base[i][j][2] = b[2]*(1.0-a) + n[2]*a

        #cv2.imshow('base', base)
        #cv2.waitKey()

        return base
        
    def render_image_list(self, image_names=[],
                          cm_per_pixel=15.0, blend_cm=200, keypoints=False):
        # compute blend diameter in consistent pixel units
        blend_px = int(blend_cm/cm_per_pixel)+1
        if blend_px % 2 == 0:
            blend_px += 1

        minx = None; maxx = None; miny = None; maxy = None
        for name in image_names:
            image = self.m.findImageByName(name)
            (x0, y0, x1, y1) = self.imageCoverage(image)
            if minx == None or x0 < minx:
                minx = x0
            if miny == None or y0 < miny:
                miny = y0
            if maxx == None or x1 > maxx:
                maxx = x1
            if maxy == None or y1 > maxy:
                maxy = y1
        print "Group area coverage: (%.2f %.2f) (%.2f %.2f)" \
            % (minx, miny, maxx, maxy)

        x = int(100.0 * (maxx - minx) / cm_per_pixel)
        y = int(100.0 * (maxy - miny) / cm_per_pixel)
        print "New image dimensions: (%d %d)" % (x, y)
        base_image = np.zeros((y,x,3), np.uint8)

        for name in image_names:
            image = self.m.findImageByName(name)
            w, h, out = self.render_image(image, cm_per_pixel,
                                          keypoints,
                                          bounds=(minx, miny, maxx, maxy))
            base_image = self.render_add_to_image(base_image, out, blend_px)
            #(x0, y0, x1, y1) = self.imageCoverage(image)
            #w0 = int(100.0 * (x0 - minx) / cm_per_pixel)
            #h0 = int(100.0 * (maxy - y1) / cm_per_pixel)
            #print "roi (%d:%d %d:%d)" % ( w0, w, h0  , h )
            #roi = blank_image[h0:h, w0:w]
            #roi = out
            #roi = np.ones((h,w,3), np.uint8)

            #cv2.imshow('output', base_image)
            #cv2.waitKey()
        output_name = "output.jpg"
        cv2.imwrite(output_name, base_image)

        #s_img = cv2.imread("smaller_image.png", -1)
        #for c in range(0,3):
        #    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] = s_img[:,:,c] * (s_img[:,:,3]/255.0) +  l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] * (1.0 - s_img[:,:,3]/255.0)

    def render_images_over_point(self, x=0.0, y=0.0, pad=20.0,
                                 cm_per_pixel=15.0, blend_cm=200,
                                 keypoints=False):
        # build list of images covering target point
        coverage_list = []
        for image in self.image_list:
            (x0, y0, x1, y1) = self.imageCoverage(image)
            if x >= x0-pad and x <= x1+pad:
                if y >= y0-pad and y <= y1+pad:
                    if image.connections > 0:
                        # only add images that connect to other images
                        coverage_list.append(image)

        # sort by # of connections
        print "presort = %s" % str(coverage_list)
        coverage_list = sorted(coverage_list,
                               key=lambda image: image.connections,
                               reverse=True)
        print "postsort = %s" % str(coverage_list)

        # build name list
        name_list = []
        for image in coverage_list:
            name_list.append(image.name)

        self.affinePlaceImages(coverage_list)
        self.render_image_list(name_list,
                               cm_per_pixel=cm_per_pixel, blend_cm=blend_cm,
                               keypoints=keypoints)
