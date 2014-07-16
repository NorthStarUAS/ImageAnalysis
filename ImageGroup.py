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
import Placer
import Render


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
        self.m = Matcher.Matcher()
        self.placer = Placer.Placer()
        self.render = Render.Render()
        detectparams = dict(detector="sift", nfeatures=1000)
        #detectparams = dict(detector="surf", hessian_threshold=600)
        #detectparams = dict(detector="orb",  orb_max_features=2000,
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
        self.placer.setImageList(self.image_list)
        self.render.setImageList(self.image_list)

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
        self.render.setRefCoord(self.ref_lon, self.ref_lat)
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
        coord_list = []
        corner_list = []
        grid_list = []
        # but don't save the results so we don't bias future elements
        # with moving previous elements
        coord_list, corner_list, grid_list = self.projectImageKeypoints(image)
        error = self.m.imageError(i, alt_coord_list=coord_list, method=method)
        if method == "average":
            image.error = error
        elif method == "stddev":
            image.stddev = error
        print "Fit %s (%s) is %.2f %.2f %.2f %.2f (avg=%.3f stddev=%.3f)" \
            % (image.name, method,
               image.yaw_bias, image.roll_bias, image.pitch_bias,
               image.alt_bias, image.error, image.stddev)
        image.save_info()

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

