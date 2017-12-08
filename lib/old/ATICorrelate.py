#!/usr/bin/python

import commands
import csv
import fileinput
import fnmatch
import math
import os
import pyexiv2
import re

import spline

# AuraUAS/navigation
from aurauas.flightdata import flight_loader, flight_interp

time_fuzz = 1.5
#time_fuzz = 3.0
r2d = 180 / math.pi

class Correlate():
    def __init__(self):
        self.triggers = []
        self.pictures = []
        self.best_matchups = []
        self.filter_lat = []
        self.filter_lon = []
        self.filter_msl = []
        self.filter_roll = []
        self.filter_pitch = []
        self.filter_yaw = []
        self.master_time_offset = 0.0

    def load_all(self, flight_dir="", image_dir=""):
        self.flight_dir = flight_dir
        self.image_dir = image_dir
        data, flight_format = flight_loader.load(flight_dir)
        #data = flight_data.load('aura', flight_dir)
        self.interp = flight_interp.FlightInterpolate()
        self.interp.build(data)
        #self.load_gps()
        #self.load_filter()
        self.load_events()
        self.load_images()

    def load_events(self):
        self.triggers = []
        path = self.flight_dir + "/event-0.csv"
        last_trigger = 0.0
        interval = 0.0
        airborne = False
        ap = False
        with open(path, 'rb') as fevent:
            reader = csv.DictReader(fevent)
            for row in reader:
                tokens = row['message'].split()
                if len(tokens) == 2 and tokens[0] == 'mission:' \
                   and tokens[1] == 'airborne':
                    print 'airborne @', row['timestamp']
                    airborne = True
                if len(tokens) == 3 and tokens[0] == 'mission:' \
                   and tokens[1] == 'on' and tokens[2] == 'ground':
                    print 'on ground @', row['timestamp']
                    airborne = False
                if len(tokens) == 6 and tokens[0] == 'control:' \
                   and tokens[4] == 'on':
                    print 'ap on @', row['timestamp']
                    ap = True
                if len(tokens) == 6 and tokens[0] == 'control:' and \
                   tokens[4] == 'off':
                    print 'ap off @', row['timestamp']
                    ap = False
                if airborne and ap and len(tokens) == 4 and \
                   tokens[0] == 'camera:':
                    timestamp = float(row['timestamp'])
                    if last_trigger > 0.0:
                        interval = timestamp - last_trigger
                    lat = float(tokens[1])
                    lon = float(tokens[2])
                    agl = float(tokens[3])
                    self.triggers.append((timestamp, interval, lat, lon, agl))	
                    last_trigger = timestamp
        print "number of triggers = " + str(len(self.triggers))

    def load_images(self):
        files = []
        self.pictures = []
        path = self.image_dir
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
                files.append(file)
        files.sort()
        last_trigger = 0.0
        interval = 0
        for f in files:
            name = path + "/" + f
            print name
            exif = pyexiv2.ImageMetadata(name)
            exif.read()
            # print exif.exif_keys
            strdate, strtime = str(exif['Exif.Image.DateTime'].value).split()
            year, month, day = strdate.split('-')
            formated = year + "/" + month + "/" + day + " " + strtime + " UTC"
            result, unixtimestr = commands.getstatusoutput('date -d "' + formated + '" "+%s"' )
            unixtime = float(unixtimestr)
            # print f + ": " + strdate + ", " + strtime + ", " + unixtimestr

            if last_trigger > 0.0:
                interval = unixtime - last_trigger

            self.pictures.append( (unixtime, interval, f) )

            last_trigger = unixtime

        print "number of images = " + str(len(self.pictures))

        #for key in exif:
        #	print key + ": " + exif[key]


    def correlate(self, start_trigger_num):
        time_error_sum = 0.0

        matchups = []
        matchups.append( (0, start_trigger_num-1) )
        trig_time = self.triggers[start_trigger_num][0] + self.master_time_offset
        pict_time = self.pictures[0][0]
        time_diff = trig_time - pict_time
        #print str(pict_time) + " <=> " + str(trig_time) + ": " + str(time_diff)
        time_error_sum += time_diff

        num_triggers = len(self.triggers)
        t = start_trigger_num

        # iterate through each picture
        p_int = 0.0
        t_int = 0.0
        max_error = 0.0
        skipped_picts = 0
        for j in range(1, len(self.pictures)):
            #print "picture: " + str(j)
            p_int += self.pictures[j][1]
            if p_int <= t_int + time_fuzz:
                skipped_picts += 1
            while (p_int > t_int + time_fuzz) and (t < num_triggers):
                t_int += self.triggers[t][1]
                #print str(j) + ": " + str(p_int) + " =?= " + str(t_int)
                t += 1
            matchups.append( (j, t-1) )
            trig_time = self.triggers[t-1][0] + self.master_time_offset
            pict_time = self.pictures[j][0]
            time_diff = trig_time - pict_time
            #print str(pict_time) + " <=> " + str(trig_time) + ": " + str(time_diff)
            time_error_sum += time_diff

            error = math.fabs(p_int - t_int)
            if error > max_error:
                max_error = error

        avg_camera_time_error = round(time_error_sum / len(self.pictures))
        return (max_error, skipped_picts, avg_camera_time_error, matchups)

    def test_correlations(self):
        num_triggers = len(self.triggers)
        num_pictures = len(self.pictures)
        max_offset = num_triggers - num_pictures
        best_correlation = -1
        best_skip = len(self.pictures)
        best_error = 10000000.0
        self.best_matchups = []
        best_camera_time_error = 0.0
        # iterate through each possible starting point
        for i in range(1, max_offset+2):
            max_error, skipped_picts, cam_time_error, matchups = self.correlate(i)
            print "report for trigger/picture start offset: " + str(i-1)
            print "  max error: " + str(max_error)
            print "  skipped pictures: " + str(skipped_picts)
            print "  camera time correction: " + str(cam_time_error)
            if False and skipped_picts == 0 and max_error < time_fuzz:
                # ideal/best case correlation, let's just stop and celebrate!
                best_correlation = i
                best_camera_time_error = cam_time_error
                self.best_matchups = list(matchups)
                break
            elif skipped_picts <= best_skip and max_error < best_error:
                # best correlation we've seen so far
                best_correlation = i
                best_skip = skipped_picts
                best_error = max_error
                best_camera_time_error = cam_time_error
                self.best_matchups = list(matchups)

        print "Best correlation:"
        print "   Trigger number: %d" % (best_correlation - 1)
        print "   Skipped pictures: %d" % best_skip
        print "   Average time error: %.2f" % best_error
        print "   Camera clock offset: %.2f" % best_camera_time_error
        return best_correlation, best_camera_time_error

    def get_match(self, match):
        picture = self.pictures[match[0]]
	trigger = self.triggers[match[1]]
        return picture, trigger

    def get_position(self, time):
        lon = self.interp.filter_lon(time)
        lat = self.interp.filter_lat(time)
	msl = self.interp.filter_alt(time)
        return lon*r2d, lat*r2d, msl*1

    def get_attitude(self, time):
	phi = self.interp.filter_phi(time)
	the = self.interp.filter_the(time)
        psix = self.interp.filter_psix(time)
        psiy = self.interp.filter_psiy(time)
        psi = math.atan2(psiy, psix)
        return phi*r2d, the*r2d, psi*r2d
