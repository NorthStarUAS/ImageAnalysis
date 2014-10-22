#!/usr/bin/python

import commands
import fileinput
import fnmatch
import math
import os
import pyexiv2

import spline

time_fuzz = 1.5
#time_fuzz = 3.0


def simple_interp(points, v):
        index = spline.binsearch(points, v)
        n = len(points) - 1
        if index < n:
            xrange = points[index+1][0] - points[index][0]
            yrange = points[index+1][1] - points[index][1]
            # print(" xrange = $xrange\n")
            if xrange > 0.0001:
                percent = (v - points[index][0]) / xrange
                # print(" percent = $percent\n")
                return points[index][1] + percent * yrange
            else:
                return points[index][1]
        else:
            return points[index][1]


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
        self.load_gps()
        self.load_filter()
        self.load_events()
        self.load_images()

    def load_gps(self):
        path = self.flight_dir + "/gps.txt"
        print path
        f = fileinput.input(path)
        count = 0
        sum = 0.0
        for line in f:
            line.strip()
            field = line.split()
            time_diff = float(field[7]) - float(field[0])
            count += 1
            sum += time_diff
            # print str(field[0]) + " " + str(field[7])
        self.master_time_offset = sum / count
        print "Average offset = " + str(self.master_time_offset)

    def load_filter(self):
        path = self.flight_dir + "/filter.txt"
        f = fileinput.input(path)
        for line in f:
            line.strip()
            field = line.split()
            self.filter_lat.append( (float(field[0]), float(field[1])) )
            self.filter_lon.append( (float(field[0]), float(field[2])) )
            self.filter_msl.append( (float(field[0]), float(field[3])) )
            self.filter_roll.append( (float(field[0]), float(field[7])) )
            self.filter_pitch.append( (float(field[0]), float(field[8])) )
            self.filter_yaw.append( (float(field[0]), float(field[9])) )
            #print str(field[0]) + " " + str(field[9])

    def load_events(self):
        path = self.flight_dir + "/events.dat"
        f = fileinput.input(path)
        last_trigger = 0.0
        interval = 0.0
        for line in f:
            line.strip()
            field = line.split()
            if len(field) > 4 and field[1] == "mission" and field[2] == "Camera" and field[3] == "Trigger:":
                timestamp = float(field[0])

                if last_trigger > 0.0:
                    interval = timestamp - last_trigger

                lat = float(field[4])
                lon = float(field[5])
                agl = float(field[6])
                self.triggers.append( (timestamp, interval, lat, lon, agl) )	
                last_trigger = float(field[0])
        print "number of triggers = " + str(len(self.triggers))

    def load_images(self):
        path = self.image_dir
        files = []
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
            #print exif.exif_keys
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
        lon_deg = simple_interp(self.filter_lon, float(time))
        lat_deg = simple_interp(self.filter_lat, float(time))
	msl = simple_interp(self.filter_msl, float(time))
        return lon_deg, lat_deg, msl

    def get_attitude(self, time):
	roll_deg = simple_interp(self.filter_roll, float(time))
	pitch_deg = simple_interp(self.filter_pitch, float(time))
	yaw_deg = simple_interp(self.filter_yaw, float(time))
        return roll_deg, pitch_deg, yaw_deg
