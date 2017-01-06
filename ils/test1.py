#!/usr/bin/python

import argparse
import calendar
from datetime import datetime, timedelta
import dateutil.parser
import ephem
import fileinput
import math
import os.path
import re
import time

# compute the ned vector to the sun
def compute_sun_ned(lon_deg, lat_deg, alt_m, timestamp):
    d = datetime.utcfromtimestamp(timestamp)
    #d = datetime.datetime.utcnow()
    
    ed = ephem.Date(d)
    #print 'ephem time utc:', ed
    #print 'localtime:', ephem.localtime(ed)

    ownship = ephem.Observer()
    ownship.lon = '%.8f' % lon_deg
    ownship.lat = '%.8f' % lat_deg
    ownship.elevation = alt_m
    ownship.date = ed

    sun = ephem.Sun(ownship)
    sun_ned = [ math.cos(sun.az), math.sin(sun.az), -math.sin(sun.alt) ]

    return sun_ned

parser = argparse.ArgumentParser(description='correct ILS for sensor orienation relative to sun at the time of the flight.')
parser.add_argument('--path', required=True, help='project path')
parser.add_argument('--tof', required=True, help='time of flight in iso format')
args = parser.parse_args()

dt = dateutil.parser.parse(args.tof)
unixtime = calendar.timegm(dt.timetuple())
print 'time of flight:', unixtime

metafile = os.path.join(args.path, 'image-metadata.txt')
fmeta = fileinput.input(metafile)
for line in fmeta:
    if re.match('^\s*File', line):
        # print "skipping csv header"
        continue
    tokens = re.split('[,\s]+', line.rstrip())
    lat_deg = float(tokens[1])
    lon_deg = float(tokens[2])
    alt_m = float(tokens[3])
    yaw_deg = float(tokens[4])
    pitch_deg = float(tokens[5])
    roll_deg = float(tokens[6])
    dumb_time = float(tokens[7]) * 1e-6
    ils = float(tokens[11])
    sun_ned = compute_sun_ned(lon_deg, lat_deg, alt_m, unixtime)
    print sun_ned


