#!/usr/bin/python

import argparse
import calendar
from datetime import datetime, timedelta
import dateutil.parser
import ephem
import fileinput
from math import cos, pi, sin
import numpy as np
import os.path
import re

from transformations import quaternion_from_euler, quaternion_matrix

d2r = pi / 180.0
r2d = 180.0 / pi

def norm(v):
    return v / np.linalg.norm(v)

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
    sun_ned = [ cos(sun.az), sin(sun.az), -sin(sun.alt) ]

    return norm(np.array(sun_ned))

def angle_between(v1, v2):
    v1_u = norm(v1)
    v2_u = norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

parser = argparse.ArgumentParser(description='correct ILS for sensor orienation relative to sun at the time of the flight.')
parser.add_argument('--path', required=True, help='project path')
parser.add_argument('--tof', required=True, help='time of flight in iso format')
args = parser.parse_args()

dt = dateutil.parser.parse(args.tof)
unixtime = calendar.timegm(dt.timetuple()) + 18000
print('time of flight (unix):', unixtime)

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
    ils = float(tokens[10])
    sun_ned = compute_sun_ned(lon_deg, lat_deg, alt_m, unixtime)
    #print sun_ned

    quat = quaternion_from_euler(yaw_deg * d2r, pitch_deg * d2r, roll_deg * d2r, 'rzyx')
    #print quat

    body2ned = quaternion_matrix(quat)[:3,:3]
    #print body2ned

    up = np.matrix( [0, 0, -1] ).T
    up_ned = norm((body2ned * up).A1)
    #print 'up:', up_ned

    forward = np.matrix( [1, 0, 0] ).T
    forward_ned = norm((body2ned * forward).A1)
    #print 'forward:', forward_ned

    rel_sun_angle = angle_between(sun_ned, up_ned) * r2d

    print('%s,%.2f,%.2f' % (tokens[0], rel_sun_angle, ils))
