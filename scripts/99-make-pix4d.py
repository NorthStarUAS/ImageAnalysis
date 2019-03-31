#!/usr/bin/python3

import argparse
import csv
import datetime
import fnmatch
import fractions
import math
import os
import pyexiv2

from auracore import wgs84      # github.com/AuraUAS/aura-core
from props import getNode

from lib import ProjectMgr

d2r = math.pi / 180.0
r2d = 180.0 / math.pi

parser = argparse.ArgumentParser(description='Create a pix4d.csv file for a folder of geotagged images.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--force-heading', type=float, help='Force heading for every image')
parser.add_argument('--force-altitude', type=float, help='Fudge altitude geotag for stupid dji phantom 4 pro v2.0')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)

# load list of images
files = []
dir_node = getNode('/config/directories', True)
image_dir = os.path.normpath(dir_node.getStringEnum('image_sources', 0))
for file in os.listdir(image_dir):
    if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
        files.append(file)
files.sort()

def dms_to_decimal(degrees, minutes, seconds, sign=' '):
    """Convert degrees, minutes, seconds into decimal degrees.

    >>> dms_to_decimal(10, 10, 10)
    10.169444444444444
    >>> dms_to_decimal(8, 9, 10, 'S')
    -8.152777777777779
    """
    return (-1 if sign[0] in 'SWsw' else 1) * (
        float(degrees)        +
        float(minutes) / 60   +
        float(seconds) / 3600
    )

images = []
# read image exif timestamp (and convert to unix seconds)
#bar = Bar('Scanning image dir:', max = len(files))
for file in files:
    name = os.path.join(image_dir, file)
    #print(name)
    exif = pyexiv2.ImageMetadata(name)
    exif.read()
    GPS = 'Exif.GPSInfo.GPS'
    # print(exif[GPS + 'AltitudeRef'],
    #       exif[GPS + 'Altitude'],
    #       exif[GPS + 'Latitude'],
    #       exif[GPS + 'LatitudeRef'],
    #       exif[GPS + 'Longitude'],
    #       exif[GPS + 'LongitudeRef'])

    elat = exif[GPS + 'Latitude'].value
    lat = dms_to_decimal(elat[0], elat[1], elat[2], exif[GPS + 'LatitudeRef'].value)
    elon = exif[GPS + 'Longitude'].value
    lon = dms_to_decimal(elon[0], elon[1], elon[2], exif[GPS + 'LongitudeRef'].value)
    ealt = exif[GPS + 'Altitude']
    alt = float(ealt.value)
    #exif[GPS + 'MapDatum'])

    # print exif.exif_keys
    if 'Exif.Image.DateTime' in exif:
        strdate, strtime = str(exif['Exif.Image.DateTime'].value).split()
        year, month, day = strdate.split('-')
        hour, minute, second = strtime.split(':')
        d = datetime.date(int(year), int(month), int(day))
        t = datetime.time(int(hour), int(minute), int(second))
        dt = datetime.datetime.combine(d, t) 
        unixtime = float(dt.strftime('%s'))
    #print('pos:', lat, lon, alt, heading)
    line = [file, lat, lon]
    if not args.force_altitude:
        line.append(alt)
    else:
        line.append(args.force_altitude)
    images.append(line)
    #bar.next()
#bar.finish()

for i in range(len(images)):
    if i > 0:
        prev = images[i-1]
    else:
        prev = None
    cur = images[i]
    if i < len(images)-1:
        next = images[i+1]
    else:
        next = None

    if not prev is None:
        (prev_hdg, rev_course, prev_dist) = \
            wgs84.geo_inverse( prev[1], prev[2], cur[1], cur[2] )
    else:
        prev_hdg = 0.0
        prev_dist = 0.0
    if not next is None:
        (next_hdg, rev_course, next_dist) = \
            wgs84.geo_inverse( cur[1], cur[2], next[1], next[2] )
    else:
        next_hdg = 0.0
        next_dist = 0.0
        
    prev_hdgx = math.cos(prev_hdg*d2r)
    prev_hdgy = math.sin(prev_hdg*d2r)
    next_hdgx = math.cos(next_hdg*d2r)
    next_hdgy = math.sin(next_hdg*d2r)
    avg_hdgx = (prev_hdgx*prev_dist + next_hdgx*next_dist) / (prev_dist + next_dist)
    avg_hdgy = (prev_hdgy*prev_dist + next_hdgy*next_dist) / (prev_dist + next_dist)
    avg_hdg = math.atan2(avg_hdgy, avg_hdgx)*r2d
    if avg_hdg < 0:
        avg_hdg += 360.0
    print("%d %.2f %.1f %.2f %.1f %.2f" % (i, prev_hdg, prev_dist, next_hdg, next_dist, avg_hdg))
    images[i].append(0)
    images[i].append(0)
    if args.force_heading:
        images[i].append(args.force_heading)
    else:
        images[i].append(avg_hdg)

# traverse the image list and create output csv file
output_file = os.path.join(image_dir, 'pix4d.csv')
with open(output_file, 'w') as csvfile:
    writer = csv.DictWriter( csvfile,
                             fieldnames=['File Name',
                                         'Lat (decimal degrees)',
                                         'Lon (decimal degrees)',
                                         'Alt (meters MSL)',
                                         'Roll (decimal degrees)',
                                         'Pitch (decimal degrees)',
                                         'Yaw (decimal degrees)'] )
    writer.writeheader()
    for line in images:
        image = line[0]
        lat_deg = line[1]
        lon_deg = line[2]
        alt_m = line[3]
        roll_deg = line[4]
        pitch_deg = line[5]
        yaw_deg = line[6]
        print(image, lat_deg, lon_deg, alt_m)
        writer.writerow( { 'File Name': os.path.basename(image),
                           'Lat (decimal degrees)': "%.10f" % lat_deg,
                           'Lon (decimal degrees)': "%.10f" % lon_deg,
                           'Alt (meters MSL)': "%.2f" % alt_m,
                           'Roll (decimal degrees)': "%.2f" % roll_deg,
                           'Pitch (decimal degrees)': "%.2f" % pitch_deg,
                           'Yaw (decimal degrees)': "%.2f" % yaw_deg } )
