#!/usr/bin/python3

import argparse
import csv
import datetime
import fnmatch
import fractions
import math
import os
import piexif
from libxmp.utils import file_to_dict

from auracore import wgs84      # github.com/AuraUAS/aura-core
from props import getNode

from lib import ProjectMgr

d2r = math.pi / 180.0
r2d = 180.0 / math.pi

parser = argparse.ArgumentParser(description='Create a pix4d.csv file for a folder of geotagged images.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--force-heading', type=float, help='Force heading for every image')
parser.add_argument('--force-altitude', type=float, help='Fudge altitude geotag for stupid dji phantom 4 pro v2.0')
parser.add_argument('--yaw-from-groundtrack', action='store_true', help='estimate yaw angle from ground track')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)

# load list of images
files = []
image_dir = args.project
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
        float(degrees[0] / degrees[1])        +
        float(minutes[0] / minutes[1]) / 60   +
        float(seconds[0] / seconds[1]) / 3600
    )

# save some work if true
images_have_yaw = False

images = []
# read image exif timestamp (and convert to unix seconds)
for file in files:
    name = os.path.join(image_dir, file)
    print(name)
    exif_dict = piexif.load(name)
    for ifd in exif_dict:
        if ifd == str("thumbnail"):
            print("thumb thumb thumbnail")
            continue
        print(ifd, ":")
        for tag in exif_dict[ifd]:
            print(ifd, tag, piexif.TAGS[ifd][tag]["name"], exif_dict[ifd][tag])

    elat = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
    lat = dms_to_decimal(elat[0], elat[1], elat[2],
                         exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef].decode('utf-8'))
    elon = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
    lon = dms_to_decimal(elon[0], elon[1], elon[2],
                         exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef].decode('utf-8'))
    #print(lon)
    ealt = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
    alt = ealt[0] / ealt[1]
    #exif_dict[GPS + 'MapDatum'])
    #print('lon ref', exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef])

    # print exif.exif_keys
    if piexif.ImageIFD.DateTime in exif_dict['0th']:
        strdate, strtime = exif_dict['0th'][piexif.ImageIFD.DateTime].decode('utf-8').split()
        year, month, day = strdate.split(':')
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

    
    # check for dji image heading tag
    xmp_top = file_to_dict(name)
    xmp = {}
    for key in xmp_top:
        for v in xmp_top[key]:
            xmp[v[0]] = v[1]
    #for key in xmp:
    #    print(key, xmp[key])
        
    if 'drone-dji:GimbalYawDegree' in xmp:
        images_have_yaw = True
        yaw_deg = float(xmp['drone-dji:GimbalYawDegree'])
        # print(name, 'yaw:', yaw_deg)
        line.append(0)          # assume zero pitch
        line.append(0)          # assume zero roll
        line.append(yaw_deg)
        
    images.append(line)

if not images_have_yaw or args.yaw_from_groundtrack:
    # do extra work to estimate yaw heading from gps ground track
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

# sanity check
output_file = os.path.join(image_dir, 'pix4d.csv')
if os.path.exists(output_file):
    print(output_file, "exists, please rename it and rerun this script.")
    quit()

# traverse the image list and create output csv file
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
