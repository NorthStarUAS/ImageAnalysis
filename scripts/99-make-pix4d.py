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

parser = argparse.ArgumentParser(description='Create a pix4d.csv file for a folder of geotagged images.')
parser.add_argument('--images', required=True, help='Directory containing the images')
args = parser.parse_args()

# load list of images
files = []
for file in os.listdir(args.images):
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
last_lon = None
last_lat = None
for file in files:
    name = os.path.join(args.images, file)
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
    if last_lat and last_lon:
        (heading, rev_course, leg_dist) = \
            wgs84.geo_inverse( last_lat, last_lon, lat, lon )
    else:
        heading = None
    #print('pos:', lat, lon, alt, heading)
    last_lat = lat
    last_lon = lon
    images.append([file, lat, lon, alt, 0.0, 0.0, heading])
    #bar.next()
#bar.finish()

# attempt to fix up the headings at the discontinuities
if len(images) > 1:
    images[0][6] = images[1][6]

for i in range(1, len(images)-1):
    if abs(images[i][6] - images[i-1][6]) > 45.0:
        images[i][6] = images[i+1][6]    

#for line in images:
#    print(line)
    
# traverse the image list and create output csv file
output_file = os.path.join(args.images, 'pix4d.csv')
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
