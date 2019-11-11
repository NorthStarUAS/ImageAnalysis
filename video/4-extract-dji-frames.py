#!/usr/bin/python3

# extract srt form of subtitles from dji movie (caption setting needs
# to be turned on when movie is recorded)
#
# ffmpeg -txt_format text -i input_file.MOV output_file.srt

import argparse
import cv2
import skvideo.io               # pip3 install scikit-video
import math
import fractions
import json
from matplotlib import pyplot as plt 
import numpy as np
import os
import pyexiv2
import re
import sys
from scipy import interpolate # strait up linear interpolation, nothing fancy

from auracore import wgs84
from aurauas_flightdata import flight_loader, flight_interp

parser = argparse.ArgumentParser(description='extract and geotag dji movie frames.')
parser.add_argument('--video', required=True, help='input video')
parser.add_argument('--cam-mount', choices=['forward', 'down', 'rear'],
                    default='down',
                    help='approximate camera mounting orientation')
parser.add_argument('--interval', type=float, default=1.0, help='extraction interval')
parser.add_argument('--distance', type=float, default=5.0, help='extraction distnace interval')
parser.add_argument('--start-time', type=float, help='begin frame grabbing at this time.')
parser.add_argument('--end-time', type=float, help='end frame grabbing at this time.')
parser.add_argument('--start-counter', type=int, default=1, help='first image counter')
parser.add_argument('--ground', type=float, required=True, help='ground altitude in meters')
parser.add_argument('--heading', type=float, required=True, help='fixed heading drone was flown at')
args = parser.parse_args()

r2d = 180.0 / math.pi

class Fraction(fractions.Fraction):
    """Only create Fractions from floats.

    >>> Fraction(0.3)
    Fraction(3, 10)
    >>> Fraction(1.1)
    Fraction(11, 10)
    """

    def __new__(cls, value, ignore=None):
        """Should be compatible with Python 2.6, though untested."""
        return fractions.Fraction.from_float(value).limit_denominator(99999)

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


def decimal_to_dms(decimal):
    """Convert decimal degrees into degrees, minutes, seconds.

    >>> decimal_to_dms(50.445891)
    [Fraction(50, 1), Fraction(26, 1), Fraction(113019, 2500)]
    >>> decimal_to_dms(-125.976893)
    [Fraction(125, 1), Fraction(58, 1), Fraction(92037, 2500)]
    """
    remainder, degrees = math.modf(abs(decimal))
    remainder, minutes = math.modf(remainder * 60)
    return [Fraction(n) for n in (degrees, minutes, remainder * 60)]

# pathname work
abspath = os.path.abspath(args.video)
basename, ext = os.path.splitext(abspath)
srtname = basename + ".srt"
dirname = basename + "_frames"
print("basename:", basename)
print("srtname:", srtname)
print("dirname:", dirname)

# check for required input files
if not os.path.isfile(args.video):
    print("%s doesn't exist, aborting ..." % args.video)
    quit()
    
if os.path.isfile(basename + ".srt"):
    srtname = basename + ".srt"
elif os.path.isfile(basename + ".SRT"):
    srtname = basename + ".SRT"
else:
    print("SRT (caption) file doesn't exist, aborting ...")
    quit()

# output directory
os.makedirs(dirname, exist_ok=True)

# read and parse srt file, setup data interpolator
need_interpolate = False
times = []
lats = []
lons = []
heights = []
ts = 0
lat = 0
lon = 0
height = 0
with open(srtname, 'r') as f:
    state = 0
    for line in f:
        if line.rstrip() == "":
            state = 0
        elif state == 0:
            counter = int(line.rstrip())
            state += 1
            # print(counter)
        elif state == 1:
            time_range = line.rstrip()
            (start, end) = time_range.split(' --> ')
            (shr, smin, ssec_txt) = start.split(':')
            (ssec, ssubsec) = ssec_txt.split(',')
            (ehr, emin, esec_txt) = end.split(':')
            (esec, esubsec) = esec_txt.split(',')
            ts = int(shr)*3600 + int(smin)*60 + int(ssec) + int(ssubsec)/1000
            te = int(ehr)*3600 + int(emin)*60 + int(esec) + int(esubsec)/1000
            print(ts, te)
            state += 1
        elif state == 2:
            # check for phantom (old) versus mavic2 (new) record
            data_line = line.rstrip()
            if re.search('\<font.*\>', data_line):
                # mavic 2
                state += 1
            else:
                # phantom
                need_interpolate = True
                m = re.search('(?<=GPS \()(.+)\)', data_line)
                (lon_txt, lat_txt, alt) = m.group(0).split(', ')
                m = re.search('(?<=, H )([\d\.]+)', data_line)
                if lat_txt != 'n/a':
                    lat = float(lat_txt)
                if lon_txt != 'n/a':
                    lon = float(lon_txt)
                height = float(m.group(0))
                # print('gps:', lat, lon, height)
                times.append(ts)
                lats.append(lat)
                lons.append(lon)
                heights.append(height)
                state = 0
        elif state == 3:
            # mavic 2 datetimem line
            datetime = line.rstrip()
            state += 1
        elif state == 4:
            # mavic 2 big data line
            data_line = line.rstrip()
            m = re.search('latitude : ([+-]?\d*\.\d*)', data_line)
            if m:
                lat = float(m.group(1))
            else:
                lat = None
            m = re.search('longt?itude : ([+-]?\d*\.\d*)', data_line)
            if m:
                lon = float(m.group(1))
            else:
                lon = None
            m = re.search('altitude.*: ([+-]?\d*\.\d*)', data_line)
            if m:
                alt = float(m.group(1))
            else:
                alt = None
            times.append(datetime)
            lats.append(lat)
            lons.append(lon)
            heights.append(alt)
        else:
            pass

if need_interpolate:
    print('setting up interpolators')
    interp_lats = interpolate.interp1d(times, lats,
                                       bounds_error=False, fill_value=0.0)
    interp_lons = interpolate.interp1d(times, lons,
                                       bounds_error=False, fill_value=0.0)
    interp_heights = interpolate.interp1d(times, heights,
                                          bounds_error=False, fill_value=0.0)

# fetch video metadata
metadata = skvideo.io.ffprobe(args.video)
#print(metadata.keys())
#print(json.dumps(metadata["video"], indent=4))
fps_string = metadata['video']['@avg_frame_rate']
(num, den) = fps_string.split('/')
fps = float(num) / float(den)
codec = metadata['video']['@codec_long_name']
w = int(metadata['video']['@width'])
h = int(metadata['video']['@height'])
print('fps:', fps)
print('codec:', codec)
print('output size:', w, 'x', h)

# extract frames
print("Opening ", args.video)
reader = skvideo.io.FFmpegReader(args.video, inputdict={}, outputdict={})

meta = os.path.join(dirname, "image-metadata.txt")
f = open(meta, 'w')
print("writing meta data to", meta)

last_time = -1000000
counter = 0
img_counter = args.start_counter
last_lat = 0
last_lon = 0
for frame in reader.nextFrame():
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    time = float(counter) / fps
    counter += 1
    print("frame: ", counter, "%.3f" % time)

    if args.start_time and time < args.start_time:
        continue
    if args.end_time and time > args.end_time:
        break

    if need_interpolate:
        lat_deg = interp_lats(time)
        lon_deg = interp_lons(time)
        alt_m = interp_heights(time) + args.ground
    else:
        if counter - 1 >= len(times):
            print("MORE FRAMES THAN SRT ENTRIS")
            continue
        datetime = times[counter - 1]
        lat_deg = lats[counter - 1]
        lon_deg = lons[counter - 1]
        alt_m = heights[counter - 1]
    if abs(lat_deg) < 0.001 and abs(lon_deg) < 0.001:
        continue
    (c1, c2, dist_m) = wgs84.geo_inverse(lat_deg, lon_deg, last_lat, last_lon)
    print("dist:", dist_m)
    #if time >= last_time + args.interval and dist_m >= args.distance:
    if dist_m >= args.distance:
        last_time = time
        file = os.path.join(dirname, "img_%04d" % img_counter + ".jpg")
        img_counter += 1
        cv2.imwrite(file, frame)
        # geotag the image
        exif = pyexiv2.ImageMetadata(file)
        exif.read()
        print(lat_deg, lon_deg, alt_m)
        exif['Exif.Image.DateTime'] = datetime
        GPS = 'Exif.GPSInfo.GPS'
        exif[GPS + 'AltitudeRef']  = '0' if alt_m >= 0 else '1'
        exif[GPS + 'Altitude']     = Fraction(alt_m)
        exif[GPS + 'Latitude']     = decimal_to_dms(lat_deg)
        exif[GPS + 'LatitudeRef']  = 'N' if lat_deg >= 0 else 'S'
        exif[GPS + 'Longitude']    = decimal_to_dms(lon_deg)
        exif[GPS + 'LongitudeRef'] = 'E' if lon_deg >= 0 else 'W'
        exif[GPS + 'MapDatum']     = 'WGS-84'
        exif.write()
        head, tail = os.path.split(file)
        f.write("%s,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f,%.2f\n" % (tail, lat_deg, lon_deg, alt_m, args.heading, 0.0, 0.0, time))
        last_lat = lat_deg
        last_lon = lon_deg
        
f.close()
