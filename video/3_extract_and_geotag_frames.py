#!/usr/bin/python3

# find our custom built opencv first

import argparse
import cv2
import skvideo.io               # pip3 install scikit-video
import math
import fractions
from matplotlib import pyplot as plt 
import numpy as np
import os
import pyexiv2
import re
import sys

from aurauas.flightdata import flight_loader, flight_interp

import correlate

parser = argparse.ArgumentParser(description='correlate movie data to flight data.')
parser.add_argument('--flight', required=True, help='load specified aura flight log')
parser.add_argument('--movie', required=True, help='original movie if extracting frames')
parser.add_argument('--cam-mount', choices=['forward', 'down', 'rear'],
                    default='down',
                    help='approximate camera mounting orientation')
parser.add_argument('--interval', type=float, default=1.0, help='capture interval')
parser.add_argument('--resample-hz', type=float, default=60.0, help='resample rate (hz)')
parser.add_argument('--time-shift', type=float, help='skip autocorrelation and use this offset time')
parser.add_argument('--start-time', type=float, help='fast forward to this flight log time before begining movie render.')
parser.add_argument('--ground', type=float, help='ground altitude in meters')
parser.add_argument('--plot', action='store_true', help='Plot stuff at the end of the run')
args = parser.parse_args()

r2d = 180.0 / math.pi
counter = 0

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

if args.resample_hz <= 0.001:
    print("Resample rate (hz) needs to be greater than zero.")
    quit()
    
# pathname work
abspath = os.path.abspath(args.movie)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.movie)
movie_log = filename + ".csv"
camera_config = dirname + "/camera.json"

if 'recalibrate' in args:
    recal_file = args.recalibrate
else:
    recal_file = None
data, flight_format = flight_loader.load(args.flight, recal_file)
print("imu records:", len(data['imu']))
print("gps records:", len(data['gps']))
if 'air' in data:
    print("airdata records:", len(data['air']))
print("filter records:", len(data['filter']))
if 'pilot' in data:
    print("pilot records:", len(data['pilot']))
if 'act' in data:
    print("act records:", len(data['act']))
if len(data['imu']) == 0 and len(data['gps']) == 0:
    print("not enough data loaded to continue.")
    quit()

interp = flight_interp.FlightInterpolate()
interp.build(data)
    
time_shift, flight_min, flight_max = \
    correlate.sync_clocks(data, interp, movie_log, hz=args.resample_hz,
                          cam_mount=args.cam_mount,
                          force_time_shift=args.time_shift, plot=args.plot)

# quick estimate ground elevation
sum = 0.0
count = 0
for f in data['filter']:
    if interp.air_speed(f.time) < 5.0:
        sum += f.alt
        count += 1
if count > 0:
    ground_m = sum / float(count)
else:
    ground_m = data['filter'][0].alt
print("ground est:", ground_m)

if args.movie:
    metadata = skvideo.io.ffprobe(args.movie)
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
    print("Opening ", args.movie)
    reader = skvideo.io.FFmpegReader(args.movie, inputdict={}, outputdict={})

    last_time = 0.0
    abspath = os.path.abspath(args.movie)
    basename, ext = os.path.splitext(abspath)
    dirname = os.path.dirname(abspath)
    meta = dirname + "/image-metadata.txt"
    f = open(meta, 'w')
    print("writing meta data to", meta)
    
    for frame in reader.nextFrame():
        frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
        time = float(counter) / fps + time_shift
        print("frame: ", counter, "%.3f" % time, 'time shift:', time_shift)

        counter += 1
        if args.start_time and time < args.start_time:
            continue
        agl = interp.gps_alt(time) - ground_m
        if agl < 20.0:
            continue
        roll_deg = interp.filter_phi(time) * r2d
        pitch_deg = interp.filter_the(time) * r2d
        psix = interp.filter_psix(time)
        psiy = interp.filter_psiy(time)
        yaw_deg = math.atan2(psiy, psix) * r2d
        while yaw_deg < 0:
            yaw_deg += 360
        while yaw_deg > 360:
            yaw_deg -= 360
        if time >= last_time + args.interval:
            last_time = time
            file = basename + "-%06d" % counter + ".jpg"
            cv2.imwrite(file, frame)
            # geotag the image
            exif = pyexiv2.ImageMetadata(file)
            exif.read()
            lat_deg = float(interp.gps_lat(time))
            lon_deg = float(interp.gps_lon(time))
            altitude = float(interp.gps_alt(time))
            print(lat_deg, lon_deg, altitude)
            GPS = 'Exif.GPSInfo.GPS'
            exif[GPS + 'AltitudeRef']  = '0' if altitude >= 0 else '1'
            exif[GPS + 'Altitude']     = Fraction(altitude)
            exif[GPS + 'Latitude']     = decimal_to_dms(lat_deg)
            exif[GPS + 'LatitudeRef']  = 'N' if lat_deg >= 0 else 'S'
            exif[GPS + 'Longitude']    = decimal_to_dms(lon_deg)
            exif[GPS + 'LongitudeRef'] = 'E' if lon_deg >= 0 else 'W'
            exif[GPS + 'MapDatum']     = 'WGS-84'
            exif.write()
            head, tail = os.path.split(file)
            f.write("%s,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f,%.2f\n" % (tail, interp.gps_lat(time), interp.gps_lon(time), interp.gps_alt(time), yaw_deg, pitch_deg, roll_deg,time))
    f.close()
