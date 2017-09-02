#!/usr/bin/python

# find our custom built opencv first
import sys
sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")

import argparse
import cv2
import math
import fractions
from matplotlib import pyplot as plt 
import numpy as np
import os
import pyexiv2
import re
#from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate # strait up linear interpolation, nothing fancy

from aurauas.flightdata import flight_loader, flight_interp

parser = argparse.ArgumentParser(description='correlate movie data to flight data.')
parser.add_argument('--flight', help='load specified aura flight log')
parser.add_argument('--aura-flight', help='load specified aura flight log')
parser.add_argument('--px4-sdlog2', help='load specified px4 sdlog2 (csv) flight log')
parser.add_argument('--px4-ulog', help='load specified px4 ulog (csv) base path')
parser.add_argument('--umn-flight', help='load specified .mat flight log')
parser.add_argument('--sentera-flight', help='load specified sentera flight log')
parser.add_argument('--sentera2-flight', help='load specified sentera2 flight log')
parser.add_argument('--movie', required=False, help='original movie if extracting frames')
parser.add_argument('--select-cam', type=int, help='select camera calibration')
parser.add_argument('--interval', type=float, default=1.0, help='capture interval')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--resample-hz', type=float, default=30.0, help='resample rate (hz)')
parser.add_argument('--stop-count', type=int, default=1, help='how many non-frames to absorb before we decide the movie is over')
parser.add_argument('--ground', type=float, help='ground altitude in meters')
parser.add_argument('--plot', action='store_true', help='Plot stuff at the end of the run')
args = parser.parse_args()

r2d = 180.0 / math.pi
counter = 0
stop_count = 0

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
    print "Resample rate (hz) needs to be greater than zero."
    quit()
    
# pathname work
abspath = os.path.abspath(args.movie)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.movie)
movie_log = filename + ".csv"
camera_config = dirname + "/camera.json"

# load movie log
movie = []
with open(movie_log, 'rb') as f:
    for line in f:
        movie.append( re.split('[,\s]+', line.rstrip()) )

if args.flight:
    loader = 'aura'
    path = args.flight
elif args.aura_flight:
    loader = 'aura'
    path = args.aura_flight
elif args.px4_sdlog2:
    loader = 'px4_sdlog2'
    path = args.px4_sdlog2
elif args.px4_ulog:
    loader = 'px4_ulog'
    path = args.px4_ulog
elif args.sentera_flight:
    loader = 'sentera1'
    path = args.sentera_flight
elif args.sentera2_flight:
    loader = 'sentera2'
    path = args.sentera2_flight
elif args.umn_flight:
    loader = 'umn1'
    path = args.umn_flight
else:
    loader = None
    path = None
if 'recalibrate' in args:
    recal_file = args.recalibrate
else:
    recal_file = None
data = flight_loader.load(loader, path, recal_file)
print "imu records:", len(data['imu'])
print "gps records:", len(data['gps'])
if 'air' in data:
    print "airdata records:", len(data['air'])
print "filter records:", len(data['filter'])
if 'pilot' in data:
    print "pilot records:", len(data['pilot'])
if 'act' in data:
    print "act records:", len(data['act'])
if len(data['imu']) == 0 and len(data['gps']) == 0:
    print "not enough data loaded to continue."
    quit()

interp = flight_interp.FlightInterpolate()
interp.build(data)
    
# set approximate camera orienation (front, down, and rear supported)
cam_facing = 'down'

# resample movie data
movie = np.array(movie, dtype=float)
movie_interp = []
x = movie[:,0]
movie_spl_roll = interpolate.interp1d(x, movie[:,2], bounds_error=False, fill_value=0.0)
movie_spl_pitch = interpolate.interp1d(x, movie[:,3], bounds_error=False, fill_value=0.0)
movie_spl_yaw = interpolate.interp1d(x, movie[:,4], bounds_error=False, fill_value=0.0)
xmin = x.min()
xmax = x.max()
print "movie range = %.3f - %.3f (%.3f)" % (xmin, xmax, xmax-xmin)
time = xmax - xmin
for x in np.linspace(xmin, xmax, time*args.resample_hz):
    if cam_facing == 'front' or cam_facing == 'down':
        movie_interp.append( [x, movie_spl_roll(x)] )
    else:
        movie_interp.append( [x, -movie_spl_roll(x)] )
print "movie len:", len(movie_interp)

# resample flight data
flight_interp = []
if cam_facing == 'front' or cam_facing == 'rear':
    y_spline = interp.imu_p     # front/rear facing camera
else:
    y_spline = interp.imu_r     # down facing camera

# run correlation over filter time span
x = interp.imu_time
xmin = x.min()
xmax = x.max()
print "flight range = %.3f - %.3f (%.3f)" % (xmin, xmax, xmax-xmin)
time = xmax - xmin
for x in np.linspace(xmin, xmax, time*args.resample_hz):
    flight_interp.append( [x, y_spline(x)] )
print "flight len:", len(flight_interp)

# compute best correlation between movie and flight data logs
movie_interp = np.array(movie_interp, dtype=float)
flight_interp = np.array(flight_interp, dtype=float)
ycorr = np.correlate(movie_interp[:,1], flight_interp[:,1], mode='full')

# display some stats/info
max_index = np.argmax(ycorr)
print "max index:", max_index
shift = np.argmax(ycorr) - len(flight_interp)
print "shift (pos):", shift
start_diff = flight_interp[0][0] - movie_interp[0][0]
print "start time diff:", start_diff
time_shift = start_diff - (shift/args.resample_hz)
print "movie time shift:", time_shift

# estimate  tx, ty vs. r, q multiplier
tmin = np.amax( [np.amin(movie_interp[:,0]) + time_shift,
                np.amin(flight_interp[:,0]) ] )
tmax = np.amin( [np.amax(movie_interp[:,0]) + time_shift,
                np.amax(flight_interp[:,0]) ] )
print "overlap range (flight sec):", tmin, " - ", tmax

mqsum = 0.0
fqsum = 0.0
mrsum = 0.0
frsum = 0.0
count = 0
qratio = 1.0
for x in np.linspace(tmin, tmax, time*args.resample_hz):
    mqsum += abs(movie_spl_pitch(x-time_shift))
    mrsum += abs(movie_spl_yaw(x-time_shift))
    fqsum += abs(interp.imu_q(x))
    frsum += abs(interp.imu_r(x))
if fqsum > 0.001:
    qratio = mqsum / fqsum
if mrsum > 0.001:
    rratio = -mrsum / frsum
print "pitch ratio:", qratio
print "yaw ratio:", rratio

ground_m = data['gps'][0].alt
if args.ground:
    ground_m = args.ground
print 'Ground elevation:', ground_m

if args.movie:
    # extract frames
    print "Opening ", args.movie
    try:
        capture = cv2.VideoCapture(args.movie)
    except:
        print "error opening video"

    capture.read()
    counter += 1
    print "ok reading first frame"

    fps = capture.get(cv2.CAP_PROP_FPS)
    print "fps = %.2f" % fps
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    last_time = 0.0
    abspath = os.path.abspath(args.movie)
    basename, ext = os.path.splitext(abspath)
    dirname = os.path.dirname(abspath)
    meta = dirname + "/image-metadata.txt"
    print "writing meta data to", meta
    f = open(meta, 'wb')
    while True:
        ret, frame = capture.read()
        if not ret:
            # no frame
            stop_count += 1
            print "no more frames:", stop_count
            if stop_count > args.stop_count:
                break
        else:
            stop_count = 0
            
        if frame == None:
            print "Skipping bad frame ..."
            continue
        time = float(counter) / fps + time_shift
        print "frame: %d %.3f" % (counter, time)
        counter += 1
        if time < tmin or time > tmax:
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
            print lat_deg, lon_deg, altitude
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
            f.write("%s,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f\n" % (tail, interp.gps_lat(time), interp.gps_lon(time), interp.gps_alt(time), yaw_deg, pitch_deg, roll_deg))
    f.close()

if args.plot:
    # reformat the data
    flight_imu = []
    for imu in data['imu']:
        flight_imu.append([ imu.time, imu.p, imu.q, imu.r ])
    flight_imu = np.array(flight_imu)
    
    # plot the data ...
    plt.figure(1)
    plt.ylabel('roll rate (deg per sec)')
    plt.xlabel('flight time (sec)')
    plt.plot(movie[:,0] + time_shift, movie[:,2]*r2d, label='estimate from flight movie')
    if cam_facing == 'front':
        # front facing:
        plt.plot(flight_imu[:,0], flight_imu[:,1]*r2d, label='flight data log')
    elif cam_facing == 'down':
        # down facing:
        plt.plot(flight_imu[:,0], flight_imu[:,3]*r2d, label='flight data log')
    plt.legend()
    
    plt.figure(2)
    plt.plot(ycorr)

    plt.figure(3)
    plt.ylabel('pitch rate (deg per sec)')
    plt.xlabel('flight time (sec)')
    plt.plot(movie[:,0] + time_shift, (movie[:,3]/qratio)*r2d, label='estimate from flight movie')
    if cam_facing == 'front':
        # front facing:
        plt.plot(flight_imu[:,0], flight_imu[:,2]*r2d, label='flight data log')
    elif cam_facing == 'down':
        # down facing:
        plt.plot(flight_imu[:,0], flight_imu[:,2]*r2d, label='flight data log')
    plt.legend()

    plt.figure(4)
    plt.ylabel('yaw rate (deg per sec)')
    plt.xlabel('flight time (sec)')
    plt.plot(movie[:,0] + time_shift, (movie[:,4]/rratio)*r2d, label='estimate from flight movie')
    if cam_facing == 'front':
        # front facing:
        plt.plot(flight_imu[:,0], flight_imu[:,3]*r2d, label='flight data log')
    elif cam_facing == 'down':
        # down facing:
        plt.plot(flight_imu[:,0], flight_imu[:,1]*r2d, label='flight data log')
    plt.legend()

    plt.show()
