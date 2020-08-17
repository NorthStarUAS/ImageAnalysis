#!/usr/bin/env python3

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
from props import PropertyNode
import props_json

parser = argparse.ArgumentParser(description='extract and geotag dji movie frames.')
parser.add_argument('--video', required=True, help='input video')
parser.add_argument('--camera', help='select camera calibration file')
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
match_ratio = 0.75
scale = 0.4
filter_method = 'homography'
tol = 3.0
overlap = 0.25

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

# find affine transform between matching keypoints in pixel
# coordinate space.  fullAffine=True means unconstrained to
# include best warp/shear.  fullAffine=False means limit the
# matrix to only best rotation, translation, and scale.
def findAffine(src, dst, fullAffine=False):
    affine_minpts = 7
    #print("src:", src)
    #print("dst:", dst)
    if len(src) >= affine_minpts:
        # affine = cv2.estimateRigidTransform(np.array([src]), np.array([dst]), fullAffine)
        affine, status = \
            cv2.estimateAffinePartial2D(np.array([src]).astype(np.float32),
                                        np.array([dst]).astype(np.float32))
    else:
        affine = None
    #print str(affine)
    return affine

def decomposeAffine(affine):
    if affine is None:
        return (0.0, 0.0, 0.0, 1.0, 1.0)

    tx = affine[0][2]
    ty = affine[1][2]

    a = affine[0][0]
    b = affine[0][1]
    c = affine[1][0]
    d = affine[1][1]

    sx = math.sqrt( a*a + b*b )
    if a < 0.0:
        sx = -sx
    sy = math.sqrt( c*c + d*d )
    if d < 0.0:
        sy = -sy

    rotate_deg = math.atan2(-b,a) * 180.0/math.pi
    if rotate_deg < -180.0:
        rotate_deg += 360.0
    if rotate_deg > 180.0:
        rotate_deg -= 360.0
    return (rotate_deg, tx, ty, sx, sy)

def filterMatches(kp1, kp2, matches):
    mkp1, mkp2 = [], []
    idx_pairs = []
    used = np.zeros(len(kp2), np.bool_)
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * match_ratio:
            #print " dist[0] = %d  dist[1] = %d" % (m[0].distance, m[1].distance)
            m = m[0]
            # FIXME: ignore the bottom section of movie for feature detection
            #if kp1[m.queryIdx].pt[1] > h*0.75:
            #    continue
            if not used[m.trainIdx]:
                used[m.trainIdx] = True
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
                idx_pairs.append( (m.queryIdx, m.trainIdx) )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs, idx_pairs, mkp1

def filterFeatures(p1, p2, K, method):
    inliers = 0
    total = len(p1)
    space = ""
    status = []
    M = None
    if len(p1) < 7:
        # not enough points
        return None, np.zeros(total), [], []
    if method == 'homography':
        M, status = cv2.findHomography(p1, p2, cv2.LMEDS, tol)
    elif method == 'fundamental':
        M, status = cv2.findFundamentalMat(p1, p2, cv2.LMEDS, tol)
    elif method == 'essential':
        M, status = cv2.findEssentialMat(p1, p2, K, cv2.LMEDS, threshold=tol)
    elif method == 'none':
        M = None
        status = np.ones(total)
    newp1 = []
    newp2 = []
    for i, flag in enumerate(status):
        if flag:
            newp1.append(p1[i])
            newp2.append(p2[i])
    p1 = np.float32(newp1)
    p2 = np.float32(newp2)
    inliers = np.sum(status)
    total = len(status)
    #print '%s%d / %d  inliers/matched' % (space, np.sum(status), len(status))
    return M, status, np.float32(newp1), np.float32(newp2)


# pathname work
abspath = os.path.abspath(args.video)
basename, ext = os.path.splitext(abspath)
srtname = basename + ".srt"
dirname = basename + "_frames"
print("basename:", basename)
print("srtname:", srtname)
print("dirname:", dirname)

local_config = os.path.join(dirname, "camera.json")
config = PropertyNode()
if args.camera:
    # seed the camera calibration and distortion coefficients from a
    # known camera config
    print('Setting camera config from:', args.camera)
    props_json.load(args.camera, config)
    config.setString('name', args.camera)
    props_json.save(local_config, config)
elif os.path.exists(local_config):
    # load local config file if it exists
    props_json.load(local_config, config)
K_list = []
for i in range(9):
    K_list.append( config.getFloatEnum('K', i) )
K = np.copy(np.array(K_list)).reshape(3,3)
dist = []
for i in range(5):
    dist.append( config.getFloatEnum("dist_coeffs", i) )

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

# setup feature detection
detector = cv2.SIFT_create(nfeatures=1000)
FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6
flann_params = { 'algorithm': FLANN_INDEX_KDTREE,
                 'trees': 5 }
matcher = cv2.FlannBasedMatcher(flann_params, {}) # bug : need to pass empty dict (#1329)

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
kp_list_ref = []
des_list_ref = []
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
    
    write_frame = False

    # by distance camera has moved
    (c1, c2, dist_m) = wgs84.geo_inverse(lat_deg, lon_deg, last_lat, last_lon)
    print("dist:", dist_m)
    #if time >= last_time + args.interval and dist_m >= args.distance:
    if dist_m >= args.distance:
        write_frame = True
        
    # by visual overlap
    method = cv2.INTER_AREA
    frame_scale = cv2.resize(frame, (0,0), fx=scale, fy=scale,
                             interpolation=method)
    cv2.imshow('frame', frame_scale)
    gray = cv2.cvtColor(frame_scale, cv2.COLOR_BGR2GRAY)
    (h, w) = gray.shape
    kp_list = detector.detect(gray)
    kp_list, des_list = detector.compute(gray, kp_list)
    if not (des_list_ref is None) and not (des_list is None) and len(des_list_ref) and len(des_list):
        matches = matcher.knnMatch(des_list, trainDescriptors=des_list_ref, k=2)
        p1, p2, kp_pairs, idx_pairs, mkp1 = filterMatches(kp_list, kp_list_ref, matches)
        M, status, newp1, newp2 = filterFeatures(p1, p2, K, filter_method)
        filtered = []
        for i, flag in enumerate(status):
            if flag:
                filtered.append(mkp1[i])
        affine = findAffine(p2, p1, fullAffine=False)
        if affine is None:
            write_frame = True
        else:
            (rot, tx, ty, sx, sy) = decomposeAffine(affine)
            xperc = abs(tx) / w
            yperc = abs(ty) / h
            perc = math.sqrt(xperc*xperc + yperc*yperc)
            print("pixel dist:", tx, ty, "%.1f%% %.1f%%" % (xperc*100, yperc*100))
            if perc >= overlap:
                write_frame = True
    else:
        # first frame
        write_frame = True
    cv2.waitKey(1)
    
    if write_frame:
        print("WRITE FRAME")
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
        # by distance
        last_lat = lat_deg
        last_lon = lon_deg
        # by time
        last_time = time
        # by overlap
        kp_list_ref = kp_list
        des_list_ref = des_list

f.close()
