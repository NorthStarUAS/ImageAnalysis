#!/usr/bin/env python3

# use the horizon angle data (in camera space) along with estimated
# roll/pitch arates to correlate video time to flight data log time.
# Then use an optimizer to estimate the best fitting roll, pitch, yaw
# offsets for the camera mount relative to the IMU/EKF solution.

import argparse
import os

from aurauas_flightdata import flight_loader, flight_interp

import camera
import correlate

parser = argparse.ArgumentParser(description='correlate movie data to flight data.')
parser.add_argument('--flight', required=True, help='load specified aura flight log')
parser.add_argument('--video', required=True, help='original video')
parser.add_argument('--cam-mount', choices=['forward', 'down', 'rear'],
                    default='forward',
                    help='approximate camera mounting orientation')
parser.add_argument('--resample-hz', type=float, default=60.0,
                    help='resample rate (hz)')
parser.add_argument('--time-shift', type=float,
                    help='skip autocorrelation and use this offset time')
parser.add_argument('--plot', action='store_true',
                    help='Plot stuff at the end of the run')
args = parser.parse_args()

# pathname work
abspath = os.path.abspath(args.video)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.video)
horiz_log = filename + "_horiz.csv"
local_config = dirname + "/camera.json"

# load the camera config (we will modify the mounting offset later)
camera = camera.VirtualCamera()
camera.load(None, local_config)
cam_yaw, cam_pitch, cam_roll = camera.get_ypr()
K = camera.get_K()
dist = camera.get_dist()
print('Camera:', camera.get_name())

# load the flight data
data, flight_format = flight_loader.load(args.flight)
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

interp = flight_interp.InterpolationGroup(data)
iter = flight_interp.IterateGroup(data)
time_shift, flight_min, flight_max = \
    correlate.sync_horizon(data, interp, horiz_log, hz=args.resample_hz,
                           cam_mount=args.cam_mount,
                           force_time_shift=args.time_shift, plot=args.plot)
