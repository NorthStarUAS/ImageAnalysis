#!/usr/bin/env python3

# use the horizon angle data (in camera space) along with estimated
# roll/pitch arates to correlate video time to flight data log time.
# Then use an optimizer to estimate the best fitting roll, pitch, yaw
# offsets for the camera mount relative to the IMU/EKF solution.

import argparse
import numpy as np
import os
import pandas as pd
from scipy import interpolate  # strait up linear interpolation, nothing fancy

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

# for convenience
hz = args.resample_hz

# load horizon log data (derived from video)
#
# frame,video time,camera roll (deg),camera pitch (deg),
# roll rate (rad/sec),pitch rate (rad/sec)
horiz_data = pd.read_csv(horiz_log)
horiz_data.set_index('video time', inplace=True, drop=False)

# resample horizon data
horiz_interp = []
horiz_roll = interpolate.interp1d(horiz_data['video time'],
                                  horiz_data['roll rate (rad/sec)'],
                                  bounds_error=False, fill_value=0.0)
xmin = horiz_data['video time'].min()
xmax = horiz_data['video time'].max()
print("video range = %.3f - %.3f (%.3f)" % (xmin, xmax, xmax-xmin))
horiz_len = xmax - xmin
for x in np.linspace(xmin, xmax, int(round(horiz_len*hz))):
    if args.cam_mount == 'forward' or args.cam_mount == 'down':
        horiz_interp.append( [x, horiz_roll(x)] )
    else:
        # down, but makes no sense in this context!
        horiz_interp.append( [x, -horiz_roll(x)] )
print("horizon data len:", len(horiz_interp))

# resample flight data
flight_min = data['imu'][0]['time']
flight_max = data['imu'][-1]['time']
print("flight range = %.3f - %.3f (%.3f)" % (flight_min, flight_max,
                                             flight_max-flight_min))
flight_interp = []
if args.cam_mount == 'forward' or args.cam_mount == 'rear':
    # forward/rear facing camera
    y_interp = interp.group['imu'].interp['p']
elif args.cam_mount == 'left' or args.cam_mount == 'right':
    # it might be interesting to support an out-the-wing view
    print("Not currently supported camera orientation, sorry!")
    quit()
else:
    # down facing camera (which makes no sense for a horizon detector!)    
    y_interp = interp.group['imu'].interp['r']
flight_len = flight_max - flight_min
for x in np.linspace(flight_min, flight_max, int(round(flight_len*hz))):
    flight_interp.append( [x, y_interp(x)] )
print("flight len:", len(flight_interp))

# find the time correlation of video vs flight data
time_shift = \
    correlate.sync_horizon(data, flight_interp,
                           horiz_data, horiz_interp, horiz_len,
                           hz=hz, cam_mount=args.cam_mount,
                           force_time_shift=args.time_shift, plot=args.plot)
