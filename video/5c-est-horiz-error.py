#!/usr/bin/env python3

# use feature-based motion (affine) roll, pitch, yaw rates.  Compare
# rates in camera space vs. imu space and try to find an optimal
# transform to minimize the idfference between them.

import argparse
import csv
from math import atan2, pi
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import interpolate  # strait up linear interpolation, nothing fancy
import scipy.signal as signal

import sys
sys.path.append('../scripts')
from lib import transformations

from aurauas_flightdata import flight_loader, flight_interp

import camera
import correlate
from horiz_data import HorizonData

parser = argparse.ArgumentParser(description='estimate horizon error based on work from earlier scripts.')
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

smooth_cutoff_hz = 10
#method = "horiz"
method = "rates"

# pathname work
abspath = os.path.abspath(args.video)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.video)
horiz_log = filename + "_horiz.csv"
ekf_error = filename + "_error.csv"
local_config = dirname + "/camera.json"

# load the camera config (we will modify the mounting offset later)
camera = camera.VirtualCamera()
camera.load(None, local_config)
cam_yaw, cam_pitch, cam_roll = camera.get_ypr()
K = camera.get_K()
dist = camera.get_dist()
print('Camera:', camera.get_name())

# load the flight data
flight_data, flight_format = flight_loader.load(args.flight)
print("imu records:", len(flight_data['imu']))
print("gps records:", len(flight_data['gps']))
if 'air' in flight_data:
    print("airdata records:", len(flight_data['air']))
print("filter records:", len(flight_data['filter']))
if 'pilot' in flight_data:
    print("pilot records:", len(flight_data['pilot']))
if 'act' in flight_data:
    print("act records:", len(flight_data['act']))
if len(flight_data['imu']) == 0 and len(flight_data['filter']) == 0:
    print("not enough data loaded to continue.")
    quit()

interp = flight_interp.InterpolationGroup(flight_data)
iter = flight_interp.IterateGroup(flight_data)

# for convenience
hz = args.resample_hz
r2d = 180.0 / pi
d2r = pi / 180.0

# load horizon log data (derived from video)
horiz_data = HorizonData()
horiz_data.load(horiz_log)
horiz_data.smooth(smooth_cutoff_hz)
horiz_data.make_interp()
if args.plot:
    horiz_data.plot()
horiz_interp = horiz_data.resample(args.resample_hz)

plt.figure()
# plt.plot(data[:,0], data[:,1], label="video roll")
# plt.plot(data[:,0], data[:,3], label="ekf roll")
# plt.legend()
# plt.show()

# smooth ekf attitude estimate
# prep to smooth flight data (noisy data can create tiny local minima
# that the optimizer can get stuck within.
ekf = pd.DataFrame(flight_data['filter'])
ekf.set_index('time', inplace=True, drop=False)
plt.plot(ekf['phi'], label='orig')
ekf_min = ekf['time'].iat[0]
ekf_max = ekf['time'].iat[-1]
ekf_count = len(ekf)
ekf_fs =  int(round((ekf_count / (ekf_max - ekf_min))))
print("ekf fs:", ekf_fs)
b, a = signal.butter(2, 1.0, fs=ekf_fs)
ekf['phi'] = signal.filtfilt(b, a, ekf['phi'])
plt.plot(ekf['phi'], label='smooth')
ekf['the'] = signal.filtfilt(b, a, ekf['the'])
ekf['psix'] = signal.filtfilt(b, a, ekf['psix'])
ekf['psiy'] = signal.filtfilt(b, a, ekf['psiy'])
plt.plot(horiz_data.data['camera roll (deg)']*d2r, label='video phi')
plt.legend()
#plt.show()

# resample flight data
imu_min = flight_data['imu'][0]['time']
imu_max = flight_data['imu'][-1]['time']
print("flight range = %.3f - %.3f (%.3f)" % (imu_min, imu_max,
                                             imu_max-imu_min))
flight_interp = []
if args.cam_mount == 'forward' or args.cam_mount == 'rear':
    # forward/rear facing camera
    p_interp = interp.group['imu'].interp['p']
    q_interp = interp.group['imu'].interp['q']
elif args.cam_mount == 'left' or args.cam_mount == 'right':
    # it might be interesting to support an out-the-wing view
    print("Not currently supported camera orientation, sorry!")
    quit()
flight_len = imu_max - imu_min
phi_interp = interpolate.interp1d(ekf['time'], ekf['phi'], bounds_error=False, fill_value=0.0)
the_interp = interpolate.interp1d(ekf['time'], ekf['the'], bounds_error=False, fill_value=0.0)
psix_interp = interpolate.interp1d(ekf['time'], ekf['psix'], bounds_error=False, fill_value=0.0)
psiy_interp = interpolate.interp1d(ekf['time'], ekf['psiy'], bounds_error=False, fill_value=0.0)
alt_interp = interp.group['filter'].interp['alt']

for x in np.linspace(imu_min, imu_max, int(round(flight_len*hz))):
    flight_interp.append( [x, p_interp(x), q_interp(x),
                           phi_interp(x), the_interp(x),
                           psix_interp(x), psiy_interp(x)] )
print("flight len:", len(flight_interp))

# find the time correlation of video vs flight data
time_shift = \
    correlate.sync_horizon(flight_data, flight_interp,
                           horiz_data.data, horiz_interp, horiz_data.span_sec,
                           method=method, hz=hz, cam_mount=args.cam_mount,
                           force_time_shift=args.time_shift, plot=args.plot)

# presample datas to save work in the error function
tmin = np.amax( [horiz_data.tmin + time_shift, imu_min ] )
tmax = np.amin( [horiz_data.tmax + time_shift, imu_max ] )
tlen = tmax - tmin
print("overlap range (flight sec):", tmin, " - ", tmax)

# restructure ekf data
ekf = pd.DataFrame(flight_data['filter'])
ekf.set_index('time', inplace=True, drop=False)
phi_interp = interpolate.interp1d(ekf['time'], ekf['phi'], bounds_error=False, fill_value=0.0)
the_interp = interpolate.interp1d(ekf['time'], ekf['the'], bounds_error=False, fill_value=0.0)
psix_interp = interpolate.interp1d(ekf['time'], ekf['psix'], bounds_error=False, fill_value=0.0)
psiy_interp = interpolate.interp1d(ekf['time'], ekf['psiy'], bounds_error=False, fill_value=0.0)

# but should we create the estimate for the whole flight?

# Scan altitude range so we can match the portion of the flight that
# is up and away.  This means the EKF will have had a better chance to
# converge, and the horizon detection should be getting a clear view.
min_alt = None
max_alt = None
for x in np.linspace(tmin, tmax, int(round(tlen*hz))):
    alt = alt_interp(x)
    if min_alt is None or alt < min_alt:
        min_alt = alt
    if max_alt is None or alt > max_alt:
        max_alt = alt
print("altitude range: %.1f - %.1f (m)" % (min_alt, max_alt))
if False and max_alt - min_alt > 30:
    alt_threshold = min_alt + (max_alt - min_alt) * 0.5
else:
    alt_threshold = min_alt
print("Altitude threshold: %.1f (m)" % alt_threshold)

data = []
for x in np.linspace(tmin, tmax, int(round(tlen*hz))):
    # horizon
    hphi, hthe, hp, hr = horiz_data.get_vals(x - time_shift)
    # flight data
    fphi = phi_interp(x)
    fthe = the_interp(x)
    psix = psix_interp(x)
    psiy = psiy_interp(x)
    fpsi = atan2(psiy, psix)
    alt = alt_interp(x)
    if alt >= alt_threshold:
        data.append( [x, hphi, hthe, fpsi, fthe, fphi] )
print("Data points passing altitude threshold:", len(data))

data = np.array(data)
plt.figure()
plt.plot(data[:,0], data[:,1]*r2d, label="video roll")
plt.plot(data[:,0], data[:,5]*r2d, label="ekf roll")
plt.xlabel("Flight time (secs)")
plt.ylabel("Degrees")
plt.legend()
plt.figure()
plt.plot(data[:,0], data[:,2]*r2d, label="video pitch")
plt.plot(data[:,0], data[:,4]*r2d, label="ekf pitch")
plt.xlabel("Flight time (secs)")
plt.ylabel("Degrees")
plt.legend()
plt.show()

def horiz_errorFunc():
    # compute error function using global data structures
    horiz_ned = [0, 0, 0]  # any value works here (as long as it's consistent
    result = []
    for r in data:
        camera.update_PROJ(horiz_ned, r[3], r[4], r[5]) # aircraft body attit
        #print("video:", r[1]*r2d, r[2]*r2d)
        roll, pitch = camera.find_horizon()
        #result.append( r[1] - (r[5] + xk[2]) )
        #result.append( r[2] - (r[4] + xk[1]) )
        if not roll is None:
            result.append( r[1] - roll )
            result.append( r[2] - pitch )
    return np.array(result)

print("Camera mount (ypr):", camera.get_ypr() )
result = horiz_errorFunc()
rollerr = result[::2]
pitcherr = result[1::2]

# write to file
print("Wriing ekf horizon estimate error to:", ekf_error)
csvfile = open(ekf_error, 'w')
fieldnames = [ 'flight time (sec)',
               'ekf roll error (rad)', 'ekf pitch error (rad)' ]
csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
csv_writer.writeheader()
for i in range(len(data)):
    row = { 'flight time (sec)': "%.4f" % data[i][0],
            'ekf roll error (rad)': "%.3f" % rollerr[i],
            'ekf pitch error (rad)': "%.3f" % pitcherr[i] }
    csv_writer.writerow(row)
csvfile.close()

print("Plotting final result...")
print(len(result), len(data), len(data[::2]), len(rollerr), len(pitcherr))
data = np.array(data)
plt.figure()
plt.plot(data[:,0], rollerr*r2d, label="roll error")
plt.plot(data[:,0], pitcherr*r2d, label="pitch error")
plt.ylabel("Angle error (deg)")
plt.xlabel("Flight time (sec)")
plt.grid()
plt.legend()
plt.show()
