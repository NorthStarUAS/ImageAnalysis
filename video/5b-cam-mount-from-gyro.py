#!/usr/bin/env python3

# use horizon based roll/pitch rates (and maybe motion based yaw
# rates.)  Compare rates in camera space vs. imu space and try to find
# an optimal transform to minimize the idfference between them.

import argparse
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import interpolate  # strait up linear interpolation, nothing fancy

from transformations import euler_matrix

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

smooth_cutoff_hz = 10

# pathname work
abspath = os.path.abspath(args.video)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.video)
horiz_rates = filename + "_horiz.csv"
video_rates = filename + "_rates.csv"
local_config = dirname + "/camera.json"

# load the camera config (we will modify the mounting offset later)
camera = camera.VirtualCamera()
camera.load(None, local_config)
cam_yaw, cam_pitch, cam_roll = camera.get_ypr()
K = camera.get_K()
IK = camera.get_IK()
dist = camera.get_dist()
cu = K[0,2]
cv = K[1,2]
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
if len(flight_data['imu']) == 0 and len(flight_data['gps']) == 0:
    print("not enough data loaded to continue.")
    quit()

interp = flight_interp.InterpolationGroup(flight_data)
iter = flight_interp.IterateGroup(flight_data)

# for convenience
hz = args.resample_hz
r2d = 180.0 / math.pi
d2r = math.pi / 180.0

horiz_data = pd.read_csv(horiz_rates)
horiz_data.set_index('video time', inplace=True, drop=False)
xmin = horiz_data['video time'].min()
xmax = horiz_data['video time'].max()
print("horiz range:", xmin, xmax)

# load camera rotation rate data (derived from feature matching video
# frames)
#
# frame, video time, p (rad/sec), q (rad/sec), r (rad/sec)
video_data = pd.read_csv(video_rates)
video_data.set_index('video time', inplace=True, drop=False)
xmin = video_data['video time'].min()
xmax = video_data['video time'].max()
video_count = len(video_data['video time'])
print("number of video records:", video_count)
video_fs = int(round((video_count / (xmax - xmin))))
print("video fs:", video_fs)

plt.figure()
plt.plot(horiz_data['roll rate (rad/sec)'], label="horizon-based p")
plt.plot(video_data['p (rad/sec)'], label="feature-based p")
plt.plot(horiz_data['pitch rate (rad/sec)'], label="horizon-based q")
plt.plot(video_data['q (rad/sec)'], label="feature-based q")
plt.plot(video_data['r (rad/sec)'], label="feature-based r")
plt.legend()
plt.show()

# smooth the video data
import scipy.signal as signal
b, a = signal.butter(2, smooth_cutoff_hz, fs=video_fs)
horiz_data['roll rate (rad/sec)'] = signal.filtfilt(b, a, horiz_data['roll rate (rad/sec)'])
horiz_data['pitch rate (rad/sec)'] = signal.filtfilt(b, a, horiz_data['pitch rate (rad/sec)'])
video_data['p (rad/sec)'] = signal.filtfilt(b, a, video_data['p (rad/sec)'])
video_data['q (rad/sec)'] = signal.filtfilt(b, a, video_data['q (rad/sec)'])
video_data['r (rad/sec)'] = signal.filtfilt(b, a, video_data['r (rad/sec)'])

plt.figure()
plt.plot(horiz_data['roll rate (rad/sec)'], label="horizon-based p")
plt.plot(video_data['p (rad/sec)'], label="feature-based p")
plt.plot(horiz_data['pitch rate (rad/sec)'], label="horizon-based q")
plt.plot(video_data['q (rad/sec)'], label="feature-based q")
plt.plot(video_data['r (rad/sec)'], label="feature-based r")
plt.legend()
plt.show()

# resample horizon data
video_interp = []
#video_p = interpolate.interp1d(horiz_data['video time'],
#                               horiz_data['roll rate (rad/sec)'],
#                               bounds_error=False, fill_value=0.0)
video_p = interpolate.interp1d(video_data['video time'],
                               video_data['p (rad/sec)'],
                               bounds_error=False, fill_value=0.0)
#video_q = interpolate.interp1d(horiz_data['video time'],
#                               horiz_data['pitch rate (rad/sec)'],
#                               bounds_error=False, fill_value=0.0)
video_q = interpolate.interp1d(video_data['video time'],
                               video_data['q (rad/sec)'],
                               bounds_error=False, fill_value=0.0)
video_r = interpolate.interp1d(video_data['video time'],
                               video_data['r (rad/sec)'],
                               bounds_error=False, fill_value=0.0)
xmin = video_data['video time'].min()
xmax = video_data['video time'].max()
print("video range = %.3f - %.3f (%.3f)" % (xmin, xmax, xmax-xmin))
horiz_len = xmax - xmin
for x in np.linspace(xmin, xmax, int(round(horiz_len*hz))):
    if args.cam_mount == 'forward' or args.cam_mount == 'down':
        video_interp.append( [x, video_p(x), video_q(x), video_r(x)] )
print("video data len:", len(video_interp))

plt.figure()
# plt.plot(data[:,0], data[:,1], label="video roll")
# plt.plot(data[:,0], data[:,3], label="ekf roll")
# plt.legend()
# plt.show()

# smooth imu gyro data
# prep to smooth flight data (noisy data can create tiny local minima
# that the optimizer can get stuck within.
imu = pd.DataFrame(flight_data['imu'])
imu.set_index('time', inplace=True, drop=False)
plt.plot(imu['p'], label='orig')
imu_min = imu['time'].iat[0]
imu_max = imu['time'].iat[-1]
imu_count = len(imu)
imu_fs =  int(round((imu_count / (imu_max - imu_min))))
print("imu fs:", imu_fs)
b, a = signal.butter(2, smooth_cutoff_hz, fs=imu_fs)
imu['p'] = signal.filtfilt(b, a, imu['p'])
plt.plot(imu['p'], label='smooth')
imu['q'] = signal.filtfilt(b, a, imu['q'])
imu['r'] = signal.filtfilt(b, a, imu['r'])
plt.plot(video_data['p (rad/sec)'], label='video (smooth)')
plt.legend()
#plt.show()

# resample (now smoothed) flight data
print("flight range = %.3f - %.3f (%.3f)" % (imu_min, imu_max,
                                             imu_max-imu_min))
flight_interp = []
# if args.cam_mount == 'forward' or args.cam_mount == 'rear':
#     # forward/rear facing camera
#     p_interp = interp.group['imu'].interp['p']
#     q_interp = interp.group['imu'].interp['q']
#     r_interp = interp.group['imu'].interp['r']
# elif args.cam_mount == 'left' or args.cam_mount == 'right':
#     # it might be interesting to support an out-the-wing view
#     print("Not currently supported camera orientation, sorry!")
#     quit()
flight_len = imu_max - imu_min
p_interp = interpolate.interp1d(imu['time'], imu['p'], bounds_error=False, fill_value=0.0)
q_interp = interpolate.interp1d(imu['time'], imu['q'], bounds_error=False, fill_value=0.0)
r_interp = interpolate.interp1d(imu['time'], imu['r'], bounds_error=False, fill_value=0.0)
alt_interp = interp.group['filter'].interp['alt']

for x in np.linspace(imu_min, imu_max, int(round(flight_len*hz))):
    flight_interp.append( [x, p_interp(x), q_interp(x), r_interp(x) ] )
print("flight len:", len(flight_interp))

# find the time correlation of video vs flight data
time_shift = \
    correlate.sync_gyros(flight_interp, video_interp, horiz_len,
                         hz=hz, cam_mount=args.cam_mount,
                         force_time_shift=args.time_shift, plot=args.plot)

# optimizer stuffs
from scipy.optimize import least_squares

# Scan altitude range so we can match the portion of the flight that
# is up and away.  This means the EKF will have had a better chance to
# converge, and the horizon detection should be getting a clear view.
min_alt = None
max_alt = None
for filt in flight_data['filter']:
    alt = filt['alt']
    if min_alt is None or alt < min_alt:
        min_alt = alt
    if max_alt is None or alt > max_alt:
        max_alt = alt
print("altitude range: %.1f - %.1f (m)" % (min_alt, max_alt))
if max_alt - min_alt > 40:
    alt_threshold = min_alt + 30 # approx 100'
else:
    alt_threshold = (max_alt - min_alt) * 0.75
print("Altitude threshold: %.1f (m)" % alt_threshold)

# presample datas to save work in the error function
tmin = np.amax( [xmin + time_shift, imu_min ] )
tmax = np.amin( [xmax + time_shift, imu_max ] )
tlen = tmax - tmin
print("overlap range (flight sec):", tmin, " - ", tmax)
data = []
for x in np.linspace(tmin, tmax, int(round(tlen*hz))):
    # video
    vp = video_p(x-time_shift)
    vq = video_q(x-time_shift)
    vr = video_r(x-time_shift)
    # flight data
    fp = p_interp(x)
    fq = q_interp(x)
    fr = r_interp(x)
    alt = alt_interp(x)
    if alt >= alt_threshold:
        data.append( [x, vp, vq, vr, fp, fq, fr] )

initial = [0.0,  0.0, 0.0]
print("starting est:", initial)

# data = np.array(data)
# plt.figure()
# plt.plot(data[:,0], data[:,1], label="video roll")
# plt.plot(data[:,0], data[:,3], label="ekf roll")
# plt.legend()
# plt.show()

def errorFunc(xk):
    print("    Trying:", xk)
    # order is yaw, pitch, roll
    R = euler_matrix(xk[0], xk[1], xk[2], 'rzyx')[:3,:3]
    #print("R:\n", R)
    # compute error function using global data structures
    result = []
    for r in data:
        cam_gyro = r[1:4]
        imu_gyro = r[4:7]
        #print("cam_gyro:", cam_gyro, "imu_gyro:", imu_gyro)
        proj_gyro = R @ cam_gyro
        #print("proj_gyro:", proj_gyro)
        diff = imu_gyro - proj_gyro
        #print("diff:", diff)
        result.append( np.linalg.norm(diff) )
    return np.array(result)

if False:
    print("Optimizing...")
    res = least_squares(errorFunc, initial, verbose=2)
    #res = least_squares(errorFunc, initial, diff_step=0.0001, verbose=2)
    print(res)
    print("Camera mount offset:")
    print("Yaw: %.2f" % (res['x'][0]*r2d))
    print("Pitch: %.2f" % (res['x'][1]*r2d))
    print("Roll: %.2f" % (res['x'][2]*r2d))
    initial = res['x']

def myopt(func, xk, spread):
    print("Hunting for best result...")
    done = False
    estimate = list(xk)
    while not done:
        for n in range(len(estimate)):
            xdata = []
            ydata = []
            center = estimate[n]
            for x in np.linspace(center-spread, center+spread, num=11):
                estimate[n] = x
                result = func(estimate)
                avg = np.mean(result)
                std = np.std(result)
                print("angle (deg) %.2f:" % (x*r2d),
                      "avg: %.6f" % np.mean(result),
                      "std: %.4f" % np.std(result))
                xdata.append(x)
                ydata.append(avg)
            fit = np.polynomial.polynomial.polyfit( np.array(xdata), np.array(ydata), 2 )
            print("poly fit:", fit)
            poly = np.polynomial.polynomial.Polynomial(fit)
            deriv = np.polynomial.polynomial.polyder(fit)
            roots = np.polynomial.polynomial.polyroots(deriv)
            print("roots:", roots)
            estimate[n] = roots[0]
            if args.plot:
                plt.figure()
                x = np.linspace(center-spread, center+spread, num=1000)
                plt.plot(x, poly(x), 'r-')
                plt.plot(xdata, ydata, 'b*')
                plt.show()
        spread = spread / 4
        if spread < 0.001:     # rad
            done = True
    print("Minimal error for index n at angle %.2f (deg)\n" % (estimate[n] * r2d))
    return estimate

print("Hunting for optimal yaw offset ...")
spread = 25*d2r
est = list(initial)
result = myopt(errorFunc, est, spread)

print("Best result:", np.array(result)*r2d)
