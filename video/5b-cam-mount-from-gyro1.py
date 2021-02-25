#!/usr/bin/env python3

# use feature-based motion (affine) roll, pitch, yaw rates.  Compare
# rates in camera space vs. imu space and try to find an optimal
# transform to minimize the idfference between them.

import argparse
import math
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
from feat_data import FeatureData

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
video_rates = filename + "_rates.csv"
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
r2d = 180.0 / math.pi
d2r = math.pi / 180.0

# load camera rotation rate data (derived from feature matching video
# frames)
feat_data = FeatureData()
feat_data.load(video_rates)
feat_data.smooth(smooth_cutoff_hz)
feat_data.make_interp()
if args.plot:
    feat_data.plot()
feat_interp = feat_data.resample(args.resample_hz)

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
plt.plot(feat_data.data['p (rad/sec)'], label='video (smooth)')
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
#phi_interp = interpolate.interp1d(ekf['time'], ekf['phi'], bounds_error=False, fill_value=0.0)
#the_interp = interpolate.interp1d(ekf['time'], ekf['the'], bounds_error=False, fill_value=0.0)
#psix_interp = interpolate.interp1d(ekf['time'], ekf['psix'], bounds_error=False, fill_value=0.0)
#psiy_interp = interpolate.interp1d(ekf['time'], ekf['psiy'], bounds_error=False, fill_value=0.0)

for x in np.linspace(imu_min, imu_max, int(round(flight_len*hz))):
    flight_interp.append( [x, p_interp(x), q_interp(x), r_interp(x) ] )
print("flight len:", len(flight_interp))

# find the time correlation of video vs flight data
time_shift = \
    correlate.sync_gyros(flight_interp, feat_interp, feat_data.span_sec,
                         hz=hz, cam_mount=args.cam_mount,
                         force_time_shift=args.time_shift, plot=args.plot)

# optimizer stuffs
from scipy.optimize import least_squares

# presample datas to save work in the error function
tmin = np.amax( [feat_data.tmin + time_shift, imu_min ] )
tmax = np.amin( [feat_data.tmax + time_shift, imu_max ] )
tlen = tmax - tmin
print("overlap range (flight sec):", tmin, " - ", tmax)

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
if max_alt - min_alt > 30:
    alt_threshold = min_alt + (max_alt - min_alt) * 0.5
else:
    alt_threshold = min_alt
print("Altitude threshold: %.1f (m)" % alt_threshold)

data = []
for x in np.linspace(tmin, tmax, int(round(tlen*hz))):
    # feature-based
    vp, vq, vr = feat_data.get_vals(x - time_shift)
    # flight data
    fp = p_interp(x)
    fq = q_interp(x)
    fr = r_interp(x)
    alt = alt_interp(x)
    if alt >= alt_threshold:
        data.append( [x, vp, vq, vr, fp, fq, fr] )
print("Data points passing altitude threshold:", len(data))

# y, p, r, imu_gyro_biases (p, q, r)
initial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
    R = transformations.euler_matrix(xk[0], xk[1], xk[2], 'rzyx')[:3,:3]
    #print("R:\n", R)
    # compute error function using global data structures
    result = []
    for r in data:
        cam_gyro = r[1:4]
        imu_gyro = r[4:7] + np.array(xk[3:6])
        #print(r[4:7], imu_gyro)
        #cam_gyro[1] = 0
        #imu_gyro[1] = 0
        #cam_gyro[2] = 0
        #imu_gyro[2] = 0
        #print("cam_gyro:", cam_gyro, "imu_gyro:", imu_gyro)
        proj_gyro = R @ cam_gyro
        #print("proj_gyro:", proj_gyro)
        diff = imu_gyro - proj_gyro
        #print("diff:", diff)
        result.append( np.linalg.norm(diff) )
    return np.array(result)

if True:
    print("Optimizing...")
    res = least_squares(errorFunc, initial, verbose=2)
    #res = least_squares(errorFunc, initial, diff_step=0.0001, verbose=2)
    print(res)
    print(res['x'] * r2d)
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

if False:
    print("Optimizing...")
    spread = 30*d2r
    est = list(initial)
    result = myopt(errorFunc, est, spread)

    print("Best result:", np.array(result)*r2d)

# blowing away data for new purposes, (should clean this up)
data = []
for x in np.linspace(tmin, tmax, int(round(tlen*hz))):
    # horizon
    hphi = horiz_phi(x-time_shift)
    hthe = horiz_the(x-time_shift)
    # flight data
    fphi = phi_interp(x)
    fthe = the_interp(x)
    psix = psix_interp(x)
    psiy = psiy_interp(x)
    fpsi = math.atan2(psiy, psix)
    alt = alt_interp(x)
    if alt >= alt_threshold:
        data.append( [x, hphi, hthe, fpsi, fthe, fphi] )
    
def horiz_errorFunc(xk):
    print("    Trying:", xk)
    camera.set_ypr(xk[0]*r2d, xk[1]*r2d, xk[2]*r2d) # cam mount offset
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

print("Plotting final result...")
result = horiz_errorFunc(np.array(result))
rollerr = result[::2]
pitcherr = result[1::2]
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
