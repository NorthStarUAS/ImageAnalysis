#!/usr/bin/env python3

# use the horizon angle data (in camera space) along with estimated
# roll/pitch arates to correlate video time to flight data log time.
# Then use an optimizer to estimate the best fitting roll, pitch, yaw
# offsets for the camera mount relative to the IMU/EKF solution.

import argparse
import math
from matplotlib import pyplot as plt 
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

# load horizon log data (derived from video)
#
# frame,video time,camera roll (deg),camera pitch (deg),
# roll rate (rad/sec),pitch rate (rad/sec)
horiz_data = pd.read_csv(horiz_log)
horiz_data.set_index('video time', inplace=True, drop=False)
xmin = horiz_data['video time'].min()
xmax = horiz_data['video time'].max()
horiz_count = len(horiz_data['video time'])
print("number of horizon records:", horiz_count)
horiz_fs = int(round((horiz_count / (xmax - xmin))))
print("horiz fs:", horiz_fs)

# smooth the horizon data
import scipy.signal as signal
b, a = signal.butter(2, 1.0, fs=horiz_fs)
horiz_data['camera roll (deg)'] = signal.filtfilt(b, a, horiz_data['camera roll (deg)'])
horiz_data['camera pitch (deg)'] = signal.filtfilt(b, a, horiz_data['camera pitch (deg)'])

# resample horizon data
horiz_interp = []
horiz_p = interpolate.interp1d(horiz_data['video time'],
                               horiz_data['roll rate (rad/sec)'],
                               bounds_error=False, fill_value=0.0)
horiz_q = interpolate.interp1d(horiz_data['video time'],
                               horiz_data['pitch rate (rad/sec)'],
                               bounds_error=False, fill_value=0.0)
horiz_phi = interpolate.interp1d(horiz_data['video time'],
                                 horiz_data['camera roll (deg)'] * d2r,
                                 bounds_error=False, fill_value=0.0)
horiz_the = interpolate.interp1d(horiz_data['video time'],
                                 horiz_data['camera pitch (deg)'] * d2r,
                                 bounds_error=False, fill_value=0.0)
xmin = horiz_data['video time'].min()
xmax = horiz_data['video time'].max()
print("video range = %.3f - %.3f (%.3f)" % (xmin, xmax, xmax-xmin))
horiz_len = xmax - xmin
for x in np.linspace(xmin, xmax, int(round(horiz_len*hz))):
    if args.cam_mount == 'forward' or args.cam_mount == 'down':
        horiz_interp.append( [x, horiz_p(x), horiz_q(x),
                              horiz_phi(x), horiz_the(x)] )
print("horizon data len:", len(horiz_interp))

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
plt.plot(horiz_data['camera roll (deg)']*d2r, label='video')
plt.legend()
#plt.show()

# resample flight data
flight_min = flight_data['imu'][0]['time']
flight_max = flight_data['imu'][-1]['time']
print("flight range = %.3f - %.3f (%.3f)" % (flight_min, flight_max,
                                             flight_max-flight_min))
flight_interp = []
if args.cam_mount == 'forward' or args.cam_mount == 'rear':
    # forward/rear facing camera
    p_interp = interp.group['imu'].interp['p']
    q_interp = interp.group['imu'].interp['q']
elif args.cam_mount == 'left' or args.cam_mount == 'right':
    # it might be interesting to support an out-the-wing view
    print("Not currently supported camera orientation, sorry!")
    quit()
flight_len = flight_max - flight_min
phi_interp = interpolate.interp1d(ekf['time'], ekf['phi'], bounds_error=False, fill_value=0.0)
the_interp = interpolate.interp1d(ekf['time'], ekf['the'], bounds_error=False, fill_value=0.0)
psix_interp = interpolate.interp1d(ekf['time'], ekf['psix'], bounds_error=False, fill_value=0.0)
psiy_interp = interpolate.interp1d(ekf['time'], ekf['psiy'], bounds_error=False, fill_value=0.0)
alt_interp = interp.group['filter'].interp['alt']

for x in np.linspace(flight_min, flight_max, int(round(flight_len*hz))):
    flight_interp.append( [x, p_interp(x), q_interp(x),
                           phi_interp(x), the_interp(x),
                           psix_interp(x), psiy_interp(x)] )
print("flight len:", len(flight_interp))

# find the time correlation of video vs flight data
time_shift = \
    correlate.sync_horizon(flight_data, flight_interp,
                           horiz_data, horiz_interp, horiz_len,
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
if max_alt - min_alt > 30:
    alt_threshold = min_alt + (max_alt - min_alt) * 0.5
else:
    alt_threshold = min_alt
print("Altitude threshold: %.1f (m)" % alt_threshold)

# presample datas to save work in the error function
tmin = np.amax( [xmin + time_shift, flight_min ] )
tmax = np.amin( [xmax + time_shift, flight_max ] )
tlen = tmax - tmin
print("overlap range (flight sec):", tmin, " - ", tmax)
data = []
roll_sum = 0
pitch_sum = 0
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
    roll_sum += hphi - fphi
    pitch_sum += hthe - fthe
    
initial = [0.0,  pitch_sum / (tlen*hz), roll_sum / (tlen*hz) ] # rads
print("starting est:", initial)

# data = np.array(data)
# plt.figure()
# plt.plot(data[:,0], data[:,1], label="video roll")
# plt.plot(data[:,0], data[:,3], label="ekf roll")
# plt.legend()
# plt.show()

# precompute to save time
horiz_ned = [0, 0, 0]  # any value works here (as long as it's consistent
horiz_divs = 10
horiz_pts = []
for i in range(horiz_divs + 1):
    a = (float(i) * 360/float(horiz_divs)) * d2r
    n = math.cos(a) + horiz_ned[0]
    e = math.sin(a) + horiz_ned[1]
    d = 0.0 + horiz_ned[2]
    horiz_pts.append( [n, e, d] )
    
# a, b are line end points, p is some other point
# returns the closest point on ab to p (orthogonal projection)
def ClosestPointOnLine(a, b, p):
    ap = p - a
    ab = b - a
    return a + np.dot(ap,ab) / np.dot(ab,ab) * ab

# get the roll/pitch of camera orientation relative to specified
# horizon line
def get_projected_attitude(uv1, uv2, IK, cu, cv):
    # print('line:', line)
    du = uv2[0] - uv1[0]
    dv = uv1[1] - uv2[1]        # account for (0,0) at top left corner in image space
    roll = math.atan2(dv, du)

    if False:
        # temp test
        w = cu * 2
        h = cv * 2
        for p in [ (0, 0, 1), (w, 0, 1), (0, h, 1), (w, h, 1), (cu, cv, 1) ]:
            uvh = np.array(p)
            proj = IK.dot(uvh)
            print(p, "->", proj)
        
    p0 = ClosestPointOnLine(np.array(uv1), np.array(uv2), np.array([cu,cv]))
    uvh = np.array([p0[0], p0[1], 1.0])
    proj = IK.dot(uvh)
    #print("proj:", proj, proj/np.linalg.norm(proj))
    dot_product = np.dot(np.array([0,0,1]), proj/np.linalg.norm(proj))
    pitch = np.arccos(dot_product)
    if p0[1] < cv:
        pitch = -pitch
    #print("roll: %.1f pitch: %.1f" % (roll, pitch))
    return roll, pitch

cam_w, cam_h = camera.get_shape()
def find_horizon():
    answers = []
    for i in range(horiz_divs):
        p1 = horiz_pts[i]
        p2 = horiz_pts[i+1]
        uv1 = camera.project_ned( horiz_pts[i] )
        uv2 = camera.project_ned( horiz_pts[i+1] )
        if uv1 != None and uv2 != None:
            #print(" ", uv1, uv2)
            roll, pitch = get_projected_attitude(uv1, uv2, IK, cu, cv)
            answers.append( (roll, pitch) )
    if len(answers) > 0:
        index = int(len(answers) / 2)
        return answers[index]
    else:
        return None, None
            
def errorFunc(xk):
    print("    Trying:", xk)
    camera.set_ypr(xk[0]*r2d, xk[1]*r2d, xk[2]*r2d)
    # compute error function using global data structures
    result = []
    for r in data:
        camera.update_PROJ(horiz_ned, r[3], r[4], r[5])
        #print("video:", r[1]*r2d, r[2]*r2d)
        roll, pitch = find_horizon()
        #result.append( r[1] - (r[5] + xk[2]) )
        #result.append( r[2] - (r[4] + xk[1]) )
        if not roll is None:
            result.append( r[1] - roll )
            result.append( r[2] - pitch )
    return np.array(result)

if True:
    print("Hunting for optimal yaw offset ...")
    done = False
    spread = 10*d2r
    best_avg = None
    best_x = 0.0
    while not done:
        for x in np.linspace(best_x-spread, best_x+spread, num=11):
            initial[0] = x
            result = np.abs(errorFunc(initial))
            avg = np.mean(result)
            std = np.std(result)
            print("yaw %.2f:" % (x*r2d),
                  "avg: %.6f" % np.mean(result),
                  "std: %.4f" % np.std(result))
            if best_avg is None or avg < best_avg:
                best_avg = avg
                best_x = x
        spread = spread / 5
        if spread < 0.001:     # rad
            done = True
print("Best yaw: %.2f\n" % (best_x * r2d))

print("Optimizing...")
res = least_squares(errorFunc, initial, verbose=2)
#res = least_squares(errorFunc, initial, diff_step=0.0001, verbose=2)
print(res)
print("Camera mount offset:")
print("Yaw: %.2f" % (res['x'][0]*r2d))
print("Pitch: %.2f" % (res['x'][1]*r2d))
print("Roll: %.2f" % (res['x'][2]*r2d))

