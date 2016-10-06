#!/usr/bin/python

# find our custom built opencv first
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

import argparse
import cv2
import datetime
import ephem
import math
import fractions
from matplotlib import pyplot as plt 
import navpy
import numpy as np
import os
import pyexiv2
import re
#from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate # strait up linear interpolation, nothing fancy

from props import PropertyNode, getNode
import props_json

sys.path.append('../lib')
import transformations

# a helpful constant
d2r = math.pi / 180.0

parser = argparse.ArgumentParser(description='correlate movie data to flight data.')
parser.add_argument('--movie', required=True, help='original movie')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--resample-hz', type=float, default=30.0, help='resample rate (hz)')
parser.add_argument('--apm-log', help='APM tlog converted to csv')
parser.add_argument('--aura-dir', help='Aura flight log directory')
parser.add_argument('--stop-count', type=int, default=1, help='how many non-frames to absorb before we decide the movie is over')
args = parser.parse_args()

r2d = 180.0 / math.pi
counter = 0
stop_count = 0

# pathname work
abspath = os.path.abspath(args.movie)
filename, ext = os.path.splitext(abspath)
movie_log = filename + ".csv"
movie_config = filename + ".json"

# load config file if it exists
config = PropertyNode()
props_json.load(movie_config, config)
cam_yaw = config.getFloat('cam_yaw_deg')
cam_pitch = config.getFloat('cam_pitch_deg')
cam_roll = config.getFloat('cam_roll_deg')

# load movie log
movie = []
with open(movie_log, 'rb') as f:
    for line in f:
        movie.append( re.split('[,\s]+', line.rstrip()) )

flight_imu = []
flight_gps = []
flight_filter = []
flight_air = []
last_imu_time = -1
last_gps_time = -1

if args.apm_log:
    # load APM flight log
    agl = 0.0
    with open(args.apm_log, 'rb') as f:
        for line in f:
            tokens = re.split('[,\s]+', line.rstrip())
            if tokens[7] == 'mavlink_attitude_t':
                timestamp = float(tokens[9])/1000.0
                print timestamp
                if timestamp > last_imu_time:
                    flight_imu.append( [timestamp,
                                        float(tokens[17]), float(tokens[19]),
                                        float(tokens[21]),
                                        float(tokens[11]), float(tokens[13]),
                                        float(tokens[15])] )
                    last_imu_time = timestamp
                else:
                    print "ERROR: IMU time went backwards:", timestamp, last_imu_time
            elif tokens[7] == 'mavlink_gps_raw_int_t':
                timestamp = float(tokens[9])/1000000.0
                if timestamp > last_gps_time - 1.0:
                    flight_gps.append( [timestamp,
                                        float(tokens[11]) / 10000000.0,
                                        float(tokens[13]) / 10000000.0,
                                        float(tokens[15]) / 1000.0,
                                        agl] )
                    last_gps_time = timestamp
                else:
                    print "ERROR: GPS time went backwards:", timestamp, last_gps_time
            elif tokens[7] == 'mavlink_terrain_report_t':
                agl = float(tokens[15])
elif args.aura_dir:
    # load Aura flight log
    imu_file = args.aura_dir + "/imu-0.txt"
    gps_file = args.aura_dir + "/gps-0.txt"
    filter_file = args.aura_dir + "/filter-0.txt"
    air_file = args.aura_dir + "/air-0.txt"
    
    last_time = 0.0
    with open(imu_file, 'rb') as f:
        for line in f:
            tokens = re.split('[,\s]+', line.rstrip())
            timestamp = float(tokens[0])
            if timestamp > last_time:
                flight_imu.append( [tokens[0], tokens[1], tokens[2],
                                    tokens[3], 0.0, 0.0, 0.0] )
            else:
                print "ERROR: time went backwards:", timestamp, last_time
            last_time = timestamp

    last_time = 0.0
    with open(gps_file, 'rb') as f:
        for line in f:
            #print line
            tokens = re.split('[,\s]+', line.rstrip())
            timestamp = float(tokens[0])
            #print timestamp, last_time
            if timestamp > last_time:
                flight_gps.append( [tokens[0], tokens[1], tokens[2],
                                    tokens[3], tokens[7]] )
            else:
                print "ERROR: time went backwards:", timestamp, last_time
            last_time = timestamp

    last_time = 0.0
    with open(filter_file, 'rb') as f:
        for line in f:
            #print line
            tokens = re.split('[,\s]+', line.rstrip())
            timestamp = float(tokens[0])
            #print timestamp, last_time
            if timestamp > last_time:
                flight_filter.append( [tokens[0],
                                       tokens[1], tokens[2], tokens[3],
                                       tokens[4], tokens[5], tokens[6],
                                       tokens[7], tokens[8], tokens[9]] )
            else:
                print "ERROR: time went backwards:", timestamp, last_time
            last_time = timestamp

    last_time = 0.0
    with open(air_file, 'rb') as f:
        for line in f:
            #print line
            tokens = re.split('[,\s]+', line.rstrip())
            timestamp = float(tokens[0])
            #print timestamp, last_time
            if timestamp > last_time:
                flight_air.append( [tokens[0], tokens[1], tokens[2],
                                    tokens[3]] )
            else:
                print "ERROR: time went backwards:", timestamp, last_time
            last_time = timestamp
else:
    print "No flight log specified, cannot continue."
    quit()
    
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
    movie_interp.append( [x, movie_spl_roll(x)] )
print "movie len:", len(movie_interp)

# resample flight data
flight_imu = np.array(flight_imu, dtype=float)
flight_interp = []
x = flight_imu[:,0]
flight_imu_p = interpolate.interp1d(x, flight_imu[:,1], bounds_error=False, fill_value=0.0)
flight_imu_q = interpolate.interp1d(x, flight_imu[:,2], bounds_error=False, fill_value=0.0)
flight_imu_r = interpolate.interp1d(x, flight_imu[:,3], bounds_error=False, fill_value=0.0)
if args.apm_log:
    y_spline = flight_imu_r
elif args.aura_dir:
    y_spline = flight_imu_p
xmin = x.min()
xmax = x.max()
print "flight range = %.3f - %.3f (%.3f)" % (xmin, xmax, xmax-xmin)
time = xmax - xmin
for x in np.linspace(xmin, xmax, time*args.resample_hz):
    flight_interp.append( [x, y_spline(x)] )
print "flight len:", len(flight_interp)

flight_gps = np.array(flight_gps, dtype=np.float64)
x = flight_gps[:,0]
flight_gps_lat = interpolate.interp1d(x, flight_gps[:,1], bounds_error=False, fill_value=0.0)
flight_gps_lon = interpolate.interp1d(x, flight_gps[:,2], bounds_error=False, fill_value=0.0)
flight_gps_alt = interpolate.interp1d(x, flight_gps[:,3], bounds_error=False, fill_value=0.0)
flight_gps_unixtime = interpolate.interp1d(x, flight_gps[:,4], bounds_error=False, fill_value=0.0)

flight_filter = np.array(flight_filter, dtype=float)
x = flight_filter[:,0]
flight_filter_vn = interpolate.interp1d(x, flight_filter[:,4], bounds_error=False, fill_value=0.0)
flight_filter_ve = interpolate.interp1d(x, flight_filter[:,5], bounds_error=False, fill_value=0.0)
flight_filter_vd = interpolate.interp1d(x, flight_filter[:,6], bounds_error=False, fill_value=0.0)
flight_filter_roll = interpolate.interp1d(x, flight_filter[:,7], bounds_error=False, fill_value=0.0)
flight_filter_pitch = interpolate.interp1d(x, flight_filter[:,8], bounds_error=False, fill_value=0.0)
flight_filter_yaw = interpolate.interp1d(x, flight_filter[:,9], bounds_error=False, fill_value=0.0)

flight_air = np.array(flight_air, dtype=np.float64)
x = flight_air[:,0]
flight_air_speed = interpolate.interp1d(x, flight_air[:,3], bounds_error=False, fill_value=0.0)

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
    fqsum += abs(flight_imu_q(x))
    frsum += abs(flight_imu_r(x))
if fqsum > 0.001:
    qratio = mqsum / fqsum
if mrsum > 0.001:
    rratio = -mrsum / frsum
print "pitch ratio:", qratio
print "yaw ratio:", rratio

def compute_sun_moon_ned(lon_deg, lat_deg, alt_m, timestamp):
    d = datetime.datetime.utcfromtimestamp(timestamp)
    #d = datetime.datetime.utcnow()
    ed = ephem.Date(d)
    #print 'ephem time utc:', ed
    #print 'localtime:', ephem.localtime(ed)

    ownship = ephem.Observer()
    ownship.lon = '%.8f' % lon_deg
    ownship.lat = '%.8f' % lat_deg
    ownship.elevation = alt_m
    ownship.date = ed

    sun = ephem.Sun(ownship)
    moon = ephem.Moon(ownship)

    sun_ned = [ math.cos(sun.az), math.sin(sun.az), -math.sin(sun.alt) ]
    moon_ned = [ math.cos(moon.az), math.sin(moon.az), -math.sin(moon.alt) ]

    return sun_ned, moon_ned

def project_point(K, PROJ, ned):
    uvh = K.dot( PROJ.dot( [ned[0], ned[1], ned[2], 1.0] ).T )
    if uvh[2] > 0.1:
        uvh /= uvh[2]
        uv = ( int(np.squeeze(uvh[0,0])), int(np.squeeze(uvh[1,0])) )
        return uv
    else:
        return None

def draw_horizon(K, PROJ, ned, frame):
    divs = 10
    pts = []
    for i in range(divs + 1):
        a = (float(i) * 360/float(divs)) * d2r
        n = math.cos(a)
        e = math.sin(a)
        d = 0.0
        pts.append( [n, e, d] )

    for i in range(divs):
        p1 = pts[i]
        p2 = pts[i+1]
        uv1 = project_point(K, PROJ,
                            [ned[0] + p1[0], ned[1] + p1[1], ned[2] + p1[2]])
        uv2 = project_point(K, PROJ,
                            [ned[0] + p2[0], ned[1] + p2[1], ned[2] + p2[2]])
        if uv1 != None and uv2 != None:
            cv2.line(frame, uv1, uv2, (0,240,0), 1, cv2.CV_AA)

def draw_label(frame, label, uv, font, font_scale, thickness, center='horiz'):
        size = cv2.getTextSize(label, font, font_scale, thickness)
        if center == 'horiz':
            uv = (uv[0] - (size[0][0] / 2), uv[1])
        cv2.putText(frame, label, uv, font, font_scale, (0,255,0), thickness, cv2.CV_AA)

def draw_labeled_point(K, PROJ, frame, ned, label):
    uv = project_point(K, PROJ, [ned[0], ned[1], ned[2]])
    if uv != None:
        cv2.circle(frame, uv, 5, (0,240,0), 1, cv2.CV_AA)
    uv = project_point(K, PROJ, [ned[0], ned[1], ned[2] - 0.02])
    if uv != None:
        draw_label(frame, label, uv, font, 0.8, 1)

                       
def draw_compass_points(K, PROJ, ned, frame):
    # 30 Ticks
    divs = 12
    pts = []
    for i in range(divs):
        a = (float(i) * 360/float(divs)) * d2r
        n = math.cos(a)
        e = math.sin(a)
        uv1 = project_point(K, PROJ,
                            [ned[0] + n, ned[1] + e, ned[2] - 0.0])
        uv2 = project_point(K, PROJ,
                            [ned[0] + n, ned[1] + e, ned[2] - 0.02])
        if uv1 != None and uv2 != None:
            cv2.line(frame, uv1, uv2, (0,240,0), 1, cv2.CV_AA)

    # North
    uv = project_point(K, PROJ,
                       [ned[0] + 1.0, ned[1] + 0.0, ned[2] - 0.03])
    if uv != None:
        draw_label(frame, 'N', uv, font, 1, 2)
    # South
    uv = project_point(K, PROJ,
                       [ned[0] - 1.0, ned[1] + 0.0, ned[2] - 0.03])
    if uv != None:
        draw_label(frame, 'S', uv, font, 1, 2)
    # East
    uv = project_point(K, PROJ,
                       [ned[0] + 0.0, ned[1] + 1.0, ned[2] - 0.03])
    if uv != None:
        draw_label(frame, 'E', uv, font, 1, 2)
    # West
    uv = project_point(K, PROJ,
                       [ned[0] + 0.0, ned[1] - 1.0, ned[2] - 0.03])
    if uv != None:
        draw_label(frame, 'W', uv, font, 1, 2)

def draw_astro(K, PROJ, ned, frame):
    lat_deg = float(flight_gps_lat(time))
    lon_deg = float(flight_gps_lon(time))
    alt_m = float(flight_gps_alt(time))
    timestamp = float(flight_gps_unixtime(time))
    sun_ned, moon_ned = compute_sun_moon_ned(lon_deg, lat_deg, alt_m,
                                             timestamp)
    if sun_ned == None or moon_ned == None:
        return

    # Sun
    draw_labeled_point(K, PROJ, frame,
                       [ned[0] + sun_ned[0], ned[1] + sun_ned[1],
                        ned[2] + sun_ned[2]],
                       'Sun')
    # shadow (if sun above horizon)
    if sun_ned[2] < 0.0:
        draw_labeled_point(K, PROJ, frame,
                           [ned[0] - sun_ned[0], ned[1] - sun_ned[1],
                            ned[2] - sun_ned[2]],
                           'shadow')
    # Moon
    draw_labeled_point(K, PROJ, frame,
                       [ned[0] + moon_ned[0], ned[1] + moon_ned[1],
                        ned[2] + moon_ned[2]],
                       'Moon')

def draw_velocity_vector(K, PROJ, ned, frame, vel):
    uv = project_point(K, PROJ,
                       [ned[0] + vel[0], ned[1] + vel[1], ned[2]+ vel[2]])
    if uv != None:
        cv2.circle(frame, uv, 5, (0,240,0), 1, cv2.CV_AA)

if args.movie:
    K_RunCamHD2_1920x1080 \
        = np.array( [[ 971.96149426,   0.        , 957.46750602],
                     [   0.        , 971.67133264, 516.50578382],
                     [   0.        ,   0.        ,   1.        ]] )
    dist_runcamhd2_1920x1080 = [-0.26910665, 0.10580125, 0.00048417, 0.00000925, -0.02321387]
    K = K_RunCamHD2_1920x1080 * args.scale
    K[2,2] = 1.0
    print K
    dist = dist_runcamhd2_1920x1080

    font = cv2.FONT_HERSHEY_SIMPLEX

    # these are fixed tranforms between ned and camera reference systems
    proj2ned = np.array( [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                         dtype=float )
    ned2proj = np.linalg.inv(proj2ned)

    #cam_ypr = [-3.0, -12.0, -3.0] # yaw, pitch, roll
    #ref = [44.7260320000, -93.0771072000, 0]
    ref = [ flight_gps[0][1], flight_gps[0][2], 0.0 ]
    print 'ned ref:', ref

    # overlay hud
    print "Opening ", args.movie
    try:
        capture = cv2.VideoCapture(args.movie)
    except:
        print "error opening video"

    capture.read()
    counter += 1
    print "ok reading first frame"

    fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
    print "fps = %.2f" % fps
    fourcc = int(capture.get(cv2.cv.CV_CAP_PROP_FOURCC))
    w = capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    h = capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    last_time = 0.0

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
        print "frame: ", counter, time
        counter += 1
        vn = flight_filter_vn(time)
        ve = flight_filter_ve(time)
        vd = flight_filter_vd(time)
        yaw_rad = flight_filter_yaw(time)*d2r
        pitch_rad = flight_filter_pitch(time)*d2r
        roll_rad = flight_filter_roll(time)*d2r
        lat_deg = float(flight_gps_lat(time))
        lon_deg = float(flight_gps_lon(time))
        altitude = float(flight_gps_alt(time))
        speed = float(flight_air_speed(time))

        body2cam = transformations.quaternion_from_euler( cam_yaw * d2r,
                                                          cam_pitch * d2r,
                                                          cam_roll * d2r,
                                                          'rzyx')

        #print 'att:', [yaw_rad, pitch_rad, roll_rad]
        ned2body = transformations.quaternion_from_euler(yaw_rad,
                                                         pitch_rad,
                                                         roll_rad,
                                                         'rzyx')
        #print 'ned2body(q):', ned2body
        ned2cam_q = transformations.quaternion_multiply(ned2body, body2cam)
        ned2cam = np.matrix(transformations.quaternion_matrix(np.array(ned2cam_q))[:3,:3]).T
        #print 'ned2cam:', ned2cam
        R = ned2proj.dot( ned2cam )
        rvec, jac = cv2.Rodrigues(R)
        ned = navpy.lla2ned( lat_deg, lon_deg, altitude,
                             ref[0], ref[1], ref[2] )
        #print 'ned:', ned
        tvec = -np.matrix(R) * np.matrix(ned).T
        R, jac = cv2.Rodrigues(rvec)
        # is this R the same as the earlier R?
        PROJ = np.concatenate((R, tvec), axis=1)
        #print 'PROJ:', PROJ
        #print lat_deg, lon_deg, altitude, ref[0], ref[1], ref[2]
        #print ned

        method = cv2.INTER_AREA
        #method = cv2.INTER_LANCZOS4
        frame_scale = cv2.resize(frame, (0,0), fx=args.scale, fy=args.scale,
                                 interpolation=method)
        frame_undist = cv2.undistort(frame_scale, K, np.array(dist))

        cv2.putText(frame_undist, 'alt = %.0f' % altitude, (100, 100), font, 1.5, (0,255,0), 2,cv2.CV_AA)
        cv2.putText(frame_undist, 'kts = %.0f' % speed, (100, 150), font, 1.5, (0,255,0), 2,cv2.CV_AA)

        draw_horizon(K, PROJ, ned, frame_undist)
        draw_compass_points(K, PROJ, ned, frame_undist)
        draw_astro(K, PROJ, ned, frame_undist)
        draw_velocity_vector(K, PROJ, ned, frame_undist, [vn, ve, vd])

        cv2.imshow('hud', frame_undist)
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == ord('y'):
            cam_yaw += 0.5
            config.setFloat('cam_yaw_deg', cam_yaw)
            props_json.save(movie_config, config)
        elif key == ord('Y'):
            cam_yaw -= 0.5
            config.setFloat('cam_yaw_deg', cam_yaw)
            props_json.save(movie_config, config)
        elif key == ord('p'):
            cam_pitch += 0.5
            config.setFloat('cam_pitch_deg', cam_pitch)
            props_json.save(movie_config, config)
        elif key == ord('P'):
            cam_pitch -= 0.5
            config.setFloat('cam_pitch_deg', cam_pitch)
            props_json.save(movie_config, config)
        elif key == ord('r'):
            cam_roll -= 0.5
            config.setFloat('cam_roll_deg', cam_roll)
            props_json.save(movie_config, config)
        elif key == ord('R'):
            cam_roll += 0.5
            config.setFloat('cam_roll_deg', cam_roll)
            props_json.save(movie_config, config)


# plot the data ...
plt.figure(1)
plt.ylabel('roll rate (deg per sec)')
plt.xlabel('flight time (sec)')
plt.plot(movie[:,0] + time_shift, movie[:,2]*r2d, label='estimate from flight movie')
if args.apm_log:
    plt.plot(flight_imu[:,0], flight_imu[:,3]*r2d, label='flight data log')
else:
    plt.plot(flight_imu[:,0], flight_imu[:,1]*r2d, label='flight data log')
#plt.plot(movie_interp[:,1])
#plt.plot(flight_interp[:,1])
plt.legend()

plt.figure(2)
plt.plot(ycorr)

plt.figure(3)
plt.ylabel('pitch rate (deg per sec)')
plt.xlabel('flight time (sec)')
plt.plot(movie[:,0] + time_shift, (movie[:,3]/qratio)*r2d, label='estimate from flight movie')
plt.plot(flight_imu[:,0], flight_imu[:,2]*r2d, label='flight data log')
#plt.plot(movie_interp[:,1])
#plt.plot(flight_interp[:,1])
plt.legend()

plt.figure(4)
plt.ylabel('yaw rate (deg per sec)')
plt.xlabel('flight time (sec)')
plt.plot(movie[:,0] + time_shift, (movie[:,4]/rratio)*r2d, label='estimate from flight movie')
plt.plot(flight_imu[:,0], flight_imu[:,3]*r2d, label='flight data log')
#plt.plot(movie_interp[:,1])
#plt.plot(flight_interp[:,1])
plt.legend()

plt.show()
