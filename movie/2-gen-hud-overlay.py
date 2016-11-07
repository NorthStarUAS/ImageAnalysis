#!/usr/bin/python

# find our custom built opencv first
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")
import cv2

import argparse
import math
#import fractions
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

import hud

# helpful constants
d2r = math.pi / 180.0
mps2kt = 1.94384
kt2mps = 1 / mps2kt
ft2m = 0.3048
m2ft = 1 / ft2m

# default sizes of primatives
render_w = 1920
render_h = 1080

parser = argparse.ArgumentParser(description='correlate movie data to flight data.')
parser.add_argument('--movie', required=True, help='original movie')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--alpha', type=float, default=0.7, help='hud alpha blend')
parser.add_argument('--resample-hz', type=float, default=30.0, help='resample rate (hz)')
parser.add_argument('--apm-log', help='APM tlog converted to csv')
parser.add_argument('--aura-dir', help='Aura flight log directory')
parser.add_argument('--stop-count', type=int, default=1, help='how many non-frames to absorb before we decide the movie is over')
parser.add_argument('--plot', help='Plot stuff at the end of the run')
parser.add_argument('--auto-switch', choices=['old', 'new', 'none'], default='new', help='auto/manual switch logic helper')
parser.add_argument('--airspeed-units', choices=['kt', 'mps'], default='kt', help='display units for airspeed')
parser.add_argument('--altitude-units', choices=['ft', 'm'], default='ft', help='display units for airspeed')
args = parser.parse_args()

r2d = 180.0 / math.pi
counter = 0
stop_count = 0

# pathname work
abspath = os.path.abspath(args.movie)
filename, ext = os.path.splitext(abspath)
movie_log = filename + ".csv"
movie_config = filename + ".json"
# combinations that seem to work on linux
# ext = avi, fourcc = MJPG
# ext = avi, fourcc = XVID
# ext = mov, fourcc = MP4V

tmp_movie = filename + "_tmp.mov"
output_movie = filename + "_hud.mov"

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
flight_pilot = []
flight_ap = []
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
    if os.path.exists(args.aura_dir + "/filter-post.txt"):
        print "Notice: using filter-post.txt file because it exists!"
        filter_file = args.aura_dir + "/filter-post.txt"
    else:
        filter_file = args.aura_dir + "/filter-0.txt"
    air_file = args.aura_dir + "/air-0.txt"
    pilot_file = args.aura_dir + "/pilot-0.txt"
    ap_file = args.aura_dir + "/ap-0.txt"
    
    last_time = 0.0
    with open(imu_file, 'rb') as f:
        for line in f:
            tokens = re.split('[,\s]+', line.rstrip())
            timestamp = float(tokens[0])
            if timestamp > last_time:
                flight_imu.append( [tokens[0], tokens[1], tokens[2],
                                    tokens[3], 0.0, 0.0, 0.0] )
            else:
                # print "ERROR: time went backwards:", timestamp, last_time
                pass
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
                # print "ERROR: time went backwards:", timestamp, last_time
                pass
            last_time = timestamp

    last_time = 0.0
    with open(filter_file, 'rb') as f:
        for line in f:
            #print line
            tokens = re.split('[,\s]+', line.rstrip())
            timestamp = float(tokens[0])
            #print timestamp, last_time
            if timestamp > last_time:
                yaw = float(tokens[9])
                yaw_x = math.cos(yaw*d2r)
                yaw_y = math.sin(yaw*d2r)
                flight_filter.append( [tokens[0],
                                       tokens[1], tokens[2], tokens[3],
                                       tokens[4], tokens[5], tokens[6],
                                       tokens[7], tokens[8], tokens[9],
                                       yaw_x, yaw_y] )
            else:
                # print "ERROR: time went backwards:", timestamp, last_time
                pass
            last_time = timestamp

    last_time = 0.0
    with open(air_file, 'rb') as f:
        for line in f:
            #print line
            tokens = re.split('[,\s]+', line.rstrip())
            timestamp = float(tokens[0])
            #print timestamp, last_time
            if timestamp > last_time:
                flight_air.append( tokens )
            else:
                print "ERROR: time went backwards:", timestamp, last_time
            last_time = timestamp
            
    last_time = 0.0
    with open(pilot_file, 'rb') as f:
        for line in f:
            #print line
            tokens = re.split('[,\s]+', line.rstrip())
            timestamp = float(tokens[0])
            #print timestamp, last_time
            if timestamp > last_time:
                flight_pilot.append( tokens )
            else:
                # print "ERROR: time went backwards:", timestamp, last_time
                pass
            last_time = timestamp
            
    last_time = 0.0
    with open(ap_file, 'rb') as f:
        for line in f:
            #print line
            tokens = re.split('[,\s]+', line.rstrip())
            timestamp = float(tokens[0])
            #print timestamp, last_time
            if timestamp > last_time:
                hdg = float(tokens[1])
                hdg_x = math.cos(hdg*d2r)
                hdg_y = math.sin(hdg*d2r)
                tokens.append(hdg_x)
                tokens.append(hdg_y)
                flight_ap.append( tokens )
            else:
                # print "ERROR: time went backwards:", timestamp, last_time
                pass
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
flight_filter_yaw_x = interpolate.interp1d(x, flight_filter[:,10], bounds_error=False, fill_value=0.0)
flight_filter_yaw_y = interpolate.interp1d(x, flight_filter[:,11], bounds_error=False, fill_value=0.0)

flight_air = np.array(flight_air, dtype=np.float64)
x = flight_air[:,0]
flight_air_speed = interpolate.interp1d(x, flight_air[:,3], bounds_error=False, fill_value=0.0)
flight_air_true_alt = interpolate.interp1d(x, flight_air[:,5], bounds_error=False, fill_value=0.0)
flight_air_alpha = interpolate.interp1d(x, flight_air[:,11], bounds_error=False, fill_value=0.0)
flight_air_beta = interpolate.interp1d(x, flight_air[:,12], bounds_error=False, fill_value=0.0)

flight_pilot = np.array(flight_pilot, dtype=np.float64)
x = flight_pilot[:,0]
flight_pilot_auto = interpolate.interp1d(x, flight_pilot[:,8], bounds_error=False, fill_value=0.0)

flight_ap = np.array(flight_ap, dtype=np.float64)
x = flight_ap[:,0]
flight_ap_hdg = interpolate.interp1d(x, flight_ap[:,1], bounds_error=False, fill_value=0.0)
flight_ap_hdg_x = interpolate.interp1d(x, flight_ap[:,8], bounds_error=False, fill_value=0.0)
flight_ap_hdg_y = interpolate.interp1d(x, flight_ap[:,9], bounds_error=False, fill_value=0.0)
flight_ap_roll = interpolate.interp1d(x, flight_ap[:,2], bounds_error=False, fill_value=0.0)
flight_ap_alt = interpolate.interp1d(x, flight_ap[:,3], bounds_error=False, fill_value=0.0)
flight_ap_pitch = interpolate.interp1d(x, flight_ap[:,5], bounds_error=False, fill_value=0.0)
flight_ap_speed = interpolate.interp1d(x, flight_ap[:,7], bounds_error=False, fill_value=0.0)

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

if args.movie:
    # Mobius 1080p
    # K = np.array( [[1362.1,    0.0, 980.8],
    #                [   0.0, 1272.8, 601.3],
    #                [   0.0,    0.0,   1.0]] )
    # dist = [-0.36207197, 0.14627927, -0.00674558, 0.0008926, -0.02635695]

    # Mobius UMN-003 1920x1080
    K = np.array( [[ 1401.21111735,     0.       ,    904.25404757],
                   [    0.        ,  1400.2530882,    490.12157373],
                   [    0.        ,     0.       ,      1.        ]] )
    dist = [-0.39012303,  0.19687255, -0.00069657,  0.00465592, -0.05845262]

    # RunCamHD2 1920x1080
    # K = np.array( [[ 971.96149426,   0.        , 957.46750602],
    #                [   0.        , 971.67133264, 516.50578382],
    #                [   0.        ,   0.        ,   1.        ]] )
    # dist = [-0.26910665, 0.10580125, 0.00048417, 0.00000925, -0.02321387]

    # Runcamhd2 1920x1440
    # K = np.array( [[ 1296.11187055,     0.        ,   955.43024994],
    #                [    0.        ,  1296.01457451,   691.47053988],
    #                [    0.        ,     0.        ,     1.        ]] )
    # dist = [-0.28250371, 0.14064665, 0.00061846, 0.00014488, -0.05106045]
    
    K = K * args.scale
    K[2,2] = 1.0

    # overlay hud
    hud = hud.HUD(K)
    
    # these are fixed tranforms between ned and camera reference systems
    proj2ned = np.array( [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                         dtype=float )
    ned2proj = np.linalg.inv(proj2ned)

    #cam_ypr = [-3.0, -12.0, -3.0] # yaw, pitch, roll
    #ref = [44.7260320000, -93.0771072000, 0]
    ref = [ flight_gps[0][1], flight_gps[0][2], 0.0 ]
    hud.set_ned_ref(flight_gps[0][1], flight_gps[0][2])
    print 'ned ref:', ref

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
    w = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH) * args.scale )
    h = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) * args.scale )
    hud.set_render_size(w, h)
    
    #outfourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    #outfourcc = cv2.cv.CV_FOURCC('H', '2', '6', '4')
    #outfourcc = cv2.cv.CV_FOURCC('X', '2', '6', '4')
    #outfourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')
    outfourcc = cv2.cv.CV_FOURCC('M', 'P', '4', 'V')
    print outfourcc, fps, w, h
    output = cv2.VideoWriter(tmp_movie, outfourcc, fps, (w, h), isColor=True)

    last_time = 0.0

    # set primative sizes based on rendered resolution.
    hud.set_line_width( int(round(float(h) / 400.0)) )
    hud.set_font_size( float(h) / 900.0 )
    
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
        #yaw_rad = flight_filter_yaw(time)*d2r 
        yaw_x = flight_filter_yaw_x(time)
        yaw_y = flight_filter_yaw_y(time)
        yaw_rad = math.atan2(yaw_y, yaw_x)
        pitch_rad = flight_filter_pitch(time)*d2r
        roll_rad = flight_filter_roll(time)*d2r
        lat_deg = float(flight_gps_lat(time))
        lon_deg = float(flight_gps_lon(time))
        altitude_m = float(flight_air_true_alt(time))
        airspeed_kt = float(flight_air_speed(time))
        alpha_rad = float(flight_air_alpha(time))*d2r * 1.25
        beta_rad = float(flight_air_beta(time))*d2r * 1.25 - 0.05
        #ap_hdg = float(flight_ap_hdg(time))
        ap_hdg_x = float(flight_ap_hdg_x(time))
        ap_hdg_y = float(flight_ap_hdg_y(time))
        ap_hdg = math.atan2(ap_hdg_y, ap_hdg_x)*r2d
        ap_roll = float(flight_ap_roll(time))
        ap_pitch = float(flight_ap_pitch(time))
        ap_speed = float(flight_ap_speed(time))
        ap_alt = float(flight_ap_alt(time))
        auto_switch = float(flight_pilot_auto(time))
        if args.auto_switch == 'none':
            flight_mode = 'manual'
        elif (not args.auto_switch == 'new' and auto_switch < 0) or (args.auto_switch == 'old' and auto_switch > 0):
            flight_mode = 'manual'
        else:
            flight_mode = 'auto'            

        body2cam = transformations.quaternion_from_euler( cam_yaw * d2r,
                                                          cam_pitch * d2r,
                                                          cam_roll * d2r,
                                                          'rzyx')

        #print 'att:', [yaw_rad, pitch_rad, roll_rad]
        ned2body = transformations.quaternion_from_euler(yaw_rad,
                                                         pitch_rad,
                                                         roll_rad,
                                                         'rzyx')
        body2ned = transformations.quaternion_inverse(ned2body)
        
        #print 'ned2body(q):', ned2body
        ned2cam_q = transformations.quaternion_multiply(ned2body, body2cam)
        ned2cam = np.matrix(transformations.quaternion_matrix(np.array(ned2cam_q))[:3,:3]).T
        #print 'ned2cam:', ned2cam
        R = ned2proj.dot( ned2cam )
        rvec, jac = cv2.Rodrigues(R)
        ned = navpy.lla2ned( lat_deg, lon_deg, altitude_m,
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

        # Create hud draw space
        hud_frame = frame_undist.copy()

        hud.update_proj(PROJ)
        hud.update_ned(ned)
        hud.update_frame(hud_frame)
        
        hud.draw_horizon()
        hud.draw_compass_points()
        hud.draw_pitch_ladder(yaw_rad, beta_rad)
        hud.draw_flight_path_marker(pitch_rad, alpha_rad, yaw_rad, beta_rad)
        hud.draw_astro(float(flight_gps_lat(time)),
                       float(flight_gps_lon(time)),
                       float(flight_gps_alt(time)),
                       float(flight_gps_unixtime(time)))
        hud.draw_airports()
        hud.draw_velocity_vector([vn, ve, vd])
        if args.airspeed_units == 'mps':
            airspeed = airspeed_kt * kt2mps
        else:
            airspeed = airspeed_kt
        hud.draw_speed_tape(airspeed, ap_speed, args.airspeed_units.capitalize(), flight_mode)
        if args.altitude_units == 'm':
            altitude = altitude_m
        else:
            altitude = altitude_m * m2ft
        hud.draw_altitude_tape(altitude, ap_alt, args.altitude_units.capitalize(), flight_mode)
        if flight_mode == 'manual':
            hud.draw_nose(body2ned)
        else:
            hud.draw_vbars(yaw_rad, pitch_rad, ap_roll, ap_pitch)
            hud.draw_heading_bug(ap_hdg)
            hud.draw_bird(yaw_rad, pitch_rad, roll_rad)
            hud.draw_course(vn, ve)

        # weighted add of the HUD frame with the original frame to
        # emulate alpha blending
        alpha = args.alpha
        if alpha < 0: alpha = 0
        if alpha > 1: alpha = 1
        cv2.addWeighted(hud_frame, alpha, frame_undist, 1 - alpha, 0, hud_frame)
        
        cv2.imshow('hud', hud_frame)
        output.write(hud_frame)
        
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

output.release()
cv2.destroyAllWindows()

# now run ffmpeg as an external command to combine original audio
# track with new overlay video

# ex: ffmpeg -i opencv.avi -i orig.mov -c copy -map 0:v -map 1:a final.avi

from subprocess import call
result = call(["ffmpeg", "-i", tmp_movie, "-i", args.movie, "-c", "copy", "-map", "0:v", "-map", "1:a", output_movie])
print "ffmpeg result code:", result
if result == 0:
    print "removing temp movie:", tmp_movie
    os.remove(tmp_movie)
    print "output movie:", output_movie

if args.plot:
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
