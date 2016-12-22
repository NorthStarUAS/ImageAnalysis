#!/usr/bin/python

# find our custom built opencv first
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")
import cv2

import argparse
import math
from matplotlib import pyplot as plt 
import navpy
import numpy as np
import os
import re
from scipy import interpolate # strait up linear interpolation, nothing fancy

from props import PropertyNode, getNode
import props_json

from nav.data import flight_data

sys.path.append('../lib')
import transformations

import hud
import interp

# helpful constants
d2r = math.pi / 180.0

# default sizes of primatives
render_w = 1920
render_h = 1080

parser = argparse.ArgumentParser(description='correlate movie data to flight data.')
parser.add_argument('--movie', required=True, help='original movie')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--alpha', type=float, default=0.7, help='hud alpha blend')
parser.add_argument('--resample-hz', type=float, default=30.0, help='resample rate (hz)')
parser.add_argument('--flight', help='Aura flight log directory')
parser.add_argument('--stop-count', type=int, default=1, help='how many non-frames to absorb before we decide the movie is over')
parser.add_argument('--plot', help='Plot stuff at the end of the run')
parser.add_argument('--auto-switch', choices=['old', 'new', 'none'], default='new', help='auto/manual switch logic helper')
parser.add_argument('--airspeed-units', choices=['kt', 'mps'], default='kt', help='display units for airspeed')
parser.add_argument('--altitude-units', choices=['ft', 'm'], default='ft', help='display units for airspeed')
parser.add_argument('--start-time', type=float, help='fast forward to this flight log time before begining movie render.')
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

if args.flight:
    data = flight_data.load('aura', args.flight)
    interp.build(data)
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
flight_interp = []
# y_spline = interp.imu_r     # down facing camera
y_spline = interp.imu_p     # front facing camera

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

if args.movie:
    # Mobius 1080p
    # K = np.array( [[1362.1,    0.0, 980.8],
    #                [   0.0, 1272.8, 601.3],
    #                [   0.0,    0.0,   1.0]] )
    # dist = [-0.36207197, 0.14627927, -0.00674558, 0.0008926, -0.02635695]

    # Mobius UMN-003 1920x1080
    # K = np.array( [[ 1401.21111735,     0.       ,    904.25404757],
    #                [    0.        ,  1400.2530882,    490.12157373],
    #                [    0.        ,     0.       ,      1.        ]] )
    # dist = [-0.39012303,  0.19687255, -0.00069657,  0.00465592, -0.05845262]

    # RunCamHD2 1920x1080
    K = np.array( [[ 971.96149426,   0.        , 957.46750602],
                   [   0.        , 971.67133264, 516.50578382],
                   [   0.        ,   0.        ,   1.        ]] )
    dist = [-0.26910665, 0.10580125, 0.00048417, 0.00000925, -0.02321387]

    # Runcamhd2 1920x1440
    # K = np.array( [[ 1296.11187055,     0.        ,   955.43024994],
    #                [    0.        ,  1296.01457451,   691.47053988],
    #                [    0.        ,     0.        ,     1.        ]] )
    # dist = [-0.28250371, 0.14064665, 0.00061846, 0.00014488, -0.05106045]
    
    K = K * args.scale
    K[2,2] = 1.0

    # overlay hud(s)
    hud1 = hud.HUD(K)
    hud2 = hud.HUD(K)
    
    # these are fixed tranforms between ned and camera reference systems
    proj2ned = np.array( [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                         dtype=float )
    ned2proj = np.linalg.inv(proj2ned)

    #cam_ypr = [-3.0, -12.0, -3.0] # yaw, pitch, roll
    #ref = [44.7260320000, -93.0771072000, 0]
    ref = [ data['gps'][0].lat, data['gps'][0].lon, 0.0 ]
    hud1.set_ned_ref(data['gps'][0].lat, data['gps'][0].lon)
    hud2.set_ned_ref(data['gps'][0].lat, data['gps'][0].lon)
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
    hud1.set_render_size(w, h)
    hud2.set_render_size(w, h)
    
    #outfourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    #outfourcc = cv2.cv.CV_FOURCC('H', '2', '6', '4')
    #outfourcc = cv2.cv.CV_FOURCC('X', '2', '6', '4')
    #outfourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')
    outfourcc = cv2.cv.CV_FOURCC('M', 'P', '4', 'V')
    print outfourcc, fps, w, h
    output = cv2.VideoWriter(tmp_movie, outfourcc, fps, (w, h), isColor=True)

    last_time = 0.0

    # set primative sizes based on rendered resolution.
    size = math.sqrt(float(h)*float(w))
    hud1.set_line_width( int(round(size/500.0)) )
    hud1.set_font_size( size / 1000.0 )
    hud1.set_color( hud.green2 )
    hud1.set_units( args.airspeed_units, args.altitude_units)

    hud2.set_line_width( int(round(float(h) / 400.0)) )
    hud2.set_font_size( float(h) / 900.0 )
    hud2.set_color( hud.red2 )
    hud2.set_units( args.airspeed_units, args.altitude_units)
    
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
        if args.start_time and time < args.start_time:
            continue
        vn = interp.filter_vn(time)
        ve = interp.filter_ve(time)
        vd = interp.filter_vd(time)
        #yaw_rad = interp.filter_yaw(time)*d2r 
        psix = interp.filter_psix(time)
        psiy = interp.filter_psiy(time)
        yaw_rad = math.atan2(psiy, psix)
        pitch_rad = interp.filter_the(time)
        roll_rad = interp.filter_phi(time)
        lat_deg = float(interp.gps_lat(time))
        lon_deg = float(interp.gps_lon(time))
        altitude_m = float(interp.air_true_alt(time))
        airspeed_kt = float(interp.air_speed(time))
        if False: # 'alpha' in data['air'][0]:
            alpha_rad = float(interp.air_alpha(time))*d2r
            beta_rad = float(interp.air_beta(time))*d2r
        else:
            alpha_rad = None
            beta_rad = None
        ap_hdgx = float(interp.ap_hdgx(time))
        ap_hdgy = float(interp.ap_hdgy(time))
        ap_hdg = math.atan2(ap_hdgy, ap_hdgx)*r2d
        ap_roll = float(interp.ap_roll(time))
        ap_pitch = float(interp.ap_pitch(time))
        ap_speed = float(interp.ap_speed(time))
        ap_alt_ft = float(interp.ap_alt(time))
        pilot_ail = float(interp.pilot_ail(time))
        pilot_ele = float(interp.pilot_ele(time))
        pilot_thr = float(interp.pilot_thr(time))
        pilot_rud = float(interp.pilot_rud(time))
        auto_switch = float(interp.pilot_auto(time))
        act_ail = float(interp.act_ail(time))
        act_ele = float(interp.act_ele(time))
        act_thr = float(interp.act_thr(time))
        act_rud = float(interp.act_rud(time))
        if args.auto_switch == 'none':
            flight_mode = 'manual'
        elif (args.auto_switch == 'new' and auto_switch < 0) or (args.auto_switch == 'old' and auto_switch > 0):
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
        hud1_frame = frame_undist.copy()

        hud1.update_proj(PROJ)
        hud1.update_cam_att(cam_yaw, cam_pitch, cam_roll)
        hud1.update_ned(ned)
        hud1.update_lla([lat_deg, lon_deg, altitude_m])
        hud1.update_time(time, interp.gps_unixtime(time))
        hud1.update_vel(vn, ve, vd)
        hud1.update_att_rad(roll_rad, pitch_rad, yaw_rad)
        hud1.update_airdata(airspeed_kt, altitude_m, alpha_rad, beta_rad)
        hud1.update_ap(flight_mode, ap_roll, ap_pitch, ap_hdg,
                        ap_speed, ap_alt_ft)
        hud1.update_pilot(pilot_ail, pilot_ele, pilot_thr, pilot_rud)
        hud1.update_act(act_ail, act_ele, act_thr, act_rud)
        hud1.update_frame(hud1_frame)
        hud1.draw()
        
        # weighted add of the HUD frame with the original frame to
        # emulate alpha blending
        alpha = args.alpha
        if alpha < 0: alpha = 0
        if alpha > 1: alpha = 1
        cv2.addWeighted(hud1_frame, alpha, frame_undist, 1 - alpha, 0, hud1_frame)
        
        cv2.imshow('hud', hud1_frame)
        output.write(hud1_frame)
        
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
    # down facing:
    # plt.plot(flight_imu[:,0], flight_imu[:,3]*r2d, label='flight data log')
    # front facing:
    plt.plot(flight_imu[:,0], flight_imu[:,1]*r2d, label='flight data log')
    plt.legend()

    plt.figure(2)
    plt.plot(ycorr)

    plt.figure(3)
    plt.ylabel('pitch rate (deg per sec)')
    plt.xlabel('flight time (sec)')
    plt.plot(movie[:,0] + time_shift, (movie[:,3]/qratio)*r2d, label='estimate from flight movie')
    plt.plot(flight_imu[:,0], flight_imu[:,2]*r2d, label='flight data log')
    plt.legend()

    plt.figure(4)
    plt.ylabel('yaw rate (deg per sec)')
    plt.xlabel('flight time (sec)')
    plt.plot(movie[:,0] + time_shift, (movie[:,4]/rratio)*r2d, label='estimate from flight movie')
    plt.plot(flight_imu[:,0], flight_imu[:,3]*r2d, label='flight data log')
    plt.legend()

    plt.show()
