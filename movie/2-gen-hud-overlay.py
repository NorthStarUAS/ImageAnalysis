#!/usr/bin/python

# find our custom built opencv first
import sys
sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")
import cv2

import argparse
import math
from matplotlib import pyplot as plt 
import navpy
import numpy as np
import os
import re
from scipy import interpolate # strait up linear interpolation, nothing fancy

from props import PropertyNode
import props_json

from aurauas.flightdata import flight_loader, flight_interp

sys.path.append('../lib')
import transformations

import cam_calib
import hud
import features

# helpful constants
d2r = math.pi / 180.0

# default sizes of primatives
render_w = 1920
render_h = 1080

parser = argparse.ArgumentParser(description='correlate movie data to flight data.')
parser.add_argument('--flight', help='load specified aura flight log')
parser.add_argument('--movie', required=True, help='original movie')
parser.add_argument('--select-cam', type=int, help='select camera calibration')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--alpha', type=float, default=0.7, help='hud alpha blend')
parser.add_argument('--resample-hz', type=float, default=30.0, help='resample rate (hz)')
parser.add_argument('--start-time', type=float, help='fast forward to this flight log time before begining movie render.')
parser.add_argument('--stop-count', type=int, default=1, help='how many non-frames to absorb before we decide the movie is over')
parser.add_argument('--plot', action='store_true', help='Plot stuff at the end of the run')
parser.add_argument('--auto-switch', choices=['old', 'new', 'none', 'on'], default='new', help='auto/manual switch logic helper')
parser.add_argument('--airspeed-units', choices=['kt', 'mps'], default='kt', help='display units for airspeed')
parser.add_argument('--altitude-units', choices=['ft', 'm'], default='ft', help='display units for airspeed')
parser.add_argument('--aileron-scale', type=float, default=1.0, help='useful for reversing aileron in display')
parser.add_argument('--elevator-scale', type=float, default=1.0, help='useful for reversing elevator in display')
parser.add_argument('--features', help='feature database')
args = parser.parse_args()

r2d = 180.0 / math.pi
counter = 0
stop_count = 0

# pathname work
abspath = os.path.abspath(args.movie)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.movie)
movie_log = filename + ".csv"
camera_config = dirname + "/camera.json"
# combinations that seem to work on linux
# ext = avi, fourcc = MJPG
# ext = avi, fourcc = XVID
# ext = mov, fourcc = MP4V

tmp_movie = filename + "_tmp.mov"
output_movie = filename + "_hud.mov"

# load config file if it exists
config = PropertyNode()
props_json.load(camera_config, config)
cam_yaw = config.getFloatEnum('mount_ypr', 0)
cam_pitch = config.getFloatEnum('mount_ypr', 1)
cam_roll = config.getFloatEnum('mount_ypr', 2)

# setup camera calibration and distortion coefficients
if args.select_cam:
    # set the camera calibration from known preconfigured setups
    name, K, dist = cam_calib.set_camera_calibration(args.select_cam)
    config.setString('name', name)
    config.setFloat("fx", K[0][0])
    config.setFloat("fy", K[1][1])
    config.setFloat("cu", K[0][2])
    config.setFloat("cv", K[1][2])
    for i, d in enumerate(dist):
        config.setFloatEnum("dist_coeffs", i, d)
    props_json.save(camera_config, config)
else:
    # load the camera calibration from the config file
    name = config.getString('name')
    size = config.getLen("dist_coeffs")
    dist = []
    for i in range(size):
        dist.append( config.getFloatEnum("dist_coeffs", i) )
    K = np.zeros( (3,3) )
    K[0][0] = config.getFloat("fx")
    K[1][1] = config.getFloat("fy")
    K[0][2] = config.getFloat("cu")
    K[1][2] = config.getFloat("cv")
    K[2][2] = 1.0
    print 'Camera:', name

# load movie log
movie = []
with open(movie_log, 'rb') as f:
    for line in f:
        movie.append( re.split('[,\s]+', line.rstrip()) )

if 'recalibrate' in args:
    recal_file = args.recalibrate
else:
    recal_file = None
data, flight_format = flight_loader.load(args.flight, recal_file)
print "imu records:", len(data['imu'])
print "gps records:", len(data['gps'])
if 'air' in data:
    print "airdata records:", len(data['air'])
print "filter records:", len(data['filter'])
if 'pilot' in data:
    print "pilot records:", len(data['pilot'])
if 'act' in data:
    print "act records:", len(data['act'])
if len(data['imu']) == 0 and len(data['gps']) == 0:
    print "not enough data loaded to continue."
    quit()

interp = flight_interp.FlightInterpolate()
interp.build(data)

# set approximate camera orienation (front, down, and rear supported)
cam_facing = 'front'

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
movie_len = xmax - xmin
for x in np.linspace(xmin, xmax, movie_len*args.resample_hz):
    if cam_facing == 'front' or cam_facing == 'down':
        #movie_interp.append( [x, movie_spl_roll(x)] )
        movie_interp.append( [x, -movie_spl_yaw(x)] ) # test, fixme
    else:
        movie_interp.append( [x, -movie_spl_roll(x)] )
print "movie len:", len(movie_interp)

# resample flight data
flight_interp = []
if cam_facing == 'front' or cam_facing == 'rear':
    #y_spline = interp.imu_p     # front/rear facing camera
    y_spline = interp.imu_r     # front/rear facing camera, test fixme
else:
    y_spline = interp.imu_r     # down facing camera

# run correlation over filter time span
x = interp.imu_time
flight_min = x.min()
flight_max = x.max()
print "flight range = %.3f - %.3f (%.3f)" % (flight_min, flight_max, flight_max-flight_min)
time = flight_max - flight_min
for x in np.linspace(flight_min, flight_max, time*args.resample_hz):
    flight_interp.append( [x, y_spline(x)] )
print "flight len:", len(flight_interp)

# compute best correlation between movie and flight data logs
movie_interp = np.array(movie_interp, dtype=float)
flight_interp = np.array(flight_interp, dtype=float)
ycorr = np.correlate(flight_interp[:,1], movie_interp[:,1], mode='full')

# display some stats/info
max_index = np.argmax(ycorr)
print "max index:", max_index

# shift = np.argmax(ycorr) - len(flight_interp)
# print "shift (pos):", shift
# start_diff = flight_interp[0][0] - movie_interp[0][0]
# print "start time diff:", start_diff
# time_shift = start_diff - (shift/args.resample_hz)
# print "movie time shift:", time_shift

# need to subtract movie_len off peak point time because of how
# correlate works and shifts against every possible overlap
shift_sec = np.argmax(ycorr) / args.resample_hz - movie_len
print "shift (sec):", shift_sec
print flight_interp[0][0], movie_interp[0][0]
start_diff = flight_interp[0][0] - movie_interp[0][0]
print "start time diff:", start_diff
time_shift = start_diff + shift_sec
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
for x in np.linspace(tmin, tmax, (tmax-tmin)*args.resample_hz):
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

# quick estimate ground elevation
ground_m = None
for f in data['filter']:
    if ground_m == None:
        ground_m = f.alt
    if f.alt < ground_m:
        ground_m = f.alt
print "ground est:", ground_m

if args.plot:
    # reformat the data
    flight_imu = []
    for imu in data['imu']:
        flight_imu.append([ imu.time, imu.p, imu.q, imu.r ])
    flight_imu = np.array(flight_imu)
    
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

# adjust K for output scale.
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

hud1.set_ground_m(ground_m)
hud2.set_ground_m(ground_m)

if args.features:
    feats = features.load(args.features, ref)
    hud1.update_features(feats)
else:
    feats = []

print "Opening ", args.movie
try:
    capture = cv2.VideoCapture(args.movie)
except:
    print "error opening video"

capture.read()
counter += 1
print "ok reading first frame"

fps = capture.get(cv2.CAP_PROP_FPS)
print "fps = %.2f" % fps
fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale )
h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale )
hud1.set_render_size(w, h)
hud2.set_render_size(w, h)

#outfourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
#outfourcc = cv2.cv.CV_FOURCC('H', '2', '6', '4')
#outfourcc = cv2.cv.CV_FOURCC('X', '2', '6', '4')
#outfourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')
outfourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
print outfourcc, fps, w, h
output = cv2.VideoWriter(tmp_movie, outfourcc, fps, (w, h), isColor=True)

last_time = 0.0

# set primative sizes based on rendered resolution.
size = math.sqrt(h*h + w*w)
hud1.set_line_width( int(round(size/1000.0)) )
hud1.set_font_size( size / 1400.0 )
hud1.set_color( hud.green2 )
hud1.set_units( args.airspeed_units, args.altitude_units)

hud2.set_line_width( int(round(size/1000.0)) )
hud2.set_font_size( size / 1400.0 )
hud2.set_color( hud.red2 )
hud2.set_units( args.airspeed_units, args.altitude_units)

filt_alt = None

if time_shift > 0:
    # catch up the flight path history (in case the movie starts
    # mid-flight.)  Note: flight_min is the starting time of the filter data
    # set.
    print 'seeding flight track ...'
    for time in np.arange(flight_min, time_shift, 1.0 / float(fps)):
        lat_deg = float(interp.filter_lat(time))*r2d
        lon_deg = float(interp.filter_lon(time))*r2d
        #altitude_m = float(interp.air_true_alt(time))
        altitude_m = float(interp.filter_alt(time))
        ned = navpy.lla2ned( lat_deg, lon_deg, altitude_m,
                             ref[0], ref[1], ref[2] )
        hud1.update_time(time, interp.gps_unixtime(time))
        hud1.update_ned(ned)
    
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
    lat_deg = float(interp.filter_lat(time))*r2d
    lon_deg = float(interp.filter_lon(time))*r2d
    #altitude_m = float(interp.air_true_alt(time))
    altitude_m = float(interp.filter_alt(time))
    if filt_alt == None:
        filt_alt = altitude_m
    else:
        filt_alt = 0.95 * filt_alt + 0.05 * altitude_m
    if interp.air_speed != None:
        airspeed_kt = float(interp.air_speed(time))
    else:
        airspeed_kt = 0.0
    #if False: # 'alpha' in data['air'][0]:
    if interp.air_alpha and interp.air_beta:
        alpha_rad = float(interp.air_alpha(time))*d2r
        beta_rad = float(interp.air_beta(time))*d2r
        print alpha_rad, beta_rad
    else:
        alpha_rad = None
        beta_rad = None
    if interp.ap_hdgx:
        ap_hdgx = float(interp.ap_hdgx(time))
        ap_hdgy = float(interp.ap_hdgy(time))
        ap_hdg = math.atan2(ap_hdgy, ap_hdgx)*r2d
        ap_roll = float(interp.ap_roll(time))
        ap_pitch = float(interp.ap_pitch(time))
        ap_speed = float(interp.ap_speed(time))
        ap_alt_ft = float(interp.ap_alt(time))
    if interp.pilot_ail:
        pilot_ail = float(interp.pilot_ail(time))
        pilot_ele = float(interp.pilot_ele(time))
        pilot_thr = float(interp.pilot_thr(time))
        pilot_rud = float(interp.pilot_rud(time))
        auto_switch = float(interp.pilot_auto(time))
    else:
        auto_switch = 0
    if interp.act_ail:
        act_ail = float(interp.act_ail(time)) * args.aileron_scale
        act_ele = float(interp.act_ele(time)) * args.elevator_scale
        act_thr = float(interp.act_thr(time))
        act_rud = float(interp.act_rud(time))

    if args.auto_switch == 'none':
        flight_mode = 'manual'
    elif (args.auto_switch == 'new' and auto_switch < 0) or (args.auto_switch == 'old' and auto_switch > 0):
        flight_mode = 'manual'
    elif args.auto_switch == 'on':
        flight_mode = 'auto'
    else:
        flight_mode = 'auto'            

    body2cam = transformations.quaternion_from_euler( cam_yaw * d2r,
                                                      cam_pitch * d2r,
                                                      cam_roll * d2r,
                                                      'rzyx')

    # this function modifies the parameters you pass in so, avoid
    # getting our data changed out from under us, by forcing copies (a
    # = b, wasn't sufficient, but a = float(b) forced a copy.
    tmp_yaw = float(yaw_rad)
    tmp_pitch = float(pitch_rad)
    tmp_roll = float(roll_rad)    
    ned2body = transformations.quaternion_from_euler(tmp_yaw,
                                                     tmp_pitch,
                                                     tmp_roll,
                                                     'rzyx')
    body2ned = transformations.quaternion_inverse(ned2body)

    #print 'ned2body(q):', ned2body
    ned2cam_q = transformations.quaternion_multiply(ned2body, body2cam)
    ned2cam = np.matrix(transformations.quaternion_matrix(np.array(ned2cam_q))[:3,:3]).T
    #print 'ned2cam:', ned2cam
    R = ned2proj.dot( ned2cam )
    rvec, jac = cv2.Rodrigues(R)
    ned = navpy.lla2ned( lat_deg, lon_deg, filt_alt,
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

    hud1.update_time(time, interp.gps_unixtime(time))
    hud1.update_proj(PROJ)
    hud1.update_cam_att(cam_yaw, cam_pitch, cam_roll)
    hud1.update_ned(ned)
    hud1.update_lla([lat_deg, lon_deg, altitude_m])
    hud1.update_vel(vn, ve, vd)
    hud1.update_att_rad(roll_rad, pitch_rad, yaw_rad)
    hud1.update_airdata(airspeed_kt, altitude_m, alpha_rad, beta_rad)
    if interp.ap_hdgx:
        hud1.update_ap(flight_mode, ap_roll, ap_pitch, ap_hdg,
                       ap_speed, ap_alt_ft)
    else:
        hud1.update_ap(flight_mode, 0.0, 0.0, 0.0, 0.0, 0.0)
    if interp.pilot_ail:
        hud1.update_pilot(pilot_ail, pilot_ele, pilot_thr, pilot_rud)
    if interp.act_ail:
        hud1.update_act(act_ail, act_ele, act_thr, act_rud)
    if time >= flight_min and time <= flight_max:
        # only draw hud for time range when we have actual flight data
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
        config.setFloatEnum('mount_ypr', 0, cam_yaw)
        props_json.save(camera_config, config)
    elif key == ord('Y'):
        cam_yaw -= 0.5
        config.setFloatEnum('mount_ypr', 0, cam_yaw)
        props_json.save(camera_config, config)
    elif key == ord('p'):
        cam_pitch += 0.5
        config.setFloatEnum('mount_ypr', 1, cam_pitch)
        props_json.save(camera_config, config)
    elif key == ord('P'):
        cam_pitch -= 0.5
        config.setFloatEnum('mount_ypr', 1, cam_pitch)
        props_json.save(camera_config, config)
    elif key == ord('r'):
        cam_roll -= 0.5
        config.setFloatEnum('mount_ypr', 2, cam_roll)
        props_json.save(camera_config, config)
    elif key == ord('R'):
        cam_roll += 0.5
        config.setFloatEnum('mount_ypr', 2, cam_roll)
        props_json.save(camera_config, config)

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

