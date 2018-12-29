#!/usr/bin/python3

import argparse
import copy
import cv2
import skvideo.io               # pip3 install scikit-video
import math
import navpy
import numpy as np
import os
import re
import sys

from props import PropertyNode
import props_json

from aurauas.flightdata import flight_loader, flight_interp

sys.path.append('../lib')
import transformations

import correction
import correlate
import hud
import hud_glass
import features

# helpful constants
d2r = math.pi / 180.0
r2d = 180.0 / math.pi

# default sizes of primatives
render_w = 1920
render_h = 1080

# configure
experimental_overlay = False

parser = argparse.ArgumentParser(description='correlate movie data to flight data.')
parser.add_argument('--flight', help='load specified aura flight log')
parser.add_argument('--movie', required=True, help='original movie')
parser.add_argument('--camera', help='select camera calibration file')
parser.add_argument('--cam-mount', choices=['forward', 'down', 'rear'],
                    default='forward',
                    help='approximate camera mounting orientation')
parser.add_argument('--rot180', action='store_true')
parser.add_argument('--scale', type=float, default=1.0, help='scale input')
parser.add_argument('--scale-preview', type=float, default=0.25,
                    help='scale preview')
parser.add_argument('--alpha', type=float, default=0.7, help='hud alpha blend')
parser.add_argument('--resample-hz', type=float, default=60.0,
                    help='resample rate (hz)')
parser.add_argument('--start-time', type=float, help='fast forward to this flight log time before begining movie render.')
parser.add_argument('--time-shift', type=float, help='skip autocorrelation and use this offset time')
parser.add_argument('--plot', action='store_true', help='Plot stuff at the end of the run')
parser.add_argument('--auto-switch', choices=['old', 'new', 'none', 'on'], default='new', help='auto/manual switch logic helper')
parser.add_argument('--airspeed-units', choices=['kt', 'mps'], default='kt', help='display units for airspeed')
parser.add_argument('--altitude-units', choices=['ft', 'm'], default='ft', help='display units for airspeed')
parser.add_argument('--aileron-scale', type=float, default=1.0, help='useful for reversing aileron in display')
parser.add_argument('--elevator-scale', type=float, default=1.0, help='useful for reversing elevator in display')
parser.add_argument('--rudder-scale', type=float, default=1.0, help='useful for reversing rudder in display')
parser.add_argument('--flight-track-seconds', type=float, default=600.0, help='how many seconds of flight track to draw')
parser.add_argument('--keep-tmp-movie', action='store_true', help='Keep the temp movie')
parser.add_argument('--correction', help='correction table')
parser.add_argument('--features', help='feature database')
args = parser.parse_args()

counter = 0

# pathname work
abspath = os.path.abspath(args.movie)
filename, ext = os.path.splitext(abspath)
dirname = os.path.dirname(args.movie)
movie_log = filename + ".csv"
local_config = dirname + "/camera.json"

# combinations that seem to work on linux
# ext = avi, fourcc = MJPG
# ext = avi, fourcc = XVID
# ext = m4v (was mov), fourcc = MP4V

ext = 'mp4'
tmp_movie = filename + "_tmp." + ext
output_movie = filename + "_hud.mov"

config = PropertyNode()

if args.camera:
    # seed the camera calibration and distortion coefficients from a
    # known camera config
    print('Setting camera config from:', args.camera)
    props_json.load(args.camera, config)
    config.setString('name', args.camera)
    props_json.save(local_config, config)
elif os.path.exists(local_config):
    # load local config file if it exists
    props_json.load(local_config, config)
    
name = config.getString('name')
config.setLen('mount_ypr', 3, 0.0)
cam_yaw = config.getFloatEnum('mount_ypr', 0)
cam_pitch = config.getFloatEnum('mount_ypr', 1)
cam_roll = config.getFloatEnum('mount_ypr', 2)

K_list = []
for i in range(9):
    K_list.append( config.getFloatEnum('K', i) )
K = np.copy(np.array(K_list)).reshape(3,3)
dist = []
for i in range(5):
    dist.append( config.getFloatEnum("dist_coeffs", i) )

print('Camera:', name)
print('K:\n', K)
print('dist:', dist)

# adjust K for output scale.
K = K * args.scale
K[2,2] = 1.0

if args.correction:
    correction.load(args.correction)
    
if 'recalibrate' in args:
    recal_file = args.recalibrate
else:
    recal_file = None
data, flight_format = flight_loader.load(args.flight, recal_file)
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

interp = flight_interp.FlightInterpolate()
interp.build(data)

time_shift, flight_min, flight_max = \
    correlate.sync_clocks(data, interp, movie_log, hz=args.resample_hz,
                          cam_mount=args.cam_mount,
                          force_time_shift=args.time_shift, plot=args.plot)

# quick estimate ground elevation
sum = 0.0
count = 0
for f in data['filter']:
    if interp.air_speed(f.time) < 5.0:
        sum += f.alt
        count += 1
if count > 0:
    ground_m = sum / float(count)
else:
    ground_m = data['filter'][0].alt
print("ground est:", ground_m)

# overlay hud(s)
hud1 = hud_glass.HUD(K)
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
print('ned ref:', ref)

print('temporarily disabling airport loading')
hud1.load_airports()

hud1.set_ground_m(ground_m)
hud2.set_ground_m(ground_m)

if args.features:
    feats = features.load(args.features, ref)
    hud1.update_features(feats)
else:
    feats = []

metadata = skvideo.io.ffprobe(args.movie)
#print(metadata.keys())
#print(json.dumps(metadata["video"], indent=4))
fps_string = metadata['video']['@avg_frame_rate']
(num, den) = fps_string.split('/')
fps = float(num) / float(den)
codec = metadata['video']['@codec_long_name']
w = int(round(int(metadata['video']['@width']) * args.scale))
h = int(round(int(metadata['video']['@height']) * args.scale))
print('fps:', fps)
print('codec:', codec)
print('output size:', w, 'x', h)

hud1.set_render_size(w, h)
hud2.set_render_size(w, h)

print("Opening ", args.movie)
reader = skvideo.io.FFmpegReader(args.movie, inputdict={}, outputdict={})

inputdict = {
    '-r': str(fps)
}

lossless = {
    # See all options: https://trac.ffmpeg.org/wiki/Encode/H.264
    '-vcodec': 'libx264',  # use the h.264 codec
    '-crf': '0',           # set the constant rate factor to 0, (lossless)
    '-preset': 'veryslow', # maximum compression
    '-r': str(fps)         # match input fps
}

sane = {
    # See all options: https://trac.ffmpeg.org/wiki/Encode/H.264
    '-vcodec': 'libx264',  # use the h.264 codec
    '-crf': '17',          # visually lossless (or nearly so)
    '-preset': 'medium',   # default compression
    '-r': str(fps)         # match input fps
}

writer = skvideo.io.FFmpegWriter(tmp_movie, inputdict=inputdict, outputdict=sane)

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
    print('seeding flight track ...')
    for time in np.arange(flight_min, time_shift, 1.0 / float(fps)):
        lat_deg = float(interp.filter_lat(time))*r2d
        lon_deg = float(interp.filter_lon(time))*r2d
        #altitude_m = float(interp.air_true_alt(time))
        altitude_m = float(interp.filter_alt(time))
        ned = navpy.lla2ned( lat_deg, lon_deg, altitude_m,
                             ref[0], ref[1], ref[2] )
        hud1.update_time(time, interp.gps_unixtime(time))
        hud1.update_ned(ned, args.flight_track_seconds)

shift_mod_hack = False
for frame in reader.nextFrame():
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    if args.rot180:
        frame = np.rot90(frame)
        frame = np.rot90(frame)
        
    time = float(counter) / fps + time_shift
    print("frame: ", counter, "%.3f" % time, 'time shift:', time_shift)
    
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
    if args.correction:
        yaw_rad += correction.yaw_interp(time)
        pitch_rad += correction.pitch_interp(time)
        roll_rad += correction.roll_interp(time)
    lat_deg = interp.filter_lat(time)*r2d
    lon_deg = interp.filter_lon(time)*r2d
    #altitude_m = float(interp.air_true_alt(time))
    altitude_m = interp.filter_alt(time)
    if filt_alt == None:
        filt_alt = altitude_m
    else:
        filt_alt = 0.95 * filt_alt + 0.05 * altitude_m
    if interp.air_speed:
        airspeed_kt = interp.air_speed(time)
    else:
        airspeed_kt = 0.0
    if interp.air_wind_dir:
        wind_deg = interp.air_wind_dir(time)
        wind_kt = interp.air_wind_speed(time)
    if interp.air_alpha and interp.air_beta:
        alpha_rad = float(interp.air_alpha(time))*d2r
        beta_rad = float(interp.air_beta(time))*d2r
        #print alpha_rad, beta_rad
    else:
        alpha_rad = None
        beta_rad = None
        #print 'no alpha/beta'
    if interp.ap_hdgx:
        ap_hdgx = float(interp.ap_hdgx(time))
        ap_hdgy = float(interp.ap_hdgy(time))
        ap_hdg = math.atan2(ap_hdgy, ap_hdgx)*r2d
        ap_roll = float(interp.ap_roll(time))
        ap_pitch = float(interp.ap_pitch(time))
        ap_speed = float(interp.ap_speed(time))
        ap_alt_ft = float(interp.ap_alt(time))
    if interp.pilot_ail:
        pilot_ail = float(interp.pilot_ail(time)) * args.aileron_scale
        pilot_ele = float(interp.pilot_ele(time)) * args.elevator_scale
        pilot_thr = float(interp.pilot_thr(time))
        pilot_rud = float(interp.pilot_rud(time)) * args.rudder_scale
        auto_switch = float(interp.pilot_auto(time))
    else:
        auto_switch = 0
    if interp.act_ail:
        act_ail = float(interp.act_ail(time)) * args.aileron_scale
        act_ele = float(interp.act_ele(time)) * args.elevator_scale
        act_thr = float(interp.act_thr(time))
        act_rud = float(interp.act_rud(time)) * args.rudder_scale

    if args.auto_switch == 'none':
        flight_mode = 'manual'
    elif (args.auto_switch == 'new' and auto_switch < 0) or (args.auto_switch == 'old' and auto_switch > 0):
        flight_mode = 'manual'
    elif args.auto_switch == 'on':
        flight_mode = 'auto'
    else:
        flight_mode = 'auto'            

    if interp.excite_mode:
        excite_mode = float(interp.excite_mode(time))
        test_index = float(interp.test_index(time))        

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
    if args.correction:
        ned[0] += correction.north_interp(time)
        ned[1] += correction.east_interp(time)
        ned[2] += correction.down_interp(time)
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
    if not experimental_overlay:
        hud1_frame = frame_undist.copy()
    else:
        hud1_frame = np.zeros((frame_undist.shape), np.uint8)

    hud1.update_time(time, interp.gps_unixtime(time))
    if 'event' in data:
        hud1.update_events(data['event'])
    if interp.excite_mode:
        hud1.update_test_index(excite_mode, test_index)
    hud1.update_proj(PROJ)
    hud1.update_cam_att(cam_yaw, cam_pitch, cam_roll)
    hud1.update_ned(ned, args.flight_track_seconds)
    hud1.update_lla([lat_deg, lon_deg, altitude_m])
    hud1.update_vel(vn, ve, vd)
    hud1.update_att_rad(roll_rad, pitch_rad, yaw_rad)
    if interp.air_wind_dir:
        hud1.update_airdata(airspeed_kt, altitude_m, wind_deg, wind_kt, alpha_rad, beta_rad)
    else:
        hud1.update_airdata(airspeed_kt, altitude_m)
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

    if not experimental_overlay:
        # weighted add of the HUD frame with the original frame to
        # emulate alpha blending
        alpha = args.alpha
        if alpha < 0: alpha = 0
        if alpha > 1: alpha = 1
        cv2.addWeighted(hud1_frame, alpha, frame_undist, 1 - alpha, 0, hud1_frame)
    else:
        # Now create a mask of hud and create its inverse mask also
        tmp = cv2.cvtColor(hud1_frame, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(tmp, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the hud from the original image
        tmp_bg = cv2.bitwise_and(frame_undist, frame_undist, mask=mask_inv)

        # Put hud onto the main image
        hud1_frame = cv2.add(tmp_bg, hud1_frame)

    # cv2.imshow('hud', hud1_frame)
    cv2.imshow('hud', cv2.resize(hud1_frame, None, fx=args.scale_preview, fy=args.scale_preview))
    writer.writeFrame(hud1_frame[:,:,::-1])  #write the frame as RGB not BGR

    key = cv2.waitKeyEx(5)
    if key == -1:
        # no key press
        continue

    print('key:', key)
    
    if key == 27:
        break
    elif key == ord('y'):
        if shift_mod_hack:
            cam_yaw -= 0.5
        else:
            cam_yaw += 0.5
        config.setFloatEnum('mount_ypr', 0, cam_yaw)
        props_json.save(local_config, config)
        shift_mod_hack = False
    elif key == ord('p'):
        if shift_mod_hack:
            cam_pitch -= 0.5
        else:
            cam_pitch += 0.5
        config.setFloatEnum('mount_ypr', 1, cam_pitch)
        props_json.save(local_config, config)
        shift_mod_hack = False
    elif key == ord('r'):
        if shift_mod_hack:
            cam_roll += 0.5
        else:
            cam_roll -= 0.5
        config.setFloatEnum('mount_ypr', 2, cam_roll)
        props_json.save(local_config, config)
        shift_mod_hack = False
    elif key == ord('-'):
        time_shift -= 1.0/60.0
        shift_mod_hack = False
    elif key == ord('+'):
        time_shift += 1.0/60.0
        shift_mod_hack = False
    elif key == 65505 or key == 65506:
        shift_mod_hack = True
        
writer.close()
cv2.destroyAllWindows()

# now run ffmpeg as an external command to combine original audio
# track with new overlay video

# ex: ffmpeg -i opencv.avi -i orig.mov -c copy -map 0:v -map 1:a final.avi

from subprocess import call
result = call(["ffmpeg", "-i", tmp_movie, "-i", args.movie, "-c", "copy", "-map", "0:v", "-map", "1:a", output_movie])
print("ffmpeg result code:", result)
if result == 0 and not args.keep_tmp_movie:
    print("removing temp movie:", tmp_movie)
    os.remove(tmp_movie)
    print("output movie:", output_movie)

