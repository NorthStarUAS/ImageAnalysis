import cv2
import datetime
import ephem                    # dnf install python3-pyephem
import math
import navpy
import numpy as np
import re

from auracore import wgs84

import sys
sys.path.append('../scripts')
from lib import transformations

import airports

# helpful constants
d2r = math.pi / 180.0
r2d = 180.0 / math.pi
mps2kt = 1.94384
kt2mps = 1 / mps2kt
ft2m = 0.3048
m2ft = 1 / ft2m

# color definitions
black = (0, 0, 0)
green2 = (0, 238, 0)
red2 = (0, 0, 238)
royalblue = (225, 105, 65)
medium_orchid = (186, 85, 211)
dark_orchid = (123, 56, 139)
yellow = (50, 255, 255)
dark_yellow = (33, 170, 170)
white = (255, 255, 255)
gray50 = (128, 128, 128)

class HUD:
    def __init__(self, K):
        self.K = K
        self.PROJ = None
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        self.line_width = 1
        self.color = green2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_size = 0.6
        self.render_w = 0
        self.render_h = 0
        self.lla = [0.0, 0.0, 0.0]
        self.time = 0
        self.unixtime = 0
        self.ned = [0.0, 0.0, 0.0]
        self.ned_history = []
        self.ned_last_time = 0.0
        self.grid = []
        self.ref = None
        self.vn = 0.0
        self.ve = 0.0
        self.vd = 0.0
        self.vel_filt = [0.0, 0.0, 0.0]
        self.phi_rad = 0
        self.the_rad = 0
        self.psi_rad = 0
        self.gc_rad = 0
        self.gc_rot = 0
        self.frame = None
        self.airspeed_units = 'kt'
        self.altitude_units = 'ft'
        self.airspeed_kt = 0
        self.altitude_m = 0
        self.ground_m = 0
        self.wind_deg = 0.0
        self.wind_kt = 0.0
        self.flight_mode = 'none'
        self.ap_roll = 0
        self.ap_pitch = 0
        self.ap_hdg = 0
        self.ap_speed = 0
        self.ap_altitude_ft = 0
        self.alpha_rad = 0
        self.beta_rad = 0
        self.filter_vn = 0.0
        self.filter_ve = 0.0
        self.tf_vel = 0.5
        self.pilot_ail = 0.0
        self.pilot_ele = 0.0
        self.pilot_thr = 0.0
        self.pilot_rud = 0.0
        self.act_ail = 0.0
        self.act_ele = 0.0
        self.act_thr = 0.0
        self.act_rud = 0.0
        self.airports = []
        self.features = []
        self.nose_uv = [0, 0]
        self.dg_img = cv2.imread('hdg_hud.png', -1) # load with transparency
        self.shaded_areas = {}
        self.next_event_index = -1
        self.active_events = []
        self.task_id = ''
        self.target_waypoint_idx = 0
        self.route = []
        self.home = {'lon': 0.0, 'lat': 0.0}
        self.circle = {'lon': 0.0, 'lat': 0.0, 'radius': 100}
        self.land = {'heading_deg': 0, 'side': -1, 'turn_radius_m': 75,
                     'extend_final_leg_m': 150, 'glideslope_rad': 0.1 }
        
    def set_render_size(self, w, h):
        self.render_w = w
        self.render_h = h
        
    def set_line_width(self, line_width):
        self.line_width = line_width
        if self.line_width < 1:
            self.line_width = 1

    def set_color(self, color):
        self.color = color
        
    def set_font_size(self, font_size):
        self.font_size = font_size
        if self.font_size < 0.4:
            self.font_size = 0.4

    def set_units(self, airspeed_units, altitude_units):
        self.airspeed_units = airspeed_units
        self.altitude_units = altitude_units
        
    def set_ned_ref(self, lat, lon):
        self.ref = [ lat, lon, 0.0]
        
    def load_airports(self):
        if self.ref:
            self.airports = airports.load('apt.csv', self.ref, 30000)
        else:
            print('no ned ref set, unable to load nearby airports.')

    def set_ground_m(self, ground_m):
        self.ground_m = ground_m
        
    def update_frame(self, frame):
        self.frame = frame

    def update_lla(self, lla):
        self.lla = lla

    def update_time(self, time, unixtime):
        self.time = time
        self.unixtime = unixtime
        print('unix:', unixtime)

    def update_test_index(self, mode, index):
        self.excite_mode = mode
        self.test_index = index

    def update_ned_history(self, ned, seconds):
        if int(self.time) > self.ned_last_time:
            self.ned_last_time = int(self.time)
            self.ned_history.append(ned)
            while len(self.ned_history) > seconds:
                self.ned_history.pop(0)
        
    def update_ned(self, ned, seconds):
        self.ned = ned[:]
        self.update_ned_history(ned, seconds)

    def update_features(self, feature_list):
        self.features = feature_list
        
    def update_proj(self, PROJ):
        self.PROJ = PROJ

    def update_cam_att(self, cam_yaw, cam_pitch, cam_roll):
        self.cam_yaw = cam_yaw
        self.cam_pitch = cam_pitch
        self.cam_roll = cam_roll
        
    def update_vel(self, vn, ve, vd):
        self.vn = vn
        self.ve = ve
        self.vd = vd
        
    def update_att_rad(self, phi_rad, the_rad, psi_rad):
        self.phi_rad = phi_rad
        self.the_rad = the_rad
        self.psi_rad = psi_rad

    def update_airdata(self, airspeed_kt, altitude_m, wind_deg=0, wind_kt=0,
                       alpha_rad=0, beta_rad=0):
        if airspeed_kt >= 0:
            self.airspeed_kt = airspeed_kt
        else:
            self.airspeed_kt = 0
        self.altitude_m = altitude_m
        self.wind_deg = wind_deg
        self.wind_kt = wind_kt
        self.alpha_rad = alpha_rad
        self.beta_rad = beta_rad

    def update_ap(self, flight_mode, ap_roll, ap_pitch, ap_hdg,
                  ap_speed, ap_altitude_ft):
        self.flight_mode = flight_mode
        self.ap_roll = ap_roll
        self.ap_pitch = ap_pitch
        self.ap_hdg = ap_hdg
        self.ap_speed = ap_speed
        self.ap_altitude_ft = ap_altitude_ft

    def update_pilot(self, aileron, elevator, throttle, rudder):
        self.pilot_ail = aileron
        self.pilot_ele = elevator
        self.pilot_thr = throttle
        self.pilot_rud = rudder
        
    def update_act(self, aileron, elevator, throttle, rudder):
        self.act_ail = aileron
        self.act_ele = elevator
        self.act_thr = throttle
        self.act_rud = rudder

    def update_task(self, record):
        if 'ap' in record:
            route_size = record['ap']['route_size']
            if route_size < len(self.route):
                print("Trimming route to size:", route_size)
                self.route = self.route[:route_size]
            elif route_size > len(self.route):
                print("Expanding route to size:", route_size)
                while route_size > len(self.route):
                    self.route.append({'lon': 0.0, 'lat': 0.0})
            #print(record['ap']['wpt_index'])
            if not math.isnan(record['ap']['wpt_index']):
                wp_index = int(record['ap']['wpt_index'])
            else:
                wp_index = 0
            if wp_index < route_size:
                self.target_waypoint_idx = record['ap']['target_waypoint_idx']
                self.route[wp_index]['lon'] = record['ap']['wpt_longitude_deg']
                self.route[wp_index]['lat'] = record['ap']['wpt_latitude_deg']
            elif wp_index == 65534:
                self.circle['lon'] = record['ap']['wpt_longitude_deg']
                self.circle['lat'] = record['ap']['wpt_latitude_deg']
                radius = record['ap']['task_attrib'] / 10.0
                if ( radius > 1.0 ):
                    self.circle['radius'] = radius
                else:
                    self.circle['radius'] = 100.0
            elif wp_index == 65535:
                self.home['lon'] = record['ap']['wpt_longitude_deg']
                self.home['lat'] = record['ap']['wpt_latitude_deg']
            self.task_id = record['ap']['current_task_id']
    
    def update_events(self, events):
        # expire active events
        event_timer = 15
        while len(self.active_events):
            if self.active_events[0]['time'] < self.time - event_timer:
                del self.active_events[0]
            else:
                break
        # find new events since the last update
        if self.next_event_index < 0:
            # find starting index
            i = 0
            while i < len(events) and events[i]['time'] < self.time:
                i += 1
            print('events starting index:', i)
            self.next_event_index = i
        else:
            i = self.next_event_index
        while i < len(events) and events[i]['time'] <= self.time:
            # trim potential enclosing double quotes from message
            if events[i]['message'][0] == '"':
                events[i]['message'] = events[i]['message'][1:]
            if events[i]['message'][-1] == '"':
                events[i]['message'] = events[i]['message'][:-1]
            
            if re.match('camera:', events[i]['message']):
                print('ignoring:', events[i]['message'])
            elif re.match('remote command: executed: \(\d+\) ',
                        events[i]['message']):
                result = re.split('remote command: executed: \(\d+\) ',
                                  events[i]['message'])
                if len(result) > 1:
                    if result[1] == 'hb':
                        # ignore heartbeat events
                        pass
                    else:
                        # the events system uses ',' as a delimeter
                        events[i]['message'] = re.sub(',', ' ', result[1])
                        self.active_events.append(events[i])
                        # decode landing setup events
                        tokens = re.split(' ', events[i]['message'])
                        if tokens[0] == 'set':
                            if tokens[1] == '/task/land/direction':
                                if tokens[2] == 'left':
                                    self.land['side'] = -1.0
                                elif tokens[2] == 'right':
                                    self.land['side'] = 1.0
                            elif tokens[1] == '/task/land/turn_radius_m':
                                self.land['turn_radius_m'] = float(tokens[2])
                            elif tokens[1] == '/task/land/glideslope_deg':
                                self.land['glideslope_rad'] = float(tokens[2]) * d2r
                            elif tokens[1] == '/task/land/extend_final_leg_m':
                                self.land['extend_final_leg_m'] = float(tokens[2])
                            elif tokens[0] == 'task' and tokens[1] == 'land':
                                self.land['heading_deg'] = float(tokens[2])
                
                else:
                    print('problem interpreting event:', events[i]['message'])
            else:           
                self.active_events.append(events[i])
            i += 1
        self.next_event_index = i
        if len(self.active_events):
            print('active events:')
            for e in self.active_events:
                print(' ', e['time'], e['message'])
            
    def compute_sun_moon_ned(self, lon_deg, lat_deg, alt_m, timestamp):
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

        sun_ned = [ math.cos(sun.az) * math.cos(sun.alt),
                    math.sin(sun.az) * math.cos(sun.alt),
                    -math.sin(sun.alt) ]
        moon_ned = [ math.cos(moon.az) * math.cos(moon.alt),
                     math.sin(moon.az) * math.cos(moon.alt),
                     -math.sin(moon.alt) ]

        return sun_ned, moon_ned

    # project from ned coordinates to image uv coordinates using
    # camera K and PROJ matrices
    def project_ned(self, ned):
        uvh = self.K.dot( self.PROJ.dot( [ned[0], ned[1], ned[2], 1.0] ).T )
        if uvh[2] > 0.2:
            uvh /= uvh[2]
            uv = ( int(round(np.squeeze(uvh[0,0]))),
                   int(round(np.squeeze(uvh[1,0]))) )
            return uv
        else:
            return None

    # project from camera 3d coordinates to image uv coordinates using
    # camera K matrix
    def project_xyz(self, v):
        uvh = self.K.dot( [v[0], v[1], v[2]] )
        # print(uvh)
        if uvh[2] > 0.2:
            uvh /= uvh[2]
            uv = ( int(round(uvh[0])),
                   int(round(uvh[1])) )
            # print(uv)
            return uv
        else:
            return None

    # transform roll, pitch offset angles into world coordinates and
    # the project back into image uv coordinates
    def ar_helper(self, q0, a0, a1):
        q1 = transformations.quaternion_from_euler(-a1*d2r, -a0*d2r, 0.0,
                                                   'rzyx')
        q = transformations.quaternion_multiply(q1, q0)
        v = transformations.quaternion_transform(q, [1.0, 0.0, 0.0])
        uv = self.project_ned( [self.ned[0] + v[0],
                                self.ned[1] + v[1],
                                self.ned[2] + v[2]] )
        return uv

    # transform roll, pitch offset angles into camera coordinates and
    # the project back into image uv coordinates
    def cam_helper(self, a0, a1):
        cam2body = np.array( [[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]],
                             dtype=float )
        yaw = self.cam_yaw * d2r
        pitch = self.cam_pitch * d2r
        roll = self.cam_roll * d2r
        body2cam = transformations.quaternion_from_euler( roll, yaw, pitch,
                                                          'rzyx' )
        #print(a0, a1)
        q1 = transformations.quaternion_from_euler( 0.0, -a1*d2r, -a0*d2r,
                                                    'rzyx' )
        q = transformations.quaternion_multiply(q1, body2cam)
        v = transformations.quaternion_transform(q, [0.0, 0.0, 1.0])
        uv = self.project_xyz( v )
        return uv

    def draw_horizon(self):
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
            uv1 = self.project_ned( [self.ned[0] + p1[0],
                                     self.ned[1] + p1[1],
                                     self.ned[2] + p1[2]] )
            uv2 = self.project_ned( [self.ned[0] + p2[0],
                                     self.ned[1] + p2[1],
                                     self.ned[2] + p2[2]] )
            if uv1 != None and uv2 != None:
                cv2.line(self.frame, uv1, uv2, self.color, self.line_width,
                         cv2.LINE_AA)

    def draw_pitch_ladder(self, beta_rad=0.0):
        a1 = 2.0
        a2 = 8.0
        #slide_rad = self.psi_rad - beta_rad
        slide_rad = self.psi_rad
        q0 = transformations.quaternion_about_axis(slide_rad, [0.0, 0.0, -1.0])
        for a0 in range(5,35,5):
            # above horizon

            # right horizontal
            uv1 = self.ar_helper(q0, a0, a1)
            uv2 = self.ar_helper(q0, a0, a2)
            if uv1 != None and uv2 != None:
                cv2.line(self.frame, uv1, uv2, self.color, self.line_width,
                         cv2.LINE_AA)
                du = uv2[0] - uv1[0]
                dv = uv2[1] - uv1[1]
                uv = ( uv1[0] + int(1.25*du), uv1[1] + int(1.25*dv) )
                self.draw_label("%d" % a0, uv, self.font_size, self.line_width)
            # right tick
            uv1 = self.ar_helper(q0, a0-0.5, a1)
            uv2 = self.ar_helper(q0, a0, a1)
            if uv1 != None and uv2 != None:
                cv2.line(self.frame, uv1, uv2, self.color, self.line_width,
                         cv2.LINE_AA)
            # left horizontal
            uv1 = self.ar_helper(q0, a0, -a1)
            uv2 = self.ar_helper(q0, a0, -a2)
            if uv1 != None and uv2 != None:
                cv2.line(self.frame, uv1, uv2, self.color, self.line_width,
                         cv2.LINE_AA)
                du = uv2[0] - uv1[0]
                dv = uv2[1] - uv1[1]
                uv = ( uv1[0] + int(1.25*du), uv1[1] + int(1.25*dv) )
                self.draw_label("%d" % a0, uv, self.font_size, self.line_width)
            # left tick
            uv1 = self.ar_helper(q0, a0-0.5, -a1)
            uv2 = self.ar_helper(q0, a0, -a1)
            if uv1 != None and uv2 != None:
                cv2.line(self.frame, uv1, uv2, self.color, self.line_width,
                         cv2.LINE_AA)

            # below horizon

            # right horizontal
            uv1 = self.ar_helper(q0, -a0, a1)
            uv2 = self.ar_helper(q0, -a0-0.5, a2)
            if uv1 != None and uv2 != None:
                du = uv2[0] - uv1[0]
                dv = uv2[1] - uv1[1]
                for i in range(0,3):
                    tmp1 = (uv1[0] + int(0.375*i*du), uv1[1] + int(0.375*i*dv))
                    tmp2 = (tmp1[0] + int(0.25*du), tmp1[1] + int(0.25*dv))
                    cv2.line(self.frame, tmp1, tmp2, self.color,
                             self.line_width, cv2.LINE_AA)
                uv = ( uv1[0] + int(1.25*du), uv1[1] + int(1.25*dv) )
                self.draw_label("%d" % a0, uv, self.font_size, self.line_width)

            # right tick
            uv1 = self.ar_helper(q0, -a0+0.5, a1)
            uv2 = self.ar_helper(q0, -a0, a1)
            if uv1 != None and uv2 != None:
                cv2.line(self.frame, uv1, uv2, self.color, self.line_width,
                         cv2.LINE_AA)
            # left horizontal
            uv1 = self.ar_helper(q0, -a0, -a1)
            uv2 = self.ar_helper(q0, -a0-0.5, -a2)
            if uv1 != None and uv2 != None:
                du = uv2[0] - uv1[0]
                dv = uv2[1] - uv1[1]
                for i in range(0,3):
                    tmp1 = (uv1[0] + int(0.375*i*du), uv1[1] + int(0.375*i*dv))
                    tmp2 = (tmp1[0] + int(0.25*du), tmp1[1] + int(0.25*dv))
                    cv2.line(self.frame, tmp1, tmp2, self.color,
                             self.line_width, cv2.LINE_AA)
                uv = ( uv1[0] + int(1.25*du), uv1[1] + int(1.25*dv) )
                self.draw_label("%d" % a0, uv, self.font_size, self.line_width)
            # left tick
            uv1 = self.ar_helper(q0, -a0+0.5, -a1)
            uv2 = self.ar_helper(q0, -a0, -a1)
            if uv1 != None and uv2 != None:
                cv2.line(self.frame, uv1, uv2, self.color, self.line_width,
                         cv2.LINE_AA)

    def draw_alpha_beta_marker(self):
        if True or self.alpha_rad == None or self.beta_rad == None:
            return

        q0 = transformations.quaternion_about_axis(self.psi_rad, [0.0, 0.0, -1.0])
        a0 = self.the_rad * r2d
        center = self.ar_helper(q0, a0, 0.0)
        alpha = self.alpha_rad * r2d
        beta = self.beta_rad * r2d
        tmp = self.ar_helper(q0, a0-alpha, beta)
        if tmp != None:
            uv = self.rotate_pt(tmp, center, self.phi_rad)
            if uv != None:
                r1 = int(round(self.render_h / 60))
                r2 = int(round(self.render_h / 30))
                uv1 = (uv[0]+r1, uv[1])
                uv2 = (uv[0]+r2, uv[1])
                uv3 = (uv[0]-r1, uv[1])
                uv4 = (uv[0]-r2, uv[1])
                uv5 = (uv[0], uv[1]-r1)
                uv6 = (uv[0], uv[1]-r2)
                cv2.circle(self.frame, uv, r1, self.color, self.line_width,
                           cv2.LINE_AA)
                cv2.line(self.frame, uv1, uv2, self.color, self.line_width,
                         cv2.LINE_AA)
                cv2.line(self.frame, uv3, uv4, self.color, self.line_width,
                         cv2.LINE_AA)
                cv2.line(self.frame, uv5, uv6, self.color, self.line_width,
                         cv2.LINE_AA)

    def rotate_pt(self, p, center, a):
        if type(p) is list:
            result = []
            for v in p:
                if v != None:
                    x = math.cos(a) * (v[0]-center[0]) - math.sin(a) * (v[1]-center[1]) + center[0]

                    y = math.sin(a) * (v[0]-center[0]) + math.cos(a) * (v[1]-center[1]) + center[1]
                    result.append( (int(round(x)), int(round(y))) )
                else:
                    return None
            return result                
        else:
            if p != None:
                x = math.cos(a) * (p[0]-center[0]) - math.sin(a) * (p[1]-center[1]) + center[0]

                y = math.sin(a) * (p[0]-center[0]) + math.cos(a) * (p[1]-center[1]) + center[1]
                return ( int(round(x)), int(round(y)) )
            else:
                return None

    def draw_vbars(self):
        scale = 12.0
        angle = 20 * d2r
        a1 = scale * math.cos(angle)
        a3 = scale * math.sin(angle)
        a2 = a3 * 0.4

        a0 = -self.the_rad*r2d + self.ap_pitch
        #a0 = 5

        # center point
        nose = self.cam_helper(0.0, 0.0)
        if nose == None:
            return

        # rotation point (about nose)
        # rot = self.ar_helper(self.the_rad*r2d, 0.0)
        # if rot == None:
        #     return
        
        # center point
        tmp = self.cam_helper(a0, 0.0)
        center = self.rotate_pt(tmp, nose, -self.phi_rad + self.ap_roll*d2r)
        if center == None:
            return
        half_width = int(self.line_width*0.5)
        if half_width < 1: half_width = 1
        
        # right vbar
        tmp = [ self.cam_helper(a0-a3, a1),
                self.cam_helper(a0-a3, a1+a2),
                self.cam_helper(a0-(a3-a2), a1+a2) ]
        uv = self.rotate_pt(tmp, nose, -self.phi_rad + self.ap_roll*d2r)
        if uv != None:
            pts1 = np.array([[center, uv[0], uv[1], uv[2]]])
            cv2.fillPoly(self.frame, pts1, medium_orchid)
            cv2.line(self.frame, uv[0], uv[2], dark_orchid, half_width, cv2.LINE_AA)
            cv2.polylines(self.frame, pts1, True, black, half_width, cv2.LINE_AA)

        # left vbar
        tmp = [ self.cam_helper(a0-a3, -a1),
                self.cam_helper(a0-a3, -(a1+a2)),
                self.cam_helper(a0-(a3-a2), -(a1+a2)) ]
        uv = self.rotate_pt(tmp, nose, -self.phi_rad + self.ap_roll*d2r)
        if uv != None:
            pts1 = np.array([[center, uv[0], uv[1], uv[2]]])
            cv2.fillPoly(self.frame, pts1, medium_orchid)
            cv2.line(self.frame, uv[0], uv[2], dark_orchid, half_width, cv2.LINE_AA)
            cv2.polylines(self.frame, pts1, True, black, half_width, cv2.LINE_AA)

    # draw a texture based DG
    def draw_dg(self):
        # resize
        hdg_size = int(round(self.frame.shape[1] * 0.25))
        hdg = cv2.resize(self.dg_img, (hdg_size, hdg_size))
        rows, cols = hdg.shape[:2]

        # rotate for correct heading
        M = cv2.getRotationMatrix2D((cols/2, rows/2), self.psi_rad*r2d, 1)
        hdg = cv2.warpAffine(hdg, M, (cols, rows))

        # crop top
        hdg = hdg[0:int(rows*.7),:]

        # compute some points and dimensions to be used through the
        # rest of this function
        hdg_rows = hdg.shape[0]
        hdg_cols = hdg.shape[1]
        if not self.nose_uv:
            center_col = int(round(self.frame.shape[1] * 0.5))
        else:
            center_col = self.nose_uv[0]
        print('center_col:', center_col)
        row_start = self.frame.shape[0] - hdg_rows - 1
        col_start = center_col - int(round(hdg_cols * 0.5)) - 1
        row_end = row_start + hdg_rows
        col_end = col_start + hdg_cols
        center = (center_col, row_start + int(round(rows*0.5)))
        top = (center_col, row_start)
        size1 = int(round(hdg_rows*0.04))
        size2 = int(round(hdg_rows*0.09))

        # transparent dg face
        self.shaded_areas['dg-face'] = ['circle', center, int(round(hdg_cols * 0.5)) ]
        
        # heading bug
        if self.flight_mode != 'manual':
            bug_rot = self.ap_hdg * d2r - self.psi_rad
            if bug_rot < -math.pi:
                bug_rot += 2*math.pi
            if bug_rot > math.pi:
                bug_rot -= 2*math.pi
            ref1 = (center_col, row_start)
            ref2 = (center_col, row_start + size2)
            uv0 = self.rotate_pt([ref1, ref2], center, bug_rot)
            uv1 = self.rotate_pt([ref1, ref2], center, bug_rot - 10*d2r)
            uv2 = self.rotate_pt([ref1, ref2], center, bug_rot - 5*d2r)
            uv3 = self.rotate_pt([ref1, ref2], center, bug_rot + 5*d2r)
            uv4 = self.rotate_pt([ref1, ref2], center, bug_rot + 10*d2r)
            if uv1 != None and uv2 != None and uv3 != None and uv4 != None:
                pts = np.array([[uv1[0], uv2[0], uv0[0], uv3[0], uv4[0],
                                 uv4[1], uv3[1], uv0[0], uv2[1], uv1[1]]])
                cv2.fillPoly(self.frame, pts, medium_orchid)

        overlay_img = hdg[:,:,:3]   # rgb
        overlay_mask = hdg[:,:,3:]  # alpha

        # inverse mask
        bg_mask = 255 - overlay_mask

        # convert mask to 3 channel for use as weights
        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)

        # do the magic
        #print(row_start, col_start, self.frame.shape)
        face_part = (self.frame[row_start:row_end,col_start:col_end] * (1/255.0)) * (bg_mask * (1/255.0))
        overlay_part = (overlay_img * (1/255.0)) * (overlay_mask * (1/255.0))

        dst = np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
        self.frame[row_start:row_end,col_start:col_end] = dst

        # center marker (fix this and do similar to other arrow heads)
        # this is a crude hack to get sizing and placement
        arrow1 = (center_col - size1, top[1] - size2)
        arrow2 = (center_col + size1, top[1] - size2)
        cv2.fillPoly(self.frame, np.array([[top, arrow1, arrow2]]), white)

        # ground course indicator
        gs_mps = math.sqrt(self.filter_vn*self.filter_vn + self.filter_ve*self.filter_ve)
        if gs_mps > 0.5:
            self.gc_rad = math.atan2(self.filter_ve, self.filter_vn)
        self.gc_rot = self.gc_rad - self.psi_rad
        if self.gc_rot < -math.pi:
            self.gc_rot += 2*math.pi
        if self.gc_rot > math.pi:
             self.gc_rot -= 2*math.pi
        nose = (center_col, row_start + 1)
        nose1 = (center_col, row_start + size1)
        #end = (self.nose_uv[0], center[1] + size2)
        end = (center_col, center[1])
        arrow1 = (center_col - size1, nose[1] + size2)
        arrow2 = (center_col + size1, nose[1] + size2)
        uv = self.rotate_pt([nose, arrow1, arrow2, nose1, end], center, self.gc_rot)
        if uv != None:
            pts1 = np.array([[uv[3], uv[4]]])
            pts2 = np.array([[uv[0], uv[1], uv[2]]])
            cv2.polylines(self.frame, pts1, False, yellow, int(round(self.line_width*1.5)), cv2.LINE_AA)
            cv2.fillPoly(self.frame, pts2, yellow)

        # wind indicator
        if self.wind_deg != 0 or self.wind_kt != 0:
            wind_rad = self.wind_deg * d2r
            wind_kt = self.wind_kt
            max_wind = self.ap_speed
            if wind_kt > max_wind: wind_kt = max_wind
            wc_rot = wind_rad - self.psi_rad
            if wc_rot < -math.pi:
                wc_rot += 2*math.pi
            if wc_rot > math.pi:
                wc_rot -= 2*math.pi
            size1 = int(round(hdg_rows*0.05))
            size2 = int(round(hdg_rows*0.1))
            size3 = int(round((rows*0.5) * (wind_kt/max_wind)))
            if size3 < size1 + size2:
                size3 = size1 + size2
            nose = (center_col, center[1])
            nose1 = (center_col, center[1] - size1)
            end = (center_col, center[1] - size3)
            arrow1 = (nose[0] - size1, nose[1] - size2)
            arrow2 = (nose[0] + size1, nose[1] - size2)
            uv = self.rotate_pt([nose, arrow1, arrow2, nose1, end], center, wc_rot)
            if uv != None:
                pts1 = np.array([[uv[3], uv[4]]])
                pts2 = np.array([[uv[0], uv[1], uv[2]]])
                cv2.polylines(self.frame, pts1, False, royalblue, int(round(self.line_width*1.5)), cv2.LINE_AA)
                cv2.fillPoly(self.frame, pts2, royalblue)

    def draw_heading_bug(self):
        color = medium_orchid
        size = 2
        a = math.atan2(self.ve, self.vn)
        q0 = transformations.quaternion_about_axis(self.ap_hdg*d2r,
                                                   [0.0, 0.0, -1.0])
        center = self.ar_helper(q0, 0, 0)
        pts = []
        pts.append( self.ar_helper(q0, 0, 2.0) )
        pts.append( self.ar_helper(q0, 0.0, -2.0) )
        pts.append( self.ar_helper(q0, 1.5, -2.0) )
        pts.append( self.ar_helper(q0, 1.5, -1.0) )
        pts.append( center )
        pts.append( self.ar_helper(q0, 1.5, 1.0) )
        pts.append( self.ar_helper(q0, 1.5, 2.0) )
        for i, p in enumerate(pts):
            if p == None or center == None:
                return
        cv2.line(self.frame, pts[0], pts[1], color, self.line_width, cv2.LINE_AA)
        cv2.line(self.frame, pts[1], pts[2], color, self.line_width, cv2.LINE_AA)
        cv2.line(self.frame, pts[2], pts[3], color, self.line_width, cv2.LINE_AA)
        cv2.line(self.frame, pts[3], pts[4], color, self.line_width, cv2.LINE_AA)
        cv2.line(self.frame, pts[4], pts[5], color, self.line_width, cv2.LINE_AA)
        cv2.line(self.frame, pts[5], pts[6], color, self.line_width, cv2.LINE_AA)
        cv2.line(self.frame, pts[6], pts[0], color, self.line_width, cv2.LINE_AA)

    def draw_bird(self):
        scale = 12.0
        angle = 20 * d2r
        a1 = scale * math.cos(angle)
        a3 = scale * math.sin(angle)
        a2 = a3 * 0.5
        a4 = scale * 1.15
        a5 = scale * 0.036

        # center point
        nose = self.nose_uv

        # right bird
        tmp = [ self.cam_helper(-a3, a1),
                self.cam_helper(-a3, a1-a2),
                self.cam_helper(-a3, a1-a3) ]
        uv = self.rotate_pt(tmp, nose, 0.0)
        if uv != None:
            pts1 = np.array([[nose, uv[0], uv[2]]])
            pts2 = np.array([[nose, uv[1], uv[2]]])
            cv2.fillPoly(self.frame, pts1, yellow)
            cv2.fillPoly(self.frame, pts2, dark_yellow)
            cv2.polylines(self.frame, pts1, True, black, int(self.line_width*0.5), cv2.LINE_AA)
                
        # left bird
        tmp = [ self.cam_helper(-a3, -a1),
                self.cam_helper(-a3, -a1+a2),
                self.cam_helper(-a3, -a1+a3) ]
        uv = self.rotate_pt(tmp, nose, 0.0)
        if uv != None:
            pts1 = np.array([[nose, uv[0], uv[2]]])
            pts2 = np.array([[nose, uv[1], uv[2]]])
            cv2.fillPoly(self.frame, pts1, yellow)
            cv2.fillPoly(self.frame, pts2, dark_yellow)
            cv2.polylines(self.frame, pts1, True, black, int(self.line_width*0.5), cv2.LINE_AA)

        a6 = 0.0       # for wing marker
        # a6 = self.the_rad * r2d # for horizon tracking

        # right horizon marker
        tmp = [ self.cam_helper(-a6, a4),
                self.cam_helper(-a6-a5, a4+a5),
                self.cam_helper(-a6-a5, a4+a3),
                self.cam_helper(-a6+a5, a4+a3),
                self.cam_helper(-a6+a5, a4+a5),
                self.cam_helper(-a6, a4+a3) ]
        uv = self.rotate_pt(tmp, nose, 0.0) # for wing marker
        #uv = self.rotate_pt(tmp, nose, -self.phi_rad) # for horizon tracking
        if uv != None:
            pts1 = np.array([[uv[0], uv[1], uv[2], uv[3], uv[4]]])
            pts2 = np.array([[uv[0], uv[5], uv[3], uv[4]]])
            cv2.fillPoly(self.frame, pts1, dark_yellow)
            cv2.fillPoly(self.frame, pts2, yellow)
            cv2.polylines(self.frame, pts1, True, black, int(self.line_width*0.5), cv2.LINE_AA)

        # left horizon marker
        tmp = [ self.cam_helper(-a6, -a4),
                self.cam_helper(-a6-a5, -(a4+a5)),
                self.cam_helper(-a6-a5, -(a4+a3)),
                self.cam_helper(-a6+a5, -(a4+a3)),
                self.cam_helper(-a6+a5, -(a4+a5)),
                self.cam_helper(-a6, -(a4+a3)) ]
        uv = self.rotate_pt(tmp, nose, 0.0) # for wing marker
        #uv = self.rotate_pt(tmp, nose, -self.phi_rad) # for horizon tracking
        if uv != None:
            pts1 = np.array([[uv[0], uv[1], uv[2], uv[3], uv[4]]])
            pts2 = np.array([[uv[0], uv[5], uv[3], uv[4]]])
            cv2.fillPoly(self.frame, pts1, dark_yellow)
            cv2.fillPoly(self.frame, pts2, yellow)
            cv2.polylines(self.frame, pts1, True, black, int(self.line_width*0.5), cv2.LINE_AA)

        return nose

    def draw_roll_indicator_tic(self, nose, a1, angle, length):
        v1x = math.sin(angle*d2r) * a1
        v1y = math.cos(angle*d2r) * a1
        v2x = math.sin(angle*d2r) * (a1 + length)
        v2y = math.cos(angle*d2r) * (a1 + length)
        tmp = [ self.cam_helper(v1y, v1x),
                self.cam_helper(v2y, v2x) ]
        uv = self.rotate_pt(tmp, nose, -self.phi_rad)
        if uv != None:
            cv2.polylines(self.frame, np.array([uv]), False, white, self.line_width, cv2.LINE_AA)

    def draw_roll_indicator(self):
        scale = 12.0
        a1 = scale
        a2 = scale*0.1
        a3 = scale*0.06

        # center point
        nose = self.cam_helper(0.0, 0.0)
        if nose == None:
            return

        # background arc
        tmp = []
        for a in range(-60, 60+1, 5):
            vx = math.sin(a*d2r) * a1
            vy = math.cos(a*d2r) * a1
            tmp.append( self.cam_helper(vy, vx) )
        uv = self.rotate_pt(tmp, nose, -self.phi_rad)
        if uv != None:
            cv2.polylines(self.frame, np.array([uv]), False, white, self.line_width, cv2.LINE_AA)

        self.draw_roll_indicator_tic(nose, a1, -60, a2)
        self.draw_roll_indicator_tic(nose, a1, -30, a2)
        self.draw_roll_indicator_tic(nose, a1, 30, a2)
        self.draw_roll_indicator_tic(nose, a1, 60, a2)
        self.draw_roll_indicator_tic(nose, a1, -45, a3)
        self.draw_roll_indicator_tic(nose, a1, 45, a3)
        self.draw_roll_indicator_tic(nose, a1, -20, a3)
        self.draw_roll_indicator_tic(nose, a1, 20, a3)
        self.draw_roll_indicator_tic(nose, a1, -10, a3)
        self.draw_roll_indicator_tic(nose, a1, 10, a3)

        # center marker
        tmp = [ self.cam_helper(a1, 0),
                self.cam_helper(a1+a2, 0.66),
                self.cam_helper(a1+a2, -0.65) ]
        uv = self.rotate_pt(tmp, nose, -self.phi_rad)
        if uv != None:
            cv2.fillPoly(self.frame, np.array([uv]), white)

        # roll pointer
        tmp = [ self.cam_helper(a1, 0),
                self.cam_helper(a1-a2, 0.66),
                self.cam_helper(a1-a2, -0.65) ]
        uv = self.rotate_pt(tmp, nose, 0.0)
        if uv != None:
            cv2.fillPoly(self.frame, np.array([uv]), white)

            
    def draw_course(self):
        color = yellow
        size = 2
        a = math.atan2(self.filter_ve, self.filter_vn)
        q0 = transformations.quaternion_about_axis(a, [0.0, 0.0, -1.0])
        uv1 = self.ar_helper(q0, 0, 0)
        uv2 = self.ar_helper(q0, 1.5, 1.0)
        uv3 = self.ar_helper(q0, 1.5, -1.0)
        if uv1 != None and uv2 != None and uv3 != None :
            #uv2 = self.rotate_pt(tmp2, tmp1, -self.cam_roll*d2r)
            #uv3 = self.rotate_pt(tmp3, tmp1, -self.cam_roll*d2r)
            cv2.line(self.frame, uv1, uv2, color, self.line_width, cv2.LINE_AA)
            cv2.line(self.frame, uv1, uv3, color, self.line_width, cv2.LINE_AA)

    def draw_label(self, label, uv, font_scale, thickness,
                   horiz='center', vert='center'):
            size = cv2.getTextSize(label, self.font, font_scale, thickness)
            if horiz == 'center':
                u = uv[0] - (size[0][0] / 2)
            else:
                u = uv[0]
            if vert == 'above':
                v = uv[1]
            elif vert == 'below':
                v = uv[1] + size[0][1]
            elif vert == 'center':
                v = uv[1] + (size[0][1] / 2)
            uv = (int(u), int(v))
            cv2.putText(self.frame, label, uv, self.font, font_scale,
                        self.color, thickness, cv2.LINE_AA)

    def draw_ned_point(self, ned, label=None, scale=1, vert='above'):
        uv = self.project_ned([ned[0], ned[1], ned[2]])
        if uv != None:
            cv2.circle(self.frame, uv, 4+self.line_width, self.color,
                       self.line_width, cv2.LINE_AA)
        if label:
            if vert == 'above':
                uv = self.project_ned([ned[0], ned[1], ned[2] - 0.02])
            else:
                uv = self.project_ned([ned[0], ned[1], ned[2] + 0.02])
            if uv != None:
                self.draw_label(label, uv, scale, self.line_width, vert=vert)

    def draw_lla_point(self, lla, label, draw_dist='sm'):
        pt_ned = navpy.lla2ned( lla[0], lla[1], lla[2],
                                self.ref[0], self.ref[1], self.ref[2] )
        rel_ned = [ pt_ned[0] - self.ned[0],
                    pt_ned[1] - self.ned[1],
                    pt_ned[2] - self.ned[2] ]
        hdist = math.sqrt(rel_ned[0]*rel_ned[0] + rel_ned[1]*rel_ned[1])
        dist = math.sqrt(rel_ned[0]*rel_ned[0] + rel_ned[1]*rel_ned[1]
                         + rel_ned[2]*rel_ned[2])
        m2sm = 0.000621371
        hdist_sm = hdist * m2sm
        if hdist_sm <= 10.0:
            scale = 0.9 - (hdist_sm / 10.0) * 0.3
            if hdist_sm <= 7.5:
                if draw_dist == 'm':
                    label += " (%.0f)" % hdist
                elif draw_dist == 'sm':
                    label += " (%.1f)" % hdist_sm
            # normalize, and draw relative to aircraft ned so that label
            # separation works better
            rel_ned[0] /= dist
            rel_ned[1] /= dist
            rel_ned[2] /= dist
            self.draw_ned_point([self.ned[0] + rel_ned[0],
                                 self.ned[1] + rel_ned[1],
                                 self.ned[2] + rel_ned[2]],
                                label, scale=scale, vert='below')

    def draw_compass_points(self):
        # 30 Ticks
        divs = 12
        pts = []
        for i in range(divs):
            a = (float(i) * 360/float(divs)) * d2r
            n = math.cos(a)
            e = math.sin(a)
            uv1 = self.project_ned([self.ned[0] + n,
                                    self.ned[1] + e,
                                    self.ned[2] - 0.0])
            uv2 = self.project_ned([self.ned[0] + n,
                                    self.ned[1] + e,
                                    self.ned[2] - 0.02])
            if uv1 != None and uv2 != None:
                cv2.line(self.frame, uv1, uv2, self.color, self.line_width,
                         cv2.LINE_AA)

        # North
        uv = self.project_ned([self.ned[0] + 1.0, self.ned[1] + 0.0, self.ned[2] - 0.03])
        if uv != None:
            self.draw_label('N', uv, 1, self.line_width, vert='above')
        # South
        uv = self.project_ned([self.ned[0] - 1.0, self.ned[1] + 0.0, self.ned[2] - 0.03])
        if uv != None:
            self.draw_label('S', uv, 1, self.line_width, vert='above')
        # East
        uv = self.project_ned([self.ned[0] + 0.0, self.ned[1] + 1.0, self.ned[2] - 0.03])
        if uv != None:
            self.draw_label('E', uv, 1, self.line_width, vert='above')
        # West
        uv = self.project_ned([self.ned[0] + 0.0, self.ned[1] - 1.0, self.ned[2] - 0.03])
        if uv != None:
            self.draw_label('W', uv, 1, self.line_width, vert='above')

    def draw_astro(self):
        if self.unixtime < 100000.0:
            # bail if it's clear we don't have real world unix time
            return
        
        sun_ned, moon_ned = self.compute_sun_moon_ned(self.lla[1],
                                                      self.lla[0],
                                                      self.lla[2],
                                                      self.unixtime)
        if sun_ned == None or moon_ned == None:
            return

        # Sun
        self.draw_ned_point([self.ned[0] + sun_ned[0],
                             self.ned[1] + sun_ned[1],
                             self.ned[2] + sun_ned[2]],
                            'Sun', scale=1.1)
        # shadow (if sun above horizon)
        if sun_ned[2] < 0.0:
            self.draw_ned_point([self.ned[0] - sun_ned[0],
                                 self.ned[1] - sun_ned[1],
                                 self.ned[2] - sun_ned[2]],
                                'Shadow', scale=0.9)
        # Moon
        self.draw_ned_point([self.ned[0] + moon_ned[0],
                             self.ned[1] + moon_ned[1],
                             self.ned[2] + moon_ned[2]],
                            'Moon', scale=1.1)

    def draw_airports(self):
        for apt in self.airports:
            self.draw_lla_point([ apt[1], apt[2], apt[3] ], apt[0])

    def draw_gate(self, ned1, ned2, ned3, ned4):
        uv1 = self.project_ned(ned1)
        uv2 = self.project_ned(ned2)
        uv3 = self.project_ned(ned3)
        uv4 = self.project_ned(ned4)
        if uv1 != None and uv2 != None and uv3 != None and uv4 != None:
            cv2.line(self.frame, uv1, uv2, white, self.line_width,
                     cv2.LINE_AA)
            cv2.line(self.frame, uv2, uv3, white, self.line_width,
                     cv2.LINE_AA)
            cv2.line(self.frame, uv3, uv4, white, self.line_width,
                     cv2.LINE_AA)
            cv2.line(self.frame, uv4, uv1, white, self.line_width,
                     cv2.LINE_AA)
        
    def draw_task(self):
        # home
        self.draw_lla_point([ self.home['lat'], self.home['lon'], self.ground_m ], "Home", draw_dist='')
        size = 5
        alt = self.ap_altitude_ft * ft2m
        if self.task_id == "circle":
            center_ned = navpy.lla2ned( self.circle['lat'], self.circle['lon'],
                                        alt,
                                        self.ref[0], self.ref[1], self.ref[2] )
            r = self.circle['radius']
            perim = r * math.pi
            step = round(2 * r * math.pi / 30)
            for a in np.linspace(0, 2*math.pi, step, endpoint=False):
                in_e = center_ned[1] + math.cos(a)*(r-size)
                in_n = center_ned[0] + math.sin(a)*(r-size)
                out_e = center_ned[1] + math.cos(a)*(r+size)
                out_n = center_ned[0] + math.sin(a)*(r+size)
                self.draw_gate( [in_n, in_e, center_ned[2]-size],
                                [out_n, out_e, center_ned[2]-size],
                                [out_n, out_e, center_ned[2]+size],
                                [in_n, in_e, center_ned[2]+size] )
        elif self.task_id == "route" and  self.target_waypoint_idx < len(self.route):
            i = self.target_waypoint_idx
            next = self.route[i]
            if i > 0:
                prev = self.route[i-1]
            else:
                prev = self.route[-1]
            # draw target
            self.draw_lla_point([ next['lat'], next['lon'], alt ], "Wpt %d" % i, draw_dist='m')
            # draw boxes
            next_ned = navpy.lla2ned( next['lat'], next['lon'], alt,
                                      self.ref[0], self.ref[1], self.ref[2] )
            prev_ned = navpy.lla2ned( prev['lat'], prev['lon'], alt,
                                      self.ref[0], self.ref[1], self.ref[2] )
            distn = next_ned[0] - prev_ned[0]
            diste = next_ned[1] - prev_ned[1]
            dist = math.sqrt(distn*distn + diste*diste)
            if abs(dist) < 0.0001 or dist > 10000:
                return
            vn = distn / dist
            ve = diste / dist
            uv_list = []
            d = 0
            while d < dist:
                pn = next_ned[0] - d*vn
                pe = next_ned[1] - d*ve
                pd = next_ned[2]
                ln = pn + size*ve
                le = pe - size*vn
                rn = pn - size*ve
                re = pe + size*vn
                self.draw_gate( [ln, le, pd+size], [ln, le, pd-size],
                                [rn, re, pd-size], [rn, re, pd+size] )
                d += 30
        elif self.task_id == "land":
            # target point
            tgt_ned = navpy.lla2ned( self.home['lat'], self.home['lon'],
                                     self.ground_m,
                                     self.ref[0], self.ref[1], self.ref[2] )

            # tangent point
            hdg = (self.land['heading_deg'] + 180) % 360
            final_leg_m = 2.0 * self.land['turn_radius_m'] + self.land['extend_final_leg_m']
            (tan_lat, tan_lon, az2) = \
                wgs84.geo_direct( self.home['lat'], self.home['lon'], hdg, final_leg_m )
            tan_alt = self.ground_m + final_leg_m * math.tan(self.land['glideslope_rad'])
            tan_ned = navpy.lla2ned( tan_lat, tan_lon, tan_alt,
                                     self.ref[0], self.ref[1], self.ref[2] )
                        
            # final approach gates
            distn = tgt_ned[0] - tan_ned[0]
            diste = tgt_ned[1] - tan_ned[1]
            distd = tgt_ned[2] - tan_ned[2]
            dist = math.sqrt(distn*distn + diste*diste + distd*distd)
            if abs(dist) < 0.0001:
                return
            vn = distn / dist
            ve = diste / dist
            vd = distd / dist
            uv_list = []
            if dist > 10000:
                # sanity check
                return
            step = int(dist / 30) + 1
            for d in np.linspace(0, dist, step, endpoint=True):
                pn = tgt_ned[0] - d*vn
                pe = tgt_ned[1] - d*ve
                pd = tgt_ned[2] - d*vd
                ln = pn + size*ve
                le = pe - size*vn
                rn = pn - size*ve
                re = pe + size*vn
                self.draw_gate( [ln, le, pd+size], [ln, le, pd-size],
                                [rn, re, pd-size], [rn, re, pd+size] )
                d += 30

            # circle center
            hdg = (self.land['heading_deg'] + self.land['side'] * 90) % 360
            (cc_lat, cc_lon, az2) = \
                wgs84.geo_direct( tan_lat, tan_lon, hdg, self.land['turn_radius_m'] )
            center_ned = navpy.lla2ned( cc_lat, cc_lon, tan_alt,
                                        self.ref[0], self.ref[1], self.ref[2] )

            # circle gates
            r = self.land['turn_radius_m']
            perim = 2 * r * math.pi
            ha = (90 - self.land['heading_deg']) * d2r
            sa = ha + 0.5 * math.pi * self.land['side']
            ea = sa + math.pi * self.land['side']
            step = round(r * math.pi / 30)
            for a in np.linspace(sa, ea, -self.land['side'] * step,
                                 endpoint=True):
                d = abs(a - sa)
                alt = self.ground_m + (final_leg_m + d*r) * math.tan(self.land['glideslope_rad'])
                #print('d:', d*r, tan_alt, alt)
                in_e = center_ned[1] + math.cos(a)*(r-size)
                in_n = center_ned[0] + math.sin(a)*(r-size)
                out_e = center_ned[1] + math.cos(a)*(r+size)
                out_n = center_ned[0] + math.sin(a)*(r+size)
                self.draw_gate( [in_n, in_e, -alt-size],
                                [out_n, out_e, -alt-size],
                                [out_n, out_e, -alt+size],
                                [in_n, in_e, -alt+size] )

    def draw_nose(self):
        # center point
        nose = self.cam_helper(0.0, 0.0)
        if nose == None:
            return

        r1 = int(round(self.render_h / 80))
        r2 = int(round(self.render_h / 40))
        cv2.circle(self.frame, nose, r1, self.color, self.line_width, cv2.LINE_AA)
        cv2.circle(self.frame, nose, r2, self.color, self.line_width, cv2.LINE_AA)

    def draw_velocity_vector(self):
        tf = 0.2
        vel = [self.vn, self.ve, self.vd] # filter coding convenience
        for i in range(3):
            self.vel_filt[i] = (1.0 - tf) * self.vel_filt[i] + tf * vel[i]

        uv = self.project_ned([self.ned[0] + self.vel_filt[0],
                               self.ned[1] + self.vel_filt[1],
                               self.ned[2] + self.vel_filt[2]])
        if uv != None:
            cv2.circle(self.frame, uv, 5, self.color, self.line_width, cv2.LINE_AA)

    def draw_speed_tape(self, airspeed, ap_speed, units_label):
        size = 1
        pad = 5 + self.line_width*2
        h, w, d = self.frame.shape

        # reference point
        cy = int(h * 0.5)
        cx = int(w * 0.2)
        miny = int(h * 0.2)
        maxy = int(h - miny)

        # use label/font size as a sizing reference
        asi_label = "%.0f" % airspeed
        asi_size = cv2.getTextSize(asi_label, self.font, self.font_size, self.line_width)
        xsize = asi_size[0][0] + pad
        ysize = asi_size[0][1] + pad
        spacing = int(round(asi_size[0][1] * 0.5))

        # transparent background
        self.shaded_areas['speed-tape'] = ['rectangle', (cx-ysize-xsize, miny-int(ysize*0.5)), (cx, maxy+ysize) ]

        # speed bug
        offset = int((ap_speed - airspeed) * spacing)
        if self.flight_mode == 'auto' and cy - offset >= miny and cy - offset <= maxy:
            uv1 = (cx,                  cy - offset)
            uv2 = (cx - int(ysize*0.7), cy - offset - int(ysize / 2) )
            uv3 = (cx - int(ysize*0.7), cy - offset - ysize )
            uv4 = (cx,                  cy - offset - ysize )
            uv5 = (cx,                  cy - offset + ysize )
            uv6 = (cx - int(ysize*0.7), cy - offset + ysize )
            uv7 = (cx - int(ysize*0.7), cy - offset + int(ysize / 2) )
            pts = np.array([[uv1, uv2,  uv3, uv4, uv5, uv6, uv7]])
            cv2.fillPoly(self.frame, pts, medium_orchid)

        # speed tics
        y = cy - int((20 - airspeed) * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv1 = (cx, y)
        y = cy - int((40 - airspeed) * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv2 = (cx, y)
        cv2.line(self.frame, uv1, uv2, green2, self.line_width, cv2.LINE_AA)
        for i in range(0, 65, 1):
            offset = int((i - airspeed) * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                uv1 = (cx, cy - offset)
                if i % 5 == 0:
                    uv2 = (cx - 6, cy - offset)
                else:
                    uv2 = (cx - 4, cy - offset)
                cv2.line(self.frame, uv1, uv2, white, self.line_width, cv2.LINE_AA)
        for i in range(0, 65, 5):
            offset = int((i - airspeed) * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                label = "%d" % i
                lsize = cv2.getTextSize(label, self.font, self.font_size, self.line_width)
                uv3 = (cx - 8 - lsize[0][0], cy - offset + int(lsize[0][1] / 2))
                cv2.putText(self.frame, label, uv3, self.font, self.font_size, white, self.line_width, cv2.LINE_AA)

        # current airspeed
        uv1 = (cx, cy)
        uv2 = (cx - int(ysize*0.7),         int(cy - ysize / 2) )
        uv3 = (cx - int(ysize*0.7) - xsize, int(cy - ysize / 2) )
        uv4 = (cx - int(ysize*0.7) - xsize, int(cy + ysize / 2 + 1) )
        uv5 = (cx - int(ysize*0.7),         int(cy + ysize / 2 + 1) )
        pts = np.array([[uv1, uv2, uv3, uv4, uv5]])
        cv2.fillPoly(self.frame, pts, black)
        cv2.polylines(self.frame, pts, True, white, self.line_width, cv2.LINE_AA)

        #uv = ( int(cx + ysize*0.7), int(cy + lsize[0][1] / 2))
        uv = ( int(cx - ysize*0.7 - asi_size[0][0]), int(cy + asi_size[0][1] / 2))
        cv2.putText(self.frame, asi_label, uv, self.font, self.font_size, white, self.line_width, cv2.LINE_AA)
        
        # units
        lsize = cv2.getTextSize(units_label, self.font, self.font_size, self.line_width)
        uv = ( cx - int((ysize + xsize)*0.5 + lsize[0][1]*0.5), maxy + lsize[0][1] + self.line_width*2)
        cv2.putText(self.frame, units_label, uv, self.font, self.font_size, white, self.line_width, cv2.LINE_AA)

    def draw_altitude_tape(self, altitude, ap_alt, units_label):
        size = 1
        pad = 5 + self.line_width*2
        h, w, d = self.frame.shape

        # reference point
        cy = int(h * 0.5)
        cx = int(w * 0.8)
        miny = int(h * 0.2)
        maxy = int(h - miny)

        minrange = int(altitude/100)*10 - 30
        maxrange = int(altitude/100)*10 + 30

        # current altitude (computed first so we can size all elements)
        alt_label = "%.0f" % (round(altitude/10.0) * 10)
        alt_size = cv2.getTextSize(alt_label, self.font, self.font_size, self.line_width)
        spacing = alt_size[0][1]
        xsize = alt_size[0][0] + pad
        ysize = alt_size[0][1] + pad

        # transparent background
        self.shaded_areas['altitude-tape'] = ['rectangle', (cx+ysize+xsize, miny-int(ysize*0.5)), (cx, maxy+ysize) ]
        
        # altitude bug
        offset = int((ap_alt - altitude)/10.0 * spacing)
        if self.flight_mode == 'auto' and cy - offset >= miny and cy - offset <= maxy:
            uv1 = (cx,                  cy - offset)
            uv2 = (cx + int(ysize*0.7), cy - offset - int(ysize / 2) )
            uv3 = (cx + int(ysize*0.7), cy - offset - ysize )
            uv4 = (cx,                  cy - offset - ysize )
            uv5 = (cx,                  cy - offset + ysize )
            uv6 = (cx + int(ysize*0.7), cy - offset + ysize )
            uv7 = (cx + int(ysize*0.7), cy - offset + int(ysize / 2) )
            pts = np.array([[uv1, uv2,  uv3, uv4, uv5, uv6, uv7]])
            cv2.fillPoly(self.frame, pts, medium_orchid)
            
        # draw ground
        if self.altitude_units == 'm':
            offset = cy - int((self.ground_m - altitude)/10.0 * spacing)
        else:
            offset = cy - int((self.ground_m*m2ft - altitude)/10.0 * spacing)
        if offset >= miny and offset <= maxy:
            bottom = offset + 5 * spacing
            if bottom > maxy:
                bottom = maxy
            uv1 = (cx+2, offset)
            uv2 = (cx+2, bottom)
            cv2.line(self.frame, uv1, uv2, red2, self.line_width*4, cv2.LINE_AA)
        
        # draw max altitude
        if self.altitude_units == 'm':
            offset = cy - int((self.ground_m + 121.92 - altitude)/10.0 * spacing)
        else:
            offset = cy - int((self.ground_m*m2ft + 400.0 - altitude)/10.0 * spacing)
        if offset >= miny and offset <= maxy:
            top = offset - 5 * spacing
            if top < miny:
                top = miny
            uv1 = (cx+2, offset)
            uv2 = (cx+2, top)
            cv2.line(self.frame, uv1, uv2, yellow, self.line_width*4, cv2.LINE_AA)
        # msl tics
        y = cy - int((minrange*10 - altitude)/10 * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv1 = (cx, y)
        y = cy - int((maxrange*10 - altitude)/10 * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv2 = (cx, y)
        #cv2.line(self.frame, uv1, uv2, green2, self.line_width, cv2.LINE_AA)
        for i in range(minrange, maxrange, 1):
            offset = int((i*10 - altitude)/10 * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                uv1 = (cx, cy - offset)
                if i % 5 == 0:
                    uv2 = (cx + 6, cy - offset)
                else:
                    uv2 = (cx + 4, cy - offset)
                cv2.line(self.frame, uv1, uv2, white, self.line_width, cv2.LINE_AA)
        for i in range(minrange, maxrange, 5):
            offset = int((i*10 - altitude)/10 * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                label = "%d" % (i*10)
                lsize = cv2.getTextSize(label, self.font, self.font_size, self.line_width)
                uv3 = (cx + 8 , cy - offset + int(lsize[0][1] / 2))
                cv2.putText(self.frame, label, uv3, self.font, self.font_size, white, self.line_width, cv2.LINE_AA)

        # draw current altitude
        uv1 = (cx, cy)
        uv2 = (cx + int(ysize*0.7),         cy - int(ysize / 2) )
        uv3 = (cx + int(ysize*0.7) + xsize, cy - int(ysize / 2) )
        uv4 = (cx + int(ysize*0.7) + xsize, cy + int(ysize / 2) + 1 )
        uv5 = (cx + int(ysize*0.7),         cy + int(ysize / 2) + 1 )
        pts = np.array([[uv1, uv2, uv3, uv4, uv5]])
        cv2.fillPoly(self.frame, pts, black)
        cv2.polylines(self.frame, pts, True, white, self.line_width, cv2.LINE_AA)

        uv = ( int(cx + ysize*0.7), cy + int(lsize[0][1] / 2))
        cv2.putText(self.frame, alt_label, uv, self.font, self.font_size, white, self.line_width, cv2.LINE_AA)

        # units
        lsize = cv2.getTextSize(units_label, self.font, self.font_size, self.line_width)
        uv = ( cx + int((ysize + xsize)*0.5 - lsize[0][1]*0.5), maxy + lsize[0][1] + self.line_width*2)
        cv2.putText(self.frame, units_label, uv, self.font, self.font_size, white, self.line_width, cv2.LINE_AA)


    # draw stick positions (rc transmitter sticks)
    def draw_sticks(self):
        if self.flight_mode == 'auto':
            aileron = self.act_ail
            elevator = self.act_ele
            throttle = self.act_thr
            rudder = self.act_rud
        else:
            aileron = self.pilot_ail
            elevator = self.pilot_ele
            throttle = self.pilot_thr
            rudder = self.pilot_rud
        h, w, d = self.frame.shape
        lx = int(w * 0.29)
        ly = int(h * 0.85)
        rx = w - int(w * 0.29)
        ry = int(h * 0.85)
        r1 = int(round(h * 0.09))
        if r1 < 10: r1 = 10
        r2 = int(round(h * 0.01))
        if r2 < 2: r2 = 2
        cv2.circle(self.frame, (lx,ly), r1, white, self.line_width,
                   cv2.LINE_AA)
        cv2.line(self.frame, (lx,ly-r1), (lx,ly+r1), white, 1,
                 cv2.LINE_AA)
        cv2.line(self.frame, (lx-r1,ly), (lx+r1,ly), white, 1,
                 cv2.LINE_AA)
        cv2.circle(self.frame, (rx,ry), r1, white, self.line_width,
                   cv2.LINE_AA)
        cv2.line(self.frame, (rx,ry-r1), (rx,ry+r1), white, 1,
                 cv2.LINE_AA)
        cv2.line(self.frame, (rx-r1,ry), (rx+r1,ry), white, 1,
                 cv2.LINE_AA)
        lsx = lx + int(round(rudder * r1))
        lsy = ly + r1 - int(round(2 * throttle * r1))
        cv2.circle(self.frame, (lsx,lsy), r2, white, self.line_width,
                   cv2.LINE_AA)
        rsx = rx + int(round(aileron * r1))
        rsy = ry - int(round(elevator * r1))
        cv2.circle(self.frame, (rsx,rsy), r2, white, self.line_width,
                   cv2.LINE_AA)

    def draw_time(self):
        h, w, d = self.frame.shape
        label = '%.1f' % self.time
        size = cv2.getTextSize(label, self.font, 0.7, self.line_width)
        uv = (2, h - int(size[0][1]*0.5) + 2)
        cv2.putText(self.frame, label, uv, self.font, 0.7,
                    self.color, self.line_width, cv2.LINE_AA)

    def draw_active_events(self):
        h, w, d = self.frame.shape
        ref = 2
        maxw = 0
        for e in self.active_events:
            time = e['time']
            label = "%.1f %s" % (e['time'], e['message'])
            size = cv2.getTextSize(label, self.font, 0.7, self.line_width)
            if size[0][0] > maxw:
                maxw = size[0][0]
            uv = (2, ref + size[0][1])
            ref += size[0][1] + int(size[0][1]*0.3)
            cv2.putText(self.frame, label, uv, self.font, 0.7,
                        white, self.line_width, cv2.LINE_AA)
        self.shaded_areas['events'] = ['rectangle', (0, 0), (maxw+2, ref) ]

    def draw_test_index(self):
        if not hasattr(self, 'excite_mode'):
            return
        if not self.excite_mode:
            return
        h, w, d = self.frame.shape
        label = 'T%d' % self.test_index
        size = cv2.getTextSize(label, self.font, 0.7, self.line_width)
        uv = (w - int(size[0][0]) - 2, h - int(size[0][1]*0.5) + 2)
        cv2.putText(self.frame, label, uv, self.font, 0.7,
                    self.color, self.line_width, cv2.LINE_AA)

    # draw actual flight track in 3d
    def draw_track(self):
        uv_list = []
        dist_list = []
        for ned in self.ned_history:
            dn = self.ned[0] - ned[0]
            de = self.ned[1] - ned[1]
            dd = self.ned[2] - ned[2]
            dist = math.sqrt(dn*dn + de*de + dd*dd)
            dist_list.append(dist)
            if dist > 5:
                uv = self.project_ned([ned[0], ned[1], ned[2]])
            else:
                uv = None
            uv_list.append(uv)
        if len(uv_list) > 1:
            for i in range(len(uv_list) - 1):
                dist = dist_list[i]
                if dist > 0.0:
                    size = int(round(200.0 / dist))
                else:
                    size = 2
                if size < 2: size = 2
                uv1 = uv_list[i]
                uv2 = uv_list[i+1]
                if uv1 != None and uv2 != None:
                    if uv1[0] < -self.render_w * 0.25 and uv2[0] > self.render_w * 1.25:
                        pass
                    elif uv2[0] < -self.render_w * 0.25 and uv1[0] > self.render_w * 1.25:
                        pass
                    elif abs(uv1[0] - uv2[0]) > self.render_w * 1.5:
                        pass
                    elif uv1[1] < -self.render_h * 0.25 and uv2[1] > self.render_h * 1.25:
                        pass
                    elif uv2[1] < -self.render_h * 0.25 and uv1[1] > self.render_h * 1.25:
                        pass
                    elif abs(uv1[1] - uv2[1]) > self.render_h * 1.5:
                        pass
                    else:
                        cv2.line(self.frame, uv1, uv2, white, 1,
                                 cv2.LINE_AA)
                if uv1 != None:
                    cv2.circle(self.frame, uv1, size, white,
                               self.line_width, cv2.LINE_AA)

    # draw externally provided point db features
    def draw_features(self):
        uv_list = []
        for ned in self.features:
            uv = self.project_ned([ned[0], ned[1], ned[2]])
            if uv != None:
                uv_list.append(uv)
        for uv in uv_list:
            size = 2
            if uv[0] > -self.render_w * 0.25 \
               and uv[0] < self.render_w * 1.25 \
               and uv[1] > -self.render_h * 0.25 \
               and uv[1] < self.render_h * 1.25:
                cv2.circle(self.frame, uv, size, white,
                           self.line_width, cv2.LINE_AA)

    # draw a 3d reference grid in space
    def draw_grid(self):
        if len(self.grid) == 0:
            # build the grid
            h = 100
            v = 75
            for n in range(-5*h, 5*h+1, h):
                for e in range(-5*h, 5*h+1, h):
                    for d in range(int(-self.ground_m) - 4*v, int(-self.ground_m) + 1, v):
                        self.grid.append( [n, e, d] )
        uv_list = []
        dist_list = []
        for ned in self.grid:
            dn = self.ned[0] - ned[0]
            de = self.ned[1] - ned[1]
            dd = self.ned[2] - ned[2]
            dist = math.sqrt(dn*dn + de*de + dd*dd)
            dist_list.append(dist)
            uv = self.project_ned( ned )
            uv_list.append(uv)
        for i in range(len(uv_list)):
            dist = dist_list[i]
            size = int(round(1000.0 / dist))
            if size < 1: size = 1
            uv = uv_list[i]
            if uv != None:
                cv2.circle(self.frame, uv, size, white, 1, cv2.LINE_AA)
                    
    # draw the conformal components of the hud (those that should
    # 'stick' to the real world view.
    def draw_conformal(self):
        # things near infinity
        self.draw_horizon()
        self.draw_compass_points()
        self.draw_astro()
        # midrange things
        self.draw_airports()
        self.draw_task()
        self.draw_track()
        self.draw_features()
        # cockpit things
        # currently disabled # self.draw_pitch_ladder(beta_rad=0.0)
        self.draw_alpha_beta_marker()

    # draw the fixed indications (that always stay in the same place
    # on the hud.)  note: also draw speed/alt bugs here
    def draw_fixed(self):
        if self.airspeed_units == 'mps':
            airspeed = self.airspeed_kt * kt2mps
            ap_speed = self.ap_speed * kt2mps
        else:
            airspeed = self.airspeed_kt
            ap_speed = self.ap_speed
        self.draw_speed_tape(airspeed, ap_speed,
                             self.airspeed_units.capitalize())
        if self.altitude_units == 'm':
            altitude = self.altitude_m
            ap_altitude = self.ap_altitude_ft * ft2m
        else:
            altitude = self.altitude_m * m2ft
            ap_altitude = self.ap_altitude_ft
        self.draw_altitude_tape(altitude, ap_altitude,
                                self.altitude_units.capitalize())
        self.draw_dg()
        self.draw_sticks()
        self.draw_time()
        self.draw_active_events()
        self.draw_test_index()

    # draw semi-translucent shaded areas
    def draw_shaded_areas(self):
        color = gray50
        opacity = 0.25
        overlay = self.frame.copy()
        for key in self.shaded_areas:
            area = self.shaded_areas[key]
            if area[0] == 'circle':
                cv2.circle(overlay, area[1], area[2], color, -1)
            elif area[0] == 'rectangle':
                cv2.rectangle(overlay, area[1], area[2], color, -1)
        cv2.addWeighted(overlay, opacity, self.frame, 1 - opacity, 0, self.frame)
    # draw autopilot symbology
    def draw_ap(self):
        if not self.nose_uv:
            return
        
        if self.flight_mode == 'manual':
            self.draw_nose()
        else:
            self.draw_vbars()
            # self.draw_heading_bug() # on horizon
            # self.draw_course()      # on horizon
        self.draw_bird()
        self.draw_roll_indicator()
        self.draw_velocity_vector()

    def draw(self):
        # update the ground vel filter
        self.filter_vn = (1.0 - self.tf_vel) * self.filter_vn + self.tf_vel * self.vn
        self.filter_ve = (1.0 - self.tf_vel) * self.filter_ve + self.tf_vel * self.ve

        # center point
        self.nose_uv = self.cam_helper(0.0, 0.0)

        # draw
        self.draw_conformal()
        self.draw_shaded_areas()
        self.draw_fixed()
        self.draw_ap()
        
