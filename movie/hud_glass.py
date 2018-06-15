import datetime
import ephem                    # dnf install python3-pyephem
import math
import navpy
import numpy as np

# find our custom built opencv first
import sys
sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")
import cv2

sys.path.append('../lib')
import transformations

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
        self.frame = None
        self.airspeed_units = 'kt'
        self.altitude_units = 'ft'
        self.airspeed_kt = 0
        self.altitude_m = 0
        self.ground_m = 0
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

    def update_airdata(self, airspeed_kt, altitude_m, alpha_rad=0, beta_rad=0):
        self.airspeed_kt = airspeed_kt
        self.altitude_m = altitude_m
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

        sun_ned = [ math.cos(sun.az), math.sin(sun.az), -math.sin(sun.alt) ]
        moon_ned = [ math.cos(moon.az), math.sin(moon.az), -math.sin(moon.alt) ]

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
        if self.alpha_rad == None or self.beta_rad == None:
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
        a1 = 14.0
        a2 = 2.0
        a3 = 5.0

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

        # right vbar
        tmp = [ self.cam_helper(a0-a3, a1),
                self.cam_helper(a0-a3, a1+a2),
                self.cam_helper(a0-(a3-a2), a1+a2) ]
        uv = self.rotate_pt(tmp, nose, -self.phi_rad + self.ap_roll*d2r)
        if uv != None:
            pts1 = np.array([[center, uv[0], uv[1], uv[2]]])
            cv2.fillPoly(self.frame, pts1, medium_orchid)
            cv2.line(self.frame, uv[0], uv[2], dark_orchid, int(self.line_width*0.5), cv2.LINE_AA)
            cv2.polylines(self.frame, pts1, True, black, int(self.line_width*0.5), cv2.LINE_AA)

        # left vbar
        tmp = [ self.cam_helper(a0-a3, -a1),
                self.cam_helper(a0-a3, -(a1+a2)),
                self.cam_helper(a0-(a3-a2), -(a1+a2)) ]
        uv = self.rotate_pt(tmp, nose, -self.phi_rad + self.ap_roll*d2r)
        if uv != None:
            pts1 = np.array([[center, uv[0], uv[1], uv[2]]])
            cv2.fillPoly(self.frame, pts1, medium_orchid)
            cv2.line(self.frame, uv[0], uv[2], dark_orchid, int(self.line_width*0.5), cv2.LINE_AA)
            cv2.polylines(self.frame, pts1, True, black, int(self.line_width*0.5), cv2.LINE_AA)

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
        a1 = 14.0
        a2 = 3.0
        a3 = 5.0
        a4 = 18.0
        a5 = 0.5

        # center point
        nose = self.cam_helper(0.0, 0.0)
        if nose == None:
            return

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
                
        # right horizon marker
        tmp = [ self.cam_helper(0.0, a4),
                self.cam_helper(-a5, a4+a5),
                self.cam_helper(-a5, a4+a3),
                self.cam_helper(a5, a4+a3),
                self.cam_helper(a5, a4+a5),
                self.cam_helper(0.0, a4+a3) ]
        uv = self.rotate_pt(tmp, nose, 0.0)
        if uv != None:
            pts1 = np.array([[uv[0], uv[1], uv[2], uv[3], uv[4]]])
            pts2 = np.array([[uv[0], uv[5], uv[3], uv[4]]])
            cv2.fillPoly(self.frame, pts1, dark_yellow)
            cv2.fillPoly(self.frame, pts2, yellow)
            cv2.polylines(self.frame, pts1, True, black, int(self.line_width*0.5), cv2.LINE_AA)

        # left horizon marker
        tmp = [ self.cam_helper(0.0, -a4),
                self.cam_helper(-a5, -(a4+a5)),
                self.cam_helper(-a5, -(a4+a3)),
                self.cam_helper(a5, -(a4+a3)),
                self.cam_helper(a5, -(a4+a5)),
                self.cam_helper(0.0, -(a4+a3)) ]
        uv = self.rotate_pt(tmp, nose, 0.0)
        if uv != None:
            pts1 = np.array([[uv[0], uv[1], uv[2], uv[3], uv[4]]])
            pts2 = np.array([[uv[0], uv[5], uv[3], uv[4]]])
            cv2.fillPoly(self.frame, pts1, dark_yellow)
            cv2.fillPoly(self.frame, pts2, yellow)
            cv2.polylines(self.frame, pts1, True, black, int(self.line_width*0.5), cv2.LINE_AA)

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
        a1 = 16.0

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

        self.draw_roll_indicator_tic(nose, a1, -60, 2)
        self.draw_roll_indicator_tic(nose, a1, -30, 2)
        self.draw_roll_indicator_tic(nose, a1, 30, 2)
        self.draw_roll_indicator_tic(nose, a1, 60, 2)
        self.draw_roll_indicator_tic(nose, a1, -45, 1)
        self.draw_roll_indicator_tic(nose, a1, 45, 1)
        self.draw_roll_indicator_tic(nose, a1, -20, 1)
        self.draw_roll_indicator_tic(nose, a1, 20, 1)
        self.draw_roll_indicator_tic(nose, a1, -10, 1)
        self.draw_roll_indicator_tic(nose, a1, 10, 1)

        # center marker
        tmp = [ self.cam_helper(a1, 0),
                self.cam_helper(a1+1.5, 0.66),
                self.cam_helper(a1+1.5, -0.65) ]
        uv = self.rotate_pt(tmp, nose, -self.phi_rad)
        if uv != None:
            cv2.fillPoly(self.frame, np.array([uv]), white)

        # roll pointer
        tmp = [ self.cam_helper(a1, 0),
                self.cam_helper(a1-1.5, 0.66),
                self.cam_helper(a1-1.5, -0.65) ]
        uv = self.rotate_pt(tmp, nose, 0.0)
        if uv != None:
            cv2.fillPoly(self.frame, np.array([uv]), white)

            
    def draw_course(self):
        color = yellow
        size = 2
        self.filter_vn = (1.0 - self.tf_vel) * self.filter_vn + self.tf_vel * self.vn
        self.filter_ve = (1.0 - self.tf_vel) * self.filter_ve + self.tf_vel * self.ve
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

    def draw_lla_point(self, lla, label):
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
            scale = 0.7 - (hdist_sm / 10.0) * 0.4
            if hdist_sm <= 7.5:
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
                            'Sun')
        # shadow (if sun above horizon)
        if sun_ned[2] < 0.0:
            self.draw_ned_point([self.ned[0] - sun_ned[0],
                                 self.ned[1] - sun_ned[1],
                                 self.ned[2] - sun_ned[2]],
                                'shadow', scale=0.7)
        # Moon
        self.draw_ned_point([self.ned[0] + moon_ned[0],
                             self.ned[1] + moon_ned[1],
                             self.ned[2] + moon_ned[2]],
                            'Moon')

    def draw_airports(self):
        for apt in self.airports:
            self.draw_lla_point([ apt[1], apt[2], apt[3] ], apt[0])

    def draw_nose(self):
        ned2body = transformations.quaternion_from_euler(self.psi_rad,
                                                         self.the_rad,
                                                         self.phi_rad,
                                                         'rzyx')
        body2ned = transformations.quaternion_inverse(ned2body)
        vec = transformations.quaternion_transform(body2ned, [1.0, 0.0, 0.0])
        uv = self.project_ned([self.ned[0] + vec[0],
                               self.ned[1] + vec[1],
                               self.ned[2] + vec[2]])
        r1 = int(round(self.render_h / 80))
        r2 = int(round(self.render_h / 40))
        if uv != None:
            cv2.circle(self.frame, uv, r1, self.color, self.line_width, cv2.LINE_AA)
            cv2.circle(self.frame, uv, r2, self.color, self.line_width, cv2.LINE_AA)

    def draw_velocity_vector(self):
        tf = 0.2
        vel = [self.vn, self.ve, self.vd] # filter coding convenience
        for i in range(3):
            self.vel_filt[i] = (1.0 - tf) * self.vel_filt[i] + tf * vel[i]

        uv = self.project_ned([self.ned[0] + self.vel_filt[0],
                               self.ned[1] + self.vel_filt[1],
                               self.ned[2] + self.vel_filt[2]])
        if uv != None:
            cv2.circle(self.frame, uv, 4, self.color, 1, cv2.LINE_AA)

    def draw_speed_tape(self, airspeed, ap_speed, units_label):
        color = self.color
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
        y = cy - int((0 - airspeed) * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv1 = (cx, y)
        y = cy - int((70 - airspeed) * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv2 = (cx, y)
        cv2.line(self.frame, uv1, uv2, color, self.line_width, cv2.LINE_AA)
        for i in range(0, 65, 1):
            offset = int((i - airspeed) * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                uv1 = (cx, cy - offset)
                if i % 5 == 0:
                    uv2 = (cx - 6, cy - offset)
                else:
                    uv2 = (cx - 4, cy - offset)
                cv2.line(self.frame, uv1, uv2, color, self.line_width, cv2.LINE_AA)
        for i in range(0, 65, 5):
            offset = int((i - airspeed) * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                label = "%d" % i
                lsize = cv2.getTextSize(label, self.font, self.font_size, self.line_width)
                uv3 = (cx - 8 - lsize[0][0], cy - offset + int(lsize[0][1] / 2))
                cv2.putText(self.frame, label, uv3, self.font, self.font_size, color, self.line_width, cv2.LINE_AA)

        # units
        lsize = cv2.getTextSize(units_label, self.font, self.font_size, self.line_width)
        uv = (cx - int(lsize[0][1]*0.5), maxy + lsize[0][1] + self.line_width*2)
        cv2.putText(self.frame, units_label, uv, self.font, self.font_size, color, self.line_width, cv2.LINE_AA)

        # current airspeed
        uv1 = (cx, cy)
        uv2 = (cx - int(ysize*0.7),         int(cy - ysize / 2) )
        uv3 = (cx - int(ysize*0.7) - xsize, int(cy - ysize / 2) )
        uv4 = (cx - int(ysize*0.7) - xsize, int(cy + ysize / 2 + 1) )
        uv5 = (cx - int(ysize*0.7),         int(cy + ysize / 2 + 1) )
        pts = np.array([[uv1, uv2, uv3, uv4, uv5]])
        cv2.fillPoly(self.frame, pts, black)
        cv2.polylines(self.frame, pts, True, color, self.line_width, cv2.LINE_AA)

        #uv = ( int(cx + ysize*0.7), int(cy + lsize[0][1] / 2))
        uv = ( int(cx - ysize*0.7 - asi_size[0][0]), int(cy + asi_size[0][1] / 2))
        cv2.putText(self.frame, asi_label, uv, self.font, self.font_size, color, self.line_width, cv2.LINE_AA)
        
    def draw_altitude_tape(self, altitude, ap_alt, units_label):
        color = self.color
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
            offset = int((self.ground_m - altitude)/10.0 * spacing)
        else:
            offset = int((self.ground_m*m2ft - altitude)/10.0 * spacing)
        if cy - offset >= miny and cy - offset <= maxy:
            uv1 = (cx,                cy - offset)
            uv2 = (cx + int(ysize*3), cy - offset)
            cv2.line(self.frame, uv1, uv2, red2, self.line_width*2, cv2.LINE_AA)
        
        # draw max altitude
        if self.altitude_units == 'm':
            offset = int((self.ground_m + 121.92 - altitude)/10.0 * spacing)
        else:
            offset = int((self.ground_m*m2ft + 400.0 - altitude)/10.0 * spacing)
        if cy - offset >= miny and cy - offset <= maxy:
            uv1 = (cx,                cy - offset)
            uv2 = (cx + int(ysize*2), cy - offset)
            cv2.line(self.frame, uv1, uv2, yellow, self.line_width*2, cv2.LINE_AA)
        # msl tics
        y = cy - int((minrange*10 - altitude)/10 * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv1 = (cx, y)
        y = cy - int((maxrange*10 - altitude)/10 * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv2 = (cx, y)
        cv2.line(self.frame, uv1, uv2, color, self.line_width, cv2.LINE_AA)
        for i in range(minrange, maxrange, 1):
            offset = int((i*10 - altitude)/10 * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                uv1 = (cx, cy - offset)
                if i % 5 == 0:
                    uv2 = (cx + 6, cy - offset)
                else:
                    uv2 = (cx + 4, cy - offset)
                cv2.line(self.frame, uv1, uv2, color, self.line_width, cv2.LINE_AA)
        for i in range(minrange, maxrange, 5):
            offset = int((i*10 - altitude)/10 * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                label = "%d" % (i*10)
                lsize = cv2.getTextSize(label, self.font, self.font_size, self.line_width)
                uv3 = (cx + 8 , cy - offset + int(lsize[0][1] / 2))
                cv2.putText(self.frame, label, uv3, self.font, self.font_size, color, self.line_width, cv2.LINE_AA)

        # units
        lsize = cv2.getTextSize(units_label, self.font, self.font_size, self.line_width)
        uv = (cx - int(lsize[0][1]*0.5), maxy + lsize[0][1] + self.line_width*2)
        cv2.putText(self.frame, units_label, uv, self.font, self.font_size, color, self.line_width, cv2.LINE_AA)

        # draw current altitude
        uv1 = (cx, cy)
        uv2 = (cx + int(ysize*0.7),         cy - int(ysize / 2) )
        uv3 = (cx + int(ysize*0.7) + xsize, cy - int(ysize / 2) )
        uv4 = (cx + int(ysize*0.7) + xsize, cy + int(ysize / 2) + 1 )
        uv5 = (cx + int(ysize*0.7),         cy + int(ysize / 2) + 1 )
        pts = np.array([[uv1, uv2, uv3, uv4, uv5]])
        cv2.fillPoly(self.frame, pts, black)
        cv2.polylines(self.frame, pts, True, color, self.line_width, cv2.LINE_AA)

        uv = ( int(cx + ysize*0.7), cy + int(lsize[0][1] / 2))
        cv2.putText(self.frame, alt_label, uv, self.font, self.font_size, color, self.line_width, cv2.LINE_AA)


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
        lx = int(h * 0.1)
        ly = int(h * 0.8)
        rx = w - int(h * 0.1)
        ry = int(h * 0.8)
        r1 = int(round(h * 0.09))
        if r1 < 10: r1 = 10
        r2 = int(round(h * 0.01))
        if r2 < 2: r2 = 2
        cv2.circle(self.frame, (lx,ly), r1, self.color, self.line_width,
                   cv2.LINE_AA)
        cv2.line(self.frame, (lx,ly-r1), (lx,ly+r1), self.color, 1,
                 cv2.LINE_AA)
        cv2.line(self.frame, (lx-r1,ly), (lx+r1,ly), self.color, 1,
                 cv2.LINE_AA)
        cv2.circle(self.frame, (rx,ry), r1, self.color, self.line_width,
                   cv2.LINE_AA)
        cv2.line(self.frame, (rx,ry-r1), (rx,ry+r1), self.color, 1,
                 cv2.LINE_AA)
        cv2.line(self.frame, (rx-r1,ry), (rx+r1,ry), self.color, 1,
                 cv2.LINE_AA)
        lsx = lx + int(round(rudder * r1))
        lsy = ly + r1 - int(round(2 * throttle * r1))
        cv2.circle(self.frame, (lsx,lsy), r2, self.color, self.line_width,
                   cv2.LINE_AA)
        rsx = rx + int(round(aileron * r1))
        rsy = ry - int(round(elevator * r1))
        cv2.circle(self.frame, (rsx,rsy), r2, self.color, self.line_width,
                   cv2.LINE_AA)

    def draw_time(self):
        h, w, d = self.frame.shape
        label = '%.1f' % self.time
        size = cv2.getTextSize(label, self.font, 0.7, self.line_width)
        uv = (2, h - int(size[0][1]*0.5) + 2)
        cv2.putText(self.frame, label, uv, self.font, 0.7,
                    self.color, self.line_width, cv2.LINE_AA)

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
        self.draw_track()
        self.draw_features()
        # cockpit things
        # currently disabled # self.draw_pitch_ladder(beta_rad=0.0)
        self.draw_alpha_beta_marker()
        self.draw_velocity_vector()

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
        self.draw_sticks()
        self.draw_time()
        self.draw_test_index()

    # draw autopilot symbology
    def draw_ap(self):
        if self.flight_mode == 'manual':
            self.draw_nose()
        else:
            self.draw_vbars()
            self.draw_heading_bug()
            self.draw_bird()
            self.draw_roll_indicator()
            self.draw_course()
        
    def draw(self):
        self.draw_conformal()
        self.draw_fixed()
        self.draw_ap()
        
